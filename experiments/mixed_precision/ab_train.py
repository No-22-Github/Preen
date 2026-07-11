"""任务2:训练质量 A/B 的训练脚本。

与 mem_probe_v2 的训练循环结构相同,差异:
  1. --precision fp32|bf16 开关:fp32 用 core.patch_rwkv7_for_train,bf16 用 bf16_patch
  2. 逐 step 记录 loss(写 events.jsonl),供 loss 曲线同图
  3. 训练完保存 state.npz(逐层)供 decode_compare / std 对比 / 距离计算
  4. 表头打印 device_info()(补丁要求,换机器可比)

配置(需求单任务2,复用冒烟配置):
  1.5B + NekoQA 200 条 + 2 epoch + seed42 + lr0.01 + nekoqa 模板 + ctx512 + no early stop
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from statetuner.core import forward_with_state, make_state_params, state_std
from statetuner.data import load_qa_dataset
from statetuner.templates import NEKO_QA
from statetuner.train import TrainConfig, _to_mx_batch, cosine_lr


def device_header() -> dict:
    """GPU 硬件信息(补丁要求,进表头)。单位 GB(÷10⁹),全仓统一,见 AGENTS.md。"""
    info = {}
    try:
        di = mx.metal.device_info()
        info["max_recommended_working_set_size_gb"] = round(di.get("max_recommended_working_set_size", 0) / 1e9, 2)
        info["max_buffer_length_gb"] = round(di.get("max_buffer_length", 0) / 1e9, 2)
        info["memory_size_gb"] = round(di.get("memory_size", 0) / 1e9, 2)
    except Exception as e:
        info["error"] = str(e)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="train_data/NekoQA_10k/nekoqa_smoke_200.json")
    ap.add_argument("--precision", required=True, choices=["fp32", "bf16"],
                    help="fp32=主干 patch,bf16=实验 bf16 patch")
    ap.add_argument("--out-dir", required=True, help="产物目录(state.npz + events.jsonl 写这)")
    ap.add_argument("--label", required=True, help="实验标签(如 fp32_15b / bf16_15b)")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=5, help="每 N step 记一条 loss 进 events")
    args = ap.parse_args()

    random.seed(args.seed)

    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", file=sys.stderr, flush=True)

    # 表头:device info(补丁要求)
    dh = device_header()
    log(f"device: working_set={dh.get('max_recommended_working_set_size_gb','?')}G "
        f"max_buffer={dh.get('max_buffer_length_gb','?')}G "
        f"memory={dh.get('memory_size_gb','?')}G")

    # 按 precision 选 patch
    if args.precision == "fp32":
        from statetuner.core import load_model
        log(f"loading model (fp32, 主干 patch)...")
        mdl, tok = load_model(args.model, patch=True)
    else:
        from bf16_patch import load_model_bf16
        log(f"loading model (bf16, 实验 patch)...")
        mdl, tok = load_model_bf16(args.model)
    mdl.freeze()

    # 数据
    samples = load_qa_dataset(args.data, tok, max_len=args.ctx_len)
    lens = sorted(s.length for s in samples)
    ds = {"n": len(samples), "min": lens[0], "max": lens[-1],
          "mean": round(sum(lens) / len(lens), 1)}
    log(f"data: n={ds['n']} mean={ds['mean']} max={ds['max']}")

    # state(S₀ 始终 fp32 master,即使 bf16 路径也用 fp32 参数)
    states = make_state_params(mdl, dtype=mx.float32)
    mx.eval(*states.values())
    opt = optim.Adam(learning_rate=args.lr, betas=[0.9, 0.99], eps=1e-8)
    cfg = TrainConfig(lr=args.lr, lr_floor=1e-4, warmup=args.warmup,
                      ctx_len=args.ctx_len, epochs=args.epochs, grad_clip=args.grad_clip,
                      early_stop=False, seed=args.seed)
    total = cfg.total_steps(len(samples))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"

    # 逐 step loss 记录(events.jsonl)
    events = []
    meta = {
        "type": "start", "label": args.label, "precision": args.precision,
        "model": args.model, "data": args.data, "data_stats": ds,
        "config": {"lr": args.lr, "epochs": args.epochs, "ctx_len": args.ctx_len,
                   "seed": args.seed, "warmup": args.warmup, "grad_clip": args.grad_clip},
        "device": dh,
    }
    events.append(meta)

    step = 0
    step_times = []
    for epoch in range(args.epochs):
        order = list(range(len(samples)))
        random.Random(args.seed).shuffle(order)
        losses = []
        for si in order:
            batch = _to_mx_batch(samples[si])
            inp, lab, msk = batch
            B = inp.shape[0]

            def _loss_fn(sd, inp=inp, lab=lab, msk=msk, B=B):
                logits = forward_with_state(mdl, inp, sd, B)
                lp = nn.log_softmax(logits, -1)
                g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
                return (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)

            lr = cosine_lr(step, total, cfg)
            opt.learning_rate = lr
            ts = time.time()
            loss, grads = mx.value_and_grad(_loss_fn)(states)
            grads = {k: mx.clip(g, -cfg.grad_clip, cfg.grad_clip) for k, g in grads.items()}
            states = opt.apply_gradients(grads, states)
            mx.eval(states, loss)
            te = time.time()
            step_times.append((te - ts) * 1000)
            loss_f = float(loss)
            losses.append(loss_f)

            if step % args.log_every == 0:
                rec = {"type": "step", "step": step, "epoch": epoch,
                       "loss": round(loss_f, 5), "lr": round(lr, 5)}
                events.append(rec)
                if step % 20 == 0:
                    log(f"epoch{epoch} step {step:3d}/{total} loss={loss_f:.4f} "
                        f"lr={lr:.4f} {(te-ts)*1000:.0f}ms")
            step += 1

        avg_loss = sum(losses) / len(losses)
        sstd = state_std(states)
        last10_ms = sum(step_times[-10:]) / max(1, len(step_times[-10:]))
        ev = {"type": "epoch_end", "epoch": epoch, "avg_loss": round(avg_loss, 5),
              "state_std": round(sstd, 5), "ms_per_step_last10": round(last10_ms, 1)}
        events.append(ev)
        log(f"=== epoch {epoch} end: avg_loss={avg_loss:.4f} std={sstd:.4f} "
            f"ms/step={last10_ms:.0f} ===")

    # 存 state.npz(逐层,P0 内部格式 layer_{i})
    import numpy as np
    state_path = out_dir / "state.npz"
    arrays = {f"layer_{i}": np.array(states[i]) for i in sorted(states)}
    np.savez(state_path, **arrays)

    final = {
        "type": "final",
        "final_loss": round(avg_loss, 5),
        "final_state_std": round(sstd, 5),
        "ms_per_step_mean": round(sum(step_times) / len(step_times), 1),
        "ms_per_step_last10": round(last10_ms, 1),
        "state_path": str(state_path),
        "elapsed_s": round(time.time() - t0, 1),
    }
    events.append(final)

    # 写 events.jsonl
    with open(events_path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    log(f"DONE loss={avg_loss:.4f} std={sstd:.4f} state→{state_path} "
        f"ms/step={final['ms_per_step_mean']:.0f} elapsed={final['elapsed_s']:.0f}s")
    print(json.dumps({"label": args.label, "precision": args.precision, **final,
                      "device": dh}, ensure_ascii=False))


if __name__ == "__main__":
    main()
