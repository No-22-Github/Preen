"""探针③:int8 组续训判决(lr 重启)。

从现有 int8 checkpoint(state.npz)续训 2 epoch:
  - optimizer 重建(Adam,不带历史动量)
  - lr 重启 0.01 + cos decay(从 step 0 重新走 warmup+cosine)
  - 数据与 seed 不变,其他一切不动
跑完 eval@M,判据(跑前锁死):
  相对基线 1.4034 差 <2%  → a(lr 弹药),方案 E 复活,进第 5 项
  差 2~5%               → 停,再裁
  差 >5%                → b(持续偏置)坐实,封存前允许一发定向修复

口径:
  - int8 量化(与第 3 项同白名单)
  - wkv fp32 ops 路径
  - eval@M:两组 S₀ 挂全精度 M 跑 200 条 eval loss(与 int8_eval.py 同口径)
  - 串行加载,每段清池
"""
from __future__ import annotations
import sys
import json
import time
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "mixed_precision"))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from statetuner.core import (
    load_model, patch_rwkv7_for_train, make_state_params, forward_with_state,
)
from statetuner.data import load_qa_dataset
from statetuner.train import TrainConfig, _to_mx_batch, cosine_lr

MODEL = ROOT / "models" / "converted" / "rwkv7-g1g-1.5b"
DATA = ROOT / "train_data" / "NekoQA_10k" / "nekoqa_smoke_200.json"
TC = ROOT / "experiments" / "mixed_precision" / "data" / "int8_traincompare"
INIT_STATE = TC / "15b_s42_int8" / "state.npz"  # 从第 3 项的 int8 checkpoint 续
OUT_DIR = TC / "15b_s42_int8_resume"
BASELINE_EVAL_M = 1.4034  # 基线 eval@M(int8_eval.py 测得)


def _int8_predicate(path, module):
    if "lora" in path.lower():
        return False
    if isinstance(module, nn.Linear):
        return module.weight.shape[-1] % 64 == 0
    if isinstance(module, nn.Embedding):
        return module.weight.shape[-1] % 64 == 0
    return False


def load_state_npz(path):
    z = np.load(path)
    keys = sorted([k for k in z.files if k.startswith("layer_")],
                  key=lambda x: int(x.split("_")[1]))
    return {int(k.split("_")[1]): mx.array(z[k]) for k in keys}


def eval_loss_on(model, samples, states):
    total_loss = 0.0
    n = 0
    for s in samples:
        inp, lab, msk = _to_mx_batch(s)
        logits = forward_with_state(model, inp, states, 1)
        lp = nn.log_softmax(logits, -1)
        g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
        loss = (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)
        mx.eval(loss)
        total_loss += float(loss)
        n += 1
    return total_loss / n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["int8", "fp32"], default="int8",
                    help="int8=量化续训(方案E), fp32=基线续训(对照)")
    ap.add_argument("--init-state", default=None,
                    help="初始 state.npz 路径(默认按 precision 选)")
    ap.add_argument("--out-suffix", default="",
                    help="输出目录后缀(默认 _resume 或 _resume_fp32)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--cache-limit-gb", type=float, default=4.0)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    # 按 precision 选默认 init_state 和 out_dir
    if args.init_state is None:
        if args.precision == "int8":
            args.init_state = str(TC / "15b_s42_int8" / "state.npz")
        else:
            args.init_state = str(TC / "15b_s42_fp32" / "state.npz")
    suffix = args.out_suffix or ("_resume" if args.precision == "int8" else "_resume_fp32")
    out_dir = TC / f"15b_s42_{args.precision}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_rwkv7_for_train()

    if args.cache_limit_gb:
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    t0 = time.time()
    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", file=sys.stderr, flush=True)

    log(f"═══ 续训 precision={args.precision} (lr 重启) ═══")
    log(f"init_state={args.init_state}")
    log(f"epochs={args.epochs} lr={args.lr}(重启) optimizer=重建Adam")

    # ── 加载模型 + 数据 ──
    log(f"loading model ({args.precision}) + data...")
    mdl, tok = load_model(MODEL, patch=True)
    mdl.freeze()
    if args.precision == "int8":
        nn.quantize(mdl, group_size=64, bits=8, class_predicate=_int8_predicate)
    samples = load_qa_dataset(str(DATA), tok, max_len=args.ctx_len)
    log(f"data: n={len(samples)}")

    # ── 续训:从 checkpoint 读 S₀ ──
    states = load_state_npz(args.init_state)
    std_before = float(np.array(states[0]).std())
    log(f"S₀ loaded from checkpoint, layer0 std={std_before:.5f}")

    # optimizer 重建(不带历史动量)
    opt = optim.Adam(learning_rate=args.lr, betas=[0.9, 0.99], eps=1e-8)
    # lr 重启:从 step 0 走 warmup+cosine
    cfg = TrainConfig(lr=args.lr, lr_floor=1e-4, warmup=args.warmup,
                      ctx_len=args.ctx_len, epochs=args.epochs,
                      grad_clip=args.grad_clip, early_stop=False, seed=42)
    total = cfg.total_steps(len(samples))
    order = list(range(len(samples)))
    random.Random(42).shuffle(order)
    log(f"total steps={total} (lr 重启 warmup={args.warmup})")

    # ── 训练循环(与 matrix_train 同结构)──
    step = 0
    epoch_avg_losses = []
    events = [{"type": "header", "probe": "resume", "init_state": str(INIT_STATE),
               "epochs": args.epochs, "lr": args.lr, "optimizer": "Adam(rebuilt)"}]
    for epoch in range(args.epochs):
        epoch_losses = []
        for si in order:
            batch = _to_mx_batch(samples[si])
            inp, lab, msk = batch
            B = inp.shape[0]

            def _loss_fn(sd, inp=inp, lab=lab, msk=msk, B=B):
                logits = forward_with_state(mdl, inp, sd, B)
                lp = nn.log_softmax(logits, -1)
                g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
                return (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)

            opt.learning_rate = cosine_lr(step, total, cfg)
            loss, grads = mx.value_and_grad(_loss_fn)(states)
            grads = {k: mx.clip(g, -cfg.grad_clip, cfg.grad_clip) for k, g in grads.items()}
            states = opt.apply_gradients(grads, states)
            mx.eval(states, loss)
            epoch_losses.append(float(loss))

            if step % 20 == 0:
                log(f"epoch{epoch} step {step:3d}/{total} loss={float(loss):.4f} lr={float(opt.learning_rate):.5f}")
            step += 1

        ep_avg = sum(epoch_losses) / len(epoch_losses)
        epoch_avg_losses.append(round(ep_avg, 5))
        events.append({"type": "epoch_end", "epoch": epoch, "avg_loss": round(ep_avg, 5)})
        log(f"=== epoch {epoch} end: avg_loss={ep_avg:.5f} ===")

    # ── 存续训后 state ──
    arrays = {f"layer_{i}": np.array(states[i]) for i in sorted(states)}
    np.savez(out_dir / "state.npz", **arrays)
    # 导出 .pth(真机对照用)
    from statetuner.export import export_pth
    states_dict = {i: states[i] for i in sorted(states)}
    export_pth(states_dict, out_dir / "state.pth")
    std_after = float(np.array(states[0]).std())
    log(f"S₀ after resume, layer0 std={std_after:.5f}")

    del mdl
    mx.clear_cache()

    # ── eval@M(判据)──
    log("eval@M (judgment)...")
    mdl_m, _ = load_model(MODEL, patch=True)
    mdl_m.freeze()
    eval_m = eval_loss_on(mdl_m, samples, states)
    del mdl_m
    mx.clear_cache()

    rel_diff_to_base2ep = abs(eval_m - BASELINE_EVAL_M) / BASELINE_EVAL_M

    if args.precision == "int8":
        # int8 续训:对照跑前锁死的 a/b 判据
        if rel_diff_to_base2ep < 0.02:
            verdict = "🟢 a (lr 弹药):方案 E 复活,进第 5 项"
        elif rel_diff_to_base2ep < 0.05:
            verdict = "🟡 停,再裁"
        else:
            verdict = "🔴 b (持续偏置):封存前允许一发定向修复"
        eval_before = 1.5627  # int8 2ep eval@M
    else:
        # 基线续训:只报数,不套 int8 判据(是对照不是判决)
        verdict = "📊 基线续训(对照数据,不判 a/b)"
        eval_before = BASELINE_EVAL_M  # 基线 2ep eval@M

    narrow = (eval_before - eval_m) / eval_before * 100

    print("\n" + "=" * 60)
    print(f"续训 precision={args.precision}")
    print("=" * 60)
    print(f"  续训前 eval@M = {eval_before}")
    print(f"  续训后 eval@M = {eval_m:.5f}")
    print(f"  收窄(相对续训前) = {narrow:+.2f}%")
    print(f"  基线2ep eval@M = {BASELINE_EVAL_M}")
    print(f"  相对基线2ep差   = {rel_diff_to_base2ep*100:+.2f}%")
    print(f"  → {verdict}")

    result = {
        "probe": "resume",
        "precision": args.precision,
        "init_state": args.init_state,
        "epochs": args.epochs,
        "lr_restart": args.lr,
        "optimizer": "Adam(rebuilt, no momentum)",
        "epoch_avg_losses": epoch_avg_losses,
        "eval_m_before_resume": eval_before,
        "eval_m_after_resume": round(eval_m, 5),
        "baseline_2ep_eval_m": BASELINE_EVAL_M,
        "rel_diff_to_baseline_2ep_pct": round(rel_diff_to_base2ep * 100, 3),
        "narrowing_pct": round(narrow, 3),
        "verdict": ("a_lr_ammo" if args.precision == "int8" and rel_diff_to_base2ep < 0.02
                    else "stop" if args.precision == "int8" and rel_diff_to_base2ep < 0.05
                    else "b_persistent_bias" if args.precision == "int8"
                    else "baseline_control"),
        "s0_std_before": round(std_before, 5),
        "s0_std_after": round(std_after, 5),
    }
    (out_dir / "resume_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    events.append({"type": "final", **result})
    with open(out_dir / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    log(f"写入 {out_dir}/")


if __name__ == "__main__":
    main()
