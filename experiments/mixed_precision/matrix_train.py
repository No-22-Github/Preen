"""多 seed 精度矩阵 — 单组训练 + 全套留档(bf16 默认化最后验证)。

合并 ab_train(训练+逐step loss)+ peak_probe(per-step peak 增量读数)+
decode_compare(十问解码)+ export(pth 导出回归)为单入口。

任务2留档清单(缺一该组作废):
  1. 表头: commit / precision / seed / 模型 / 数据指纹 / working_set
  2. events.jsonl: 逐 step loss
  3. state.npz + 导出 .pth(回归验证导出器在两精度产物上都正常)
  4. 逐层 std
  5. 十问贪心 golden 解码输出(json)
  6. 内存: 三口径稳态 + step 内 peak 最大值(增量读数法)
  7. ms/step 均值(只留档不跨组比较)

全程 cache_limit=4G(拟定的产品默认档)。每组独立进程串行跑。
所有内存 GB(÷1e9),表头带 working_set。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from statetuner.core import forward_with_state, make_state_params, state_std, generate
from statetuner.data import load_qa_dataset
from statetuner.templates import NEKO_QA
from statetuner.train import TrainConfig, _to_mx_batch, cosine_lr


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "?"


def device_info_gb() -> dict:
    di = mx.metal.device_info()
    return {
        "max_recommended_working_set_size_gb": round(di.get("max_recommended_working_set_size", 0) / 1e9, 2),
        "max_recommended_working_set_size_bytes": di.get("max_recommended_working_set_size", 0),
        "max_buffer_length_gb": round(di.get("max_buffer_length", 0) / 1e9, 2),
        "memory_size_gb": round(di.get("memory_size", 0) / 1e9, 2),
    }


def compressed_gb() -> float:
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        for line in out.splitlines():
            if "Pages occupied by compressor" in line:
                n = int(line.split(":")[-1].strip().rstrip(".").replace(",", ""))
                return n * 16384 / 1e9
    except Exception:
        pass
    return -1.0


def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _int8_predicate(path, module):
    """int8 量化白名单:Linear + Embedding,排除 LoRA + 不可整除层。"""
    if "lora" in path.lower():
        return False
    if isinstance(module, nn.Linear):
        return module.weight.shape[-1] % 64 == 0
    if isinstance(module, nn.Embedding):
        return module.weight.shape[-1] % 64 == 0
    return False


def load_model_for(precision: str, model_path: str):
    """按 precision 选 loader。

    fp32 → (bf16 权重, wkv fp32 ops)  基线,矩阵报告标"fp32"的那条
    bf16 → (bf16 权重, wkv bf16 ops)  矩阵报告的 bf16 那条
    int8 → (int8 量化权重, wkv fp32 ops)  方案 E,权重 int8 + wkv 仍 fp32
    """
    if precision == "fp32":
        from statetuner.core import load_model
        return load_model(model_path, patch=True)
    elif precision == "bf16":
        from bf16_patch import load_model_bf16
        return load_model_bf16(model_path)
    elif precision == "int8":
        # int8:加载 bf16 模型(走 fp32 wkv ops 路径),再量化权重
        from statetuner.core import load_model
        mdl, tok = load_model(model_path, patch=True)
        nn.quantize(mdl, group_size=64, bits=8, class_predicate=_int8_predicate)
        return mdl, tok
    else:
        raise ValueError(f"unknown precision: {precision}")


def parse_ten_questions(path: str) -> list[str]:
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("REF"):
                continue
            qs.append(line)
    return qs


def is_circular(text: str, window: int = 12, min_repeat: int = 3) -> bool:
    if len(text) < window * min_repeat:
        return False
    tail = text[-(window * min_repeat):]
    seg = tail[:window]
    return all(tail[i * window:(i + 1) * window] == seg for i in range(min_repeat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--precision", required=True, choices=["fp32", "bf16", "int8"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-limit-gb", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--questions", default="experiments/mixed_precision/ten_questions.txt")
    args = ap.parse_args()

    import random
    random.seed(args.seed)

    if args.cache_limit_gb is not None:
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"

    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", file=sys.stderr, flush=True)

    dh = device_info_gb()
    ch = commit_hash()
    dhash = file_hash(args.data)
    log(f"═══ matrix_train ═══")
    log(f"precision={args.precision} seed={args.seed} model={args.model}")
    log(f"commit={ch} working_set={dh['max_recommended_working_set_size_gb']}G")
    log(f"data={args.data} hash={dhash} cache_limit={args.cache_limit_gb}G")

    # ── 加载模型 + 数据 ──
    log("loading model + data...")
    mdl, tok = load_model_for(args.precision, args.model)
    mdl.freeze()
    samples = load_qa_dataset(args.data, tok, max_len=args.ctx_len)
    lens = sorted(s.length for s in samples)
    ds = {"n": len(samples), "min": lens[0], "max": lens[-1], "mean": round(sum(lens) / len(lens), 1)}
    log(f"data: n={ds['n']} mean={ds['mean']} max={ds['max']}")

    # ── 表头事件 ──
    events = []
    events.append({
        "type": "header", "commit": ch, "precision": args.precision, "seed": args.seed,
        "model": args.model, "data_hash": dhash, "data_stats": ds,
        "device": dh, "cache_limit_gb": args.cache_limit_gb,
        "config": {"lr": args.lr, "epochs": args.epochs, "ctx_len": args.ctx_len,
                   "warmup": args.warmup, "grad_clip": args.grad_clip},
    })

    # ── 训练循环(带 per-step peak 增量读数 + cache_limit)──
    states = make_state_params(mdl, dtype=mx.float32)
    mx.eval(*states.values())
    opt = optim.Adam(learning_rate=args.lr, betas=[0.9, 0.99], eps=1e-8)
    cfg = TrainConfig(lr=args.lr, lr_floor=1e-4, warmup=args.warmup, ctx_len=args.ctx_len,
                      epochs=args.epochs, grad_clip=args.grad_clip, early_stop=False, seed=args.seed)
    total = cfg.total_steps(len(samples))
    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)

    step = 0
    step_times = []
    mem_trace = []  # 内存采样(每 log_every 步)
    max_step_peak = 0.0
    stable_active = stable_cache = stable_compressor = 0.0
    epoch_avg_losses = []  # 每 epoch 的真实 avg loss(全 step)
    for epoch in range(args.epochs):
        epoch_losses = []  # 该 epoch 全 step loss
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
            mx.reset_peak_memory()  # per-step peak 增量读数
            ts = time.time()
            loss, grads = mx.value_and_grad(_loss_fn)(states)
            grads = {k: mx.clip(g, -cfg.grad_clip, cfg.grad_clip) for k, g in grads.items()}
            states = opt.apply_gradients(grads, states)
            mx.eval(states, loss)
            te = time.time()
            ms = (te - ts) * 1000
            step_times.append(ms)
            step_peak = mx.get_peak_memory() / 1e9
            max_step_peak = max(max_step_peak, step_peak)
            loss_f = float(loss)
            epoch_losses.append(loss_f)

            if step % args.log_every == 0:
                lr_now = float(opt.learning_rate)
                events.append({"type": "step", "step": step, "epoch": epoch,
                               "loss": round(loss_f, 5), "lr": round(lr_now, 5),
                               "step_peak_gb": round(step_peak, 3),
                               "ms": round(ms, 1)})
                # 内存三口径(步边界采样,不含步内尖峰)
                cache = mx.get_cache_memory() / 1e9
                active = mx.get_active_memory() / 1e9
                comp = compressed_gb()
                stable_active, stable_cache, stable_compressor = active, cache, comp
                mem_trace.append({"step": step, "active_gb": round(active, 3),
                                  "cache_gb": round(cache, 3),
                                  "active_plus_cache_gb": round(active + cache, 3),
                                  "step_peak_gb": round(step_peak, 3),
                                  "compress_gb": round(comp, 3)})
                if step % 20 == 0:
                    log(f"epoch{epoch} step {step:3d}/{total} loss={loss_f:.4f} "
                        f"peak={step_peak:.2f}G {ms:.0f}ms")
            step += 1

        # epoch 末:真实 avg loss(全 step)
        ep_avg = sum(epoch_losses) / len(epoch_losses)
        epoch_avg_losses.append(round(ep_avg, 5))
        events.append({"type": "epoch_end", "epoch": epoch,
                       "avg_loss": round(ep_avg, 5),
                       "state_std_at_epoch": round(state_std(states), 5)})
        log(f"=== epoch {epoch} end: avg_loss={ep_avg:.4f} ===")

    final_epoch_avg = epoch_avg_losses[-1] if epoch_avg_losses else 0

    sstd = state_std(states)
    last10_ms = sum(step_times[-10:]) / max(1, len(step_times[-10:]))
    mean_ms = sum(step_times) / len(step_times)

    # ── 存 state.npz ──
    state_path = out_dir / "state.npz"
    arrays = {f"layer_{i}": np.array(states[i]) for i in sorted(states)}
    np.savez(state_path, **arrays)

    # ── 导出 .pth(回归验证导出器)+ 逐层 std ──
    log("exporting pth + roundtrip verify...")
    from statetuner.export import export_pth, verify_roundtrip
    pth_path = out_dir / "state.pth"
    states_dict = {i: states[i] for i in states}
    export_pth(states_dict, pth_path)
    pth_ok, pth_msg = verify_roundtrip(states_dict, pth_path)

    per_layer_std = {i: round(float(np.array(states[i]).std()), 5) for i in sorted(states)}

    # ── 十问贪心解码 ──
    # 训练模型释放(避免解码时双模型驻留,AGENTS 教训:连续 load 叠峰)
    del mdl
    mx.clear_cache()
    log("decoding ten questions...")
    qs = parse_ten_questions(args.questions)
    from statetuner.core import load_model as _lm

    # int8 组双解码(@M_q 诊断 + @M 判据);其余组只 @M(现状)
    # 裁决(解码模型):配对判据只认 @M;@M_q 只作诊断,用于失败拆账。
    is_int8 = (args.precision == "int8")
    results_on_mq = None
    decode_mem = {}  # 各解码模型的三口径内存(采样一次)

    def _decode_on(mdl_inf, tok_inf):
        rs = []
        for q in qs:
            prompt = NEKO_QA.format_prefix(q=q)
            out = generate(mdl_inf, tok_inf, prompt, state=states_dict, max_tokens=120)
            rs.append({
                "q": q, "out": out,
                "circular": is_circular(out),
                "early_stop": len(out) < 240,  # 粗略:未跑到 max_tokens*2 视为自发终止
                "len": len(out),
            })
        mem = {
            "active_gb": round(mx.get_active_memory() / 1e9, 3),
            "cache_gb": round(mx.get_cache_memory() / 1e9, 3),
            "compress_gb": round(compressed_gb(), 3),
        }
        mem["active_plus_cache_gb"] = round(mem["active_gb"] + mem["cache_gb"], 3)
        return rs, mem

    if is_int8:
        # ① @M_q 诊断(量化模型,训练侧自洽)
        log("  [int8] decoding on M_q (diagnostic)...")
        mdl_mq, tok_mq = _lm(args.model, patch=False)
        nn.quantize(mdl_mq, group_size=64, bits=8, class_predicate=_int8_predicate)
        results_on_mq, decode_mem["M_q"] = _decode_on(mdl_mq, tok_mq)
        del mdl_mq, tok_mq
        mx.clear_cache()
        log(f"    M_q circular={sum(1 for r in results_on_mq if r['circular'])} "
            f"early_stop={sum(1 for r in results_on_mq if r['early_stop'])}/{len(results_on_mq)}")

    # ② @M 判据(全精度模型,部署等价性,配对判据只认这条)
    decode_label = "M (judgment)" if is_int8 else "M"
    log(f"  decoding on {decode_label}...")
    mdl_inf, tok_inf = _lm(args.model, patch=False)
    decode_results, decode_mem["M"] = _decode_on(mdl_inf, tok_inf)
    del mdl_inf, tok_inf
    mx.clear_cache()
    log(f"    M circular={sum(1 for r in decode_results if r['circular'])} "
        f"early_stop={sum(1 for r in decode_results if r['early_stop'])}/{len(decode_results)}")

    # ── 写 events.jsonl ──
    final = {
        "type": "final", "state_std": round(sstd, 5),
        "final_epoch_avg_loss": final_epoch_avg,
        "epoch_avg_losses": epoch_avg_losses,
        "ms_per_step_mean": round(mean_ms, 1), "ms_per_step_last10": round(last10_ms, 1),
        "max_step_peak_gb": round(max_step_peak, 3),
        "stable_active_gb": round(stable_active, 3),
        "stable_cache_gb": round(stable_cache, 3),
        "stable_compressor_gb": round(stable_compressor, 3),
        "stable_active_plus_cache_gb": round(stable_active + stable_cache, 3),
        "pth_export_ok": pth_ok, "pth_msg": pth_msg,
        "per_layer_std": per_layer_std,
        "n_circular": sum(1 for r in decode_results if r["circular"]),
        "n_early_stop": sum(1 for r in decode_results if r["early_stop"]),
        "elapsed_s": round(time.time() - t0, 1),
    }
    events.append(final)

    with open(events_path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ── 十问解码 json ──
    decode_json = {"precision": args.precision, "seed": args.seed,
                   "results": decode_results,
                   "n_circular": final["n_circular"],
                   "n_early_stop": final["n_early_stop"],
                   "decode_mem": decode_mem}
    if results_on_mq is not None:
        decode_json["results_on_mq"] = results_on_mq
        decode_json["n_circular_on_mq"] = sum(1 for r in results_on_mq if r["circular"])
        decode_json["n_early_stop_on_mq"] = sum(1 for r in results_on_mq if r["early_stop"])
    with open(out_dir / "decode.json", "w", encoding="utf-8") as f:
        json.dump(decode_json, f, ensure_ascii=False, indent=2)

    # ── 内存 trace json ──
    with open(out_dir / "mem_trace.json", "w", encoding="utf-8") as f:
        json.dump({"device": dh, "max_step_peak_gb": round(max_step_peak, 3),
                   "stable": {"active_gb": round(stable_active, 3), "cache_gb": round(stable_cache, 3),
                              "compressor_gb": round(stable_compressor, 3),
                              "active_plus_cache_gb": round(stable_active + stable_cache, 3)},
                   "trace": mem_trace}, f, ensure_ascii=False, indent=2)

    # ── summary(单行 stdout,供矩阵脚本收集)──
    summary = {
        "precision": args.precision, "seed": args.seed, "model": args.model,
        "final_epoch_avg_loss": final_epoch_avg,
        "final_state_std": final["state_std"], "max_step_peak_gb": final["max_step_peak_gb"],
        "stable_active_plus_cache_gb": final["stable_active_plus_cache_gb"],
        "pth_ok": pth_ok, "n_circular": final["n_circular"], "n_early_stop": final["n_early_stop"],
        "ms_per_step_mean": final["ms_per_step_mean"],
    }
    log(f"DONE loss={final_epoch_avg:.4f} std={sstd:.4f} peak={max_step_peak:.2f}G "
        f"circ={final['n_circular']}/10 stop={final['n_early_stop']}/10 "
        f"pth={'OK' if pth_ok else 'FAIL'} elapsed={final['elapsed_s']:.0f}s")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
