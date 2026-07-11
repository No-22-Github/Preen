"""任务3:bf16 版内存与速度探针。

基于 mem_probe_v2 的结构,差异:
  1. --precision fp32|bf16 开关:fp32 用 core patch(基线对照),bf16 用 bf16_patch
  2. 表头打印 device_info()(补丁要求,换机器可比)
  3. trace 里多记 active 内存(三口径完整:active/cache/compressor/ms)
  4. 稳态判定用 active+cache 对比 working_set 上限(削顶判定)

与 v2 报告同条件对比:1.5B nekoqa200 无限档 + c4G 档。
fp32 基线数据已存(tools/mem_v2/15b_nekoqa200_inf.json:cache≥8.97G削顶,2552ms;
15b_scan_c4.json:c4G 1242ms)。

判据(补丁升级):bf16 无限档 active+cache 显著低于 12.07G(如 <10G)→ 脱离天花板真稳态;
仍贴 12G → 省的量不够,D 单独救不动长样本,需与 C 组合。
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# 复用 mem_probe_v2 的数据构造和工具
from mem_probe_v2 import (
    build_fixed_length_samples,
    build_mixed_samples,
    data_stats,
    load_qa_dataset,
)
from statetuner.core import forward_with_state, make_state_params
from statetuner.data import Sample
from statetuner.templates import NEKO_QA
from statetuner.train import TrainConfig, _to_mx_batch, cosine_lr


def compressed_gb() -> float:
    """vm_stat compressor,GB 口径(÷10⁹)。覆盖 mem_probe_v2 的 GiB 版,全仓统一。"""
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        for line in out.splitlines():
            if "Pages occupied by compressor" in line:
                n = int(line.split(":")[-1].strip().rstrip(".").replace(",", ""))
                return n * 16384 / 1e9
    except Exception:
        pass
    return -1.0


def device_header() -> dict:
    """GPU 硬件信息(补丁要求,进表头)。单位 GB(÷10⁹),全仓统一,见 AGENTS.md。"""
    info = {}
    try:
        di = mx.metal.device_info()
        info["max_recommended_working_set_size_gb"] = round(
            di.get("max_recommended_working_set_size", 0) / 1e9, 2)
        info["max_recommended_working_set_size_bytes"] = di.get("max_recommended_working_set_size", 0)
        info["max_buffer_length_gb"] = round(di.get("max_buffer_length", 0) / 1e9, 2)
        info["memory_size_gb"] = round(di.get("memory_size", 0) / 1e9, 2)
    except Exception as e:
        info["error"] = str(e)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--precision", required=True, choices=["fp32", "bf16"])
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--cache-limit-gb", type=float, default=None)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-kind", default="file", choices=["file", "fixed", "mixed"])
    ap.add_argument("--data", default="train_data/NekoQA_10k/NekoQA-10K.json")
    ap.add_argument("--target-len", type=int, default=128)
    ap.add_argument("--trace-every", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)

    if args.cache_limit_gb is not None:
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    dh = device_header()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", file=sys.stderr, flush=True)

    log(f"device: working_set={dh.get('max_recommended_working_set_size_gb','?')}G "
        f"(={dh.get('max_recommended_working_set_size_bytes','?')}B) "
        f"max_buffer={dh.get('max_buffer_length_gb','?')}G "
        f"memory={dh.get('memory_size_gb','?')}G")
    ws_bytes = dh.get("max_recommended_working_set_size_bytes", 0)
    ws_gb = ws_bytes / 1e9 if ws_bytes else 0

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

    # 构造数据(复用 mem_probe_v2)
    if args.data_kind == "fixed":
        samples = build_fixed_length_samples(tok, args.target_len, args.n_samples, args.data)
    elif args.data_kind == "mixed":
        samples = build_mixed_samples(
            tok, 50, 250, 0.05, args.n_samples, args.data)
    else:
        samples = load_qa_dataset(args.data, tok, max_len=512)

    ds = data_stats(samples)
    log(f"data: {args.data_kind} n={ds['n']} mean={ds['mean']} max={ds['max']} p95={ds['p95']}")

    states = make_state_params(mdl, dtype=mx.float32)
    mx.eval(*states.values())
    opt = optim.Adam(learning_rate=args.lr, betas=[0.9, 0.99], eps=1e-8)
    cfg = TrainConfig(lr=args.lr, warmup=10, ctx_len=512)
    total = cfg.total_steps(len(samples))

    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)

    trace = []
    step = 0
    last10_ms = []
    for si in order:
        if step >= args.max_steps:
            break
        batch = _to_mx_batch(samples[si])
        inp, lab, msk = batch
        B = inp.shape[0]
        tlen = inp.shape[1]

        def _loss_fn(sd, inp=inp, lab=lab, msk=msk, B=B):
            logits = forward_with_state(mdl, inp, sd, B)
            lp = nn.log_softmax(logits, -1)
            g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
            return (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)

        opt.learning_rate = cosine_lr(step, total, cfg)
        ts = time.time()
        loss, grads = mx.value_and_grad(_loss_fn)(states)
        grads = {k: mx.clip(g, -cfg.grad_clip, cfg.grad_clip) for k, g in grads.items()}
        states = opt.apply_gradients(grads, states)
        mx.eval(states, loss)
        te = time.time()
        ms = (te - ts) * 1000
        last10_ms.append(ms)
        last10_ms = last10_ms[-10:]

        if step % args.trace_every == 0:
            cache = mx.get_cache_memory() / 1e9
            active = mx.get_active_memory() / 1e9
            rec = {
                "step": step,
                "token_len": tlen,
                "active_gb": round(active, 3),
                "cache_gb": round(cache, 3),
                "active_plus_cache_gb": round(active + cache, 3),
                "compress_gb": compressed_gb(),
                "ms": round(ms, 1),
            }
            trace.append(rec)
            if step % 10 == 0 or step == args.max_steps - 1:
                capped = ""
                if ws_gb and (active + cache) > ws_gb * 0.93:
                    capped = " ⚠️接近削顶"
                log(
                    f"step {step:3d} tlen={tlen:3d} "
                    f"active={active:.2f}G cache={cache:.2f}G "
                    f"sum={active+cache:.2f}G/{ws_gb:.2f}G{capped} "
                    f"compress={rec['compress_gb']:.2f}G {ms:.0f}ms"
                )
        step += 1

    # 稳态判定
    tail = [r["active_plus_cache_gb"] for r in trace[-20:]]
    stable = (max(tail) - min(tail)) < 0.2 if len(tail) >= 5 else False
    last10_avg = sum(last10_ms) / len(last10_ms) if last10_ms else 0
    final_sum = trace[-1]["active_plus_cache_gb"] if trace else 0
    capped = bool(ws_gb and final_sum > ws_gb * 0.93)

    result = {
        "label": args.label,
        "precision": args.precision,
        "model": args.model,
        "data_kind": args.data_kind,
        "target_len": args.target_len,
        "cache_limit_gb": args.cache_limit_gb,
        "data_stats": ds,
        "device": dh,
        "stable": stable,
        "near_capped": capped,
        "stable_active_gb": round(trace[-1]["active_gb"], 2) if trace else 0,
        "stable_cache_gb": round(trace[-1]["cache_gb"], 2) if trace else 0,
        "stable_active_plus_cache_gb": round(final_sum, 2),
        "stable_compress_gb": round(max(r["compress_gb"] for r in trace), 2) if trace else 0,
        "ms_per_step_last10": round(last10_avg, 1),
        "ms_per_step_mean": round(sum(r["ms"] for r in trace) / len(trace), 1) if trace else 0,
        "trace": trace if args.trace_every == 1 else f"(trace_every={args.trace_every}, omit)",
    }
    cap_str = " ⚠️削顶" if capped else ""
    log(
        f"DONE {args.precision} stable={stable} sum(active+cache)={final_sum:.2f}G"
        f"/{ws_gb:.2f}G{cap_str} ms={last10_avg:.0f}"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
