"""c4G 限额下 fp32 vs bf16 步内峰值对照探针(D 裁决最后一块数据)。

与 mem_probe_bf16 的核心差异(per-step peak 增量读数,需求单前置要求):
  每步开始前 mx.reset_peak_memory(),步结束后读 mx.get_peak_memory()——
  拿到的是该步内部的真实活跃峰值,长样本步的反向图尖峰逃不掉。
  旧探针按步边界采样,会漏掉步内反向瞬间的尖峰。

每步记录:(step, token_len, step内peak, cache, active, compressor, ms)
全部 GB(÷1e9),表头带 working_set + commit hash。

支持 --session 双跑:同一进程背靠背跑 bf16→fp32(Task1 要求同 session)。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

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
    """vm_stat compressor,GB(÷1e9),全仓统一。"""
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        for line in out.splitlines():
            if "Pages occupied by compressor" in line:
                n = int(line.split(":")[-1].strip().rstrip(".").replace(",", ""))
                return n * 16384 / 1e9
    except Exception:
        pass
    return -1.0


def device_info_gb() -> dict:
    di = mx.metal.device_info()
    return {
        "max_recommended_working_set_size_gb": round(di.get("max_recommended_working_set_size", 0) / 1e9, 2),
        "max_recommended_working_set_size_bytes": di.get("max_recommended_working_set_size", 0),
        "max_buffer_length_gb": round(di.get("max_buffer_length", 0) / 1e9, 2),
        "memory_size_gb": round(di.get("memory_size", 0) / 1e9, 2),
    }


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "?"


def load_model_for(precision: str, model_path: str):
    if precision == "fp32":
        from statetuner.core import load_model
        return load_model(model_path, patch=True)
    else:
        from bf16_patch import load_model_bf16
        return load_model_bf16(model_path)


def run_single(precision, mdl, samples, order, args, dh):
    """跑一趟训练,返回 trace(每步:step/token_len/step_peak/cache/active/compress/ms)。"""
    states = make_state_params(mdl, dtype=mx.float32)
    mx.eval(*states.values())
    opt = optim.Adam(learning_rate=args.lr, betas=[0.9, 0.99], eps=1e-8)
    cfg = TrainConfig(lr=args.lr, warmup=10, ctx_len=args.ctx_len)
    total = cfg.total_steps(len(samples))

    trace = []
    last10_ms = []
    step = 0
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

        # ★ 关键:步开始前 reset peak,步结束后读 peak = 该步内部真实活跃峰值
        mx.reset_peak_memory()
        ts = time.time()
        loss, grads = mx.value_and_grad(_loss_fn)(states)
        grads = {k: mx.clip(g, -cfg.grad_clip, cfg.grad_clip) for k, g in grads.items()}
        states = opt.apply_gradients(grads, states)
        mx.eval(states, loss)
        te = time.time()

        step_peak = mx.get_peak_memory() / 1e9  # GB,该步内峰值
        cache = mx.get_cache_memory() / 1e9
        active = mx.get_active_memory() / 1e9
        comp = compressed_gb()
        ms = (te - ts) * 1000
        last10_ms.append(ms)
        last10_ms = last10_ms[-10:]

        trace.append({
            "step": step,
            "token_len": tlen,
            "step_peak_gb": round(step_peak, 3),
            "cache_gb": round(cache, 3),
            "active_gb": round(active, 3),
            "active_plus_cache_gb": round(active + cache, 3),
            "compress_gb": round(comp, 3),
            "ms": round(ms, 1),
        })
        if step % 10 == 0 or step == args.max_steps - 1:
            print(
                f"  [{precision}] step {step:3d} tlen={tlen:3d} "
                f"step_peak={step_peak:.2f}G cache={cache:.2f}G "
                f"sum={active+cache:.2f}G compress={comp:.2f}G {ms:.0f}ms",
                file=sys.stderr, flush=True,
            )
        step += 1

    return trace, last10_ms


def summarize(label, precision, trace, last10_ms, dh, data_stats_):
    """汇总单趟结果。"""
    ws_gb = dh["max_recommended_working_set_size_gb"]
    # 找最长样本步
    max_step = max(trace, key=lambda r: r["token_len"])
    # 全程最大 step_peak / active+cache / compressor
    max_peak = max(r["step_peak_gb"] for r in trace)
    max_sum = max(r["active_plus_cache_gb"] for r in trace)
    max_comp = max(r["compress_gb"] for r in trace)
    mean_ms = sum(r["ms"] for r in trace) / len(trace) if trace else 0
    last10 = sum(last10_ms) / len(last10_ms) if last10_ms else 0

    return {
        "label": label,
        "precision": precision,
        "device": dh,
        "data_stats": data_stats_,
        "max_sample_step": {
            "step": max_step["step"], "token_len": max_step["token_len"],
            "step_peak_gb": max_step["step_peak_gb"],
            "active_plus_cache_gb": max_step["active_plus_cache_gb"],
            "compress_gb": max_step["compress_gb"],
        },
        "overall": {
            "max_step_peak_gb": round(max_peak, 3),
            "max_active_plus_cache_gb": round(max_sum, 3),
            "headroom_to_cap_gb": round(ws_gb * 0.95 - max_sum, 3),  # 距削顶线余量
            "max_compress_gb": round(max_comp, 3),
            "ms_per_step_mean": round(mean_ms, 1),
            "ms_per_step_last10": round(last10, 1),
        },
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--cache-limit-gb", type=float, default=4.0, help="默认 c4G")
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-kind", default="file", choices=["file", "fixed", "mixed"])
    ap.add_argument("--data", default="train_data/NekoQA_10k/nekoqa_smoke_200.json")
    # session 模式:同进程背靠背跑两个 precision
    ap.add_argument("--session", nargs="+", default=["bf16", "fp32"],
                    help="背靠背跑的 precision 顺序(默认 bf16→fp32,冷机便宜给主路线)")
    ap.add_argument("--label-prefix", default="", help="产物 label 前缀")
    args = ap.parse_args()

    random.seed(args.seed)
    if args.cache_limit_gb is not None:
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    dh = device_info_gb()
    ch = commit_hash()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", file=sys.stderr, flush=True)

    log(f"═══ peak_probe ═══")
    log(f"working_set={dh['max_recommended_working_set_size_gb']}G "
        f"max_buffer={dh['max_buffer_length_gb']}G commit={ch}")
    log(f"cache_limit={args.cache_limit_gb}G session={args.session}")

    # 数据加载需要 tokenizer,先加载一次模型(kernel 路径)拿 tokenizer,构造数据后丢弃。
    # 之后每趟 session 重新加载对应 precision 的模型(patch 不同)。
    log("loading tokenizer for data...")
    from statetuner.core import load_model
    _, tok = load_model(args.model, patch=False)
    if args.data_kind == "mixed":
        samples = build_mixed_samples(tok, 50, 250, 0.05, args.n_samples, args.data)
    elif args.data_kind == "fixed":
        samples = build_fixed_length_samples(tok, args.ctx_len, args.n_samples, args.data)
    else:
        samples = load_qa_dataset(args.data, tok, max_len=args.ctx_len)

    ds = data_stats(samples)
    log(f"data: n={ds['n']} mean={ds['mean']} max={ds['max']}")
    del tok

    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)

    results = []
    for i, precision in enumerate(args.session):
        log(f"── 跑 {precision}({i+1}/{len(args.session)})──")
        # 每趟重新加载模型(清掉前一趟的 patch 和编译缓存),但数据/order 不变
        mdl, _ = load_model_for(precision, args.model)
        mdl.freeze()
        mx.clear_cache()
        trace, last10_ms = run_single(precision, mdl, samples, order, args, dh)
        label = f"{args.label_prefix}{precision}" if args.label_prefix else precision
        summ = summarize(label, precision, trace, last10_ms, dh, ds)
        summ["commit"] = ch
        results.append(summ)
        log(f"  {precision} done: max_peak={summ['overall']['max_step_peak_gb']}G "
            f"max_sum={summ['overall']['max_active_plus_cache_gb']}G "
            f"headroom={summ['overall']['headroom_to_cap_gb']}G")
        del mdl  # 释放模型,避免两趟叠加
        mx.clear_cache()

    out = {
        "experiment": "peak_probe",
        "config": {
            "model": args.model, "data": args.data, "data_kind": args.data_kind,
            "cache_limit_gb": args.cache_limit_gb, "max_steps": args.max_steps,
            "ctx_len": args.ctx_len, "seed": args.seed,
        },
        "device": dh,
        "commit": ch,
        "runs": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"写入 {args.out}")


if __name__ == "__main__":
    main()
