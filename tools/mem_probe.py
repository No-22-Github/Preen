"""内存三口径探针 — 复现矩阵测量工具。

复用 Trainer 的训练循环主体,但:
  - 每步后采样三口径(mx.get_peak_memory / mx.get_cache_memory / RSS via ps)
  - 支持 --max-steps (只跑 N 步到稳态)
  - 每个 phase(load_model / make_state_params / value_and_grad / 每步)打点

三口径:
  mx_peak  = mx.get_peak_memory()  — MLX allocator 历史峰值(不含 wired/cache)
  mx_cache = mx.get_cache_memory() — MLX allocator 当前 cache
  mx_active= mx.get_active_memory()— MLX allocator 当前 active
  RSS      = ps 报的进程常驻内存(真实占用,含 Metal wired)

用法:
  PYTHONPATH=src .venv/bin/python tools/mem_probe.py \
    --model models/converted/rwkv7-g1d-0.4b \
    --data train_data/NekoQA_10k/nekoqa_smoke_200.json \
    --ctx-len 512 --max-steps 50 \
    --label A_nekoqa200_ctx512
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from statetuner.core import (
    forward_with_state,
    make_state_params,
    patch_rwkv7_for_train,
    state_std,
)
from statetuner.data import load_qa_dataset
from statetuner.train import TrainConfig, _to_mx_batch, cosine_lr


def rss_mb() -> float:
    """读自己进程的当前 RSS (MB)。用 macOS ps(最可信,含 Metal wired)。"""
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
        ).strip()
        return int(out) / 1024.0  # ps rss 单位 KB → MB
    except Exception:
        return -1.0


def mx3() -> dict:
    """三口径 + active。单位 GB。"""
    return {
        "mx_peak_gb": mx.get_peak_memory() / 1e9,
        "mx_cache_gb": mx.get_cache_memory() / 1e9,
        "mx_active_gb": mx.get_active_memory() / 1e9,
        "rss_gb": rss_mb() / 1024.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", required=True)
    ap.add_argument(
        "--cache-limit-gb",
        type=float,
        default=None,
        help="设 mx.set_cache_limit(GB)。默认不设(用 MLX 默认)。",
    )
    args = ap.parse_args()

    import random

    random.seed(args.seed)

    samples = []
    timings = []  # list of dict
    t0 = time.time()

    def mark(phase, extra=""):
        m = mx3()
        rec = {"t": round(time.time() - t0, 2), "phase": phase, **m}
        if extra:
            rec["note"] = extra
        timings.append(rec)
        print(
            f"[{rec['t']:6.1f}s] {phase:22s} "
            f"peak={m['mx_peak_gb']:.2f}G cache={m['mx_cache_gb']:.2f}G "
            f"active={m['mx_active_gb']:.2f}G RSS={m['rss_gb']:.2f}G "
            f"{extra}",
            file=sys.stderr,
            flush=True,
        )

    mark("start")

    # 可选 cache limit
    if args.cache_limit_gb is not None:
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
        mark(f"set_cache_limit={args.cache_limit_gb}G")

    # 1. load model (patch ops)
    mark("pre_load_model")
    from statetuner.core import load_model

    mdl, tok = load_model(args.model, patch=True)
    mdl.freeze()
    mark("post_load_model")

    # 2. load data
    samples = load_qa_dataset(args.data, tok, max_len=args.ctx_len)
    lens = sorted(s.length for s in samples)
    n = len(lens)
    dlen = {
        "n": n,
        "min": lens[0],
        "p50": lens[n // 2],
        "p95": lens[int(n * 0.95)] if n >= 20 else lens[-1],
        "max": lens[-1],
        "mean": round(sum(lens) / n, 1),
    }
    mark("post_load_data", f"n={n} mean={dlen['mean']} max={dlen['max']}")

    # 3. state params
    states = make_state_params(mdl, dtype=mx.float32)
    mx.eval(*states.values())
    mark("post_make_state")

    # 4. optimizer
    opt = optim.Adam(learning_rate=args.lr, betas=[0.9, 0.99], eps=1e-8)
    cfg = TrainConfig(lr=args.lr, warmup=10, ctx_len=args.ctx_len)
    total = cfg.total_steps(len(samples))
    mark("pre_train")

    # 5. 训练循环 (复刻 Trainer.train 主体, 但每步打点)
    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)
    step = 0
    step_rss = []
    step_peak = []
    step_cache = []
    per_step_ms = []
    for si in order:
        if step >= args.max_steps:
            break
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
        per_step_ms.append((te - ts) * 1000)

        # 每 10 步打一次点 (避免日志爆炸)
        if step % 10 == 0 or step == args.max_steps - 1:
            m = mx3()
            step_rss.append(m["rss_gb"])
            step_peak.append(m["mx_peak_gb"])
            step_cache.append(m["mx_cache_gb"])
            print(
                f"[{time.time()-t0:6.1f}s] step {step:3d} "
                f"loss={float(loss):.3f} "
                f"peak={m['mx_peak_gb']:.2f}G cache={m['mx_cache_gb']:.2f}G "
                f"RSS={m['rss_gb']:.2f}G {(te-ts)*1000:.0f}ms/step",
                file=sys.stderr,
                flush=True,
            )
        step += 1

    # final
    sstd = state_std(states)
    mark("post_train", f"steps={step} std={sstd:.4f}")

    # 强制 eval 全部再读一次峰值
    mx.eval(*states.values())
    mark("final_eval")

    summary = {
        "label": args.label,
        "model": args.model,
        "template": args.template,
        "ctx_len": args.ctx_len,
        "max_steps": args.max_steps,
        "data_len_dist": dlen,
        "per_step_ms_mean": round(sum(per_step_ms) / len(per_step_ms), 1) if per_step_ms else 0,
        "per_step_ms_last10": round(sum(per_step_ms[-10:]) / len(per_step_ms[-10:]), 1) if per_step_ms else 0,
        "rss_trace": step_rss,
        "mx_peak_trace": step_peak,
        "mx_cache_trace": step_cache,
        "timings": timings,
        "final": timings[-1] if timings else {},
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
