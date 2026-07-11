"""内存探针 v2 — 自变量判定 + 稳态判定 + 逐样本追踪。

能力:
  - 长度分档数据集构造 (--data-kind fixed --target-len 128)
  - 混合数据集 (--data-kind mixed --short-len 50 --long-len 250 --long-ratio 0.05)
  - 直接用数据文件 (--data-kind file --data path)
  - 逐样本 token 追踪 (每步记 token_len, cache)
  - 稳态判定: 连续 20 步 cache 变化 < 0.1G 视为稳态
  - cache_limit 设置 (--cache-limit-gb)
  - 三口径 + compressor + ms/step

构造受控长度集的方法 (写明):
  fixed: 从 NekoQA-10K 筛选 token 长度落在 [target-len*0.9, target-len*1.1] 的样本,
         凑满 --n-samples 条 (不够就重复抽样)。得到"近似等长"集。
         注: 编码用真实 NekoQA 样本(token 序列真实), 长度用 max_len=target_len 截断保证上界。
  mixed: 从 10K 筛 short(len<=short-len) 和 long(len>=long-len) 两堆,
         按 long-ratio 混合, mean 被短样本拉低但 max 高。
  file: 直接 load 指定 json (smoke200 等)。

用法见 tools/mem_run_matrix.sh。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from statetuner.core import forward_with_state, make_state_params, load_model
from statetuner.data import Sample, encode_template_sample, load_qa_dataset
from statetuner.templates import NEKO_QA
from statetuner.train import TrainConfig, _to_mx_batch, cosine_lr


def compressed_gb() -> float:
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        for line in out.splitlines():
            if "Pages occupied by compressor" in line:
                n = int(line.split(":")[-1].strip().rstrip(".").replace(",", ""))
                return n * 16384 / (1024**3)
    except Exception:
        pass
    return -1.0


def build_fixed_length_samples(
    tok, target_len: int, n_samples: int, source_path: str
) -> List[Sample]:
    """从 NekoQA-10K 筛选并截断到目标长度的等长集。

    方法: 真实编码所有候选, 取 length 落在 [target*0.85, target*1.15] 的;
    若不够, 放宽到 [target*0.7, target*1.3]; 仍不够则从候选里选最接近的重复。
    最终全部用 max_len=target_len+5 截断(保证上界接近 target, 不超)。
    """
    import json as _json
    import random

    with open(source_path, encoding="utf-8") as f:
        all_items = _json.load(f)

    # 编码全部候选(10K),只取 instruction/output
    cands = []
    rng = random.Random(42)
    rng.shuffle(all_items)
    for it in all_items[:3000]:  # 编码前 3000 个候选,够筛
        q = (it.get("instruction") or "").strip()
        a = (it.get("output") or "").strip()
        if not a:
            continue
        s = encode_template_sample(tok, NEKO_QA, max_len=4096, q=q, a=a)
        cands.append(s)

    # 筛目标区间
    lo, hi = target_len * 0.85, target_len * 1.15
    picked = [s for s in cands if lo <= s.length <= hi]
    if len(picked) < n_samples:
        lo, hi = target_len * 0.7, target_len * 1.3
        picked = [s for s in cands if lo <= s.length <= hi]
    if len(picked) < n_samples:
        # 不够: 重复
        picked = (picked * (n_samples // max(1, len(picked)) + 1))[:n_samples]
    picked = picked[:n_samples]

    # 用 max_len=target_len+5 截断,保证上界
    out = []
    for s in picked:
        s2 = encode_template_sample(
            tok, NEKO_QA, max_len=target_len + 5,
            q=s.cn, a=s.en,
        )
        out.append(s2)
    return out


def build_mixed_samples(
    tok, short_len: int, long_len: int, long_ratio: float, n_samples: int, source_path: str
) -> List[Sample]:
    """混合集: (1-long_ratio) 短样本 + long_ratio 长样本。mean 低, max 高。"""
    import json as _json
    import random

    with open(source_path, encoding="utf-8") as f:
        all_items = _json.load(f)
    rng = random.Random(42)
    rng.shuffle(all_items)

    shorts, longs = [], []
    for it in all_items[:3000]:
        q = (it.get("instruction") or "").strip()
        a = (it.get("output") or "").strip()
        if not a:
            continue
        s = encode_template_sample(tok, NEKO_QA, max_len=4096, q=q, a=a)
        if s.length <= short_len and len(shorts) < n_samples:
            shorts.append(s)
        elif s.length >= long_len and len(longs) < n_samples:
            longs.append(s)
        if len(shorts) >= n_samples and len(longs) >= n_samples:
            break

    n_long = max(1, int(n_samples * long_ratio))
    n_short = n_samples - n_long
    shorts = (shorts * (n_short // max(1, len(shorts)) + 1))[:n_short]
    longs = (longs * (n_long // max(1, len(longs)) + 1))[:n_long]
    mixed = shorts + longs
    rng.shuffle(mixed)
    return mixed


def data_stats(samples: List[Sample]) -> dict:
    lens = sorted(s.length for s in samples)
    n = len(lens)
    return {
        "n": n,
        "min": lens[0],
        "p50": lens[n // 2],
        "p95": lens[int(n * 0.95)] if n >= 20 else lens[-1],
        "max": lens[-1],
        "mean": round(sum(lens) / n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=200, help="数据集大小")
    ap.add_argument("--cache-limit-gb", type=float, default=None)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    # 数据构造
    ap.add_argument(
        "--data-kind",
        default="file",
        choices=["file", "fixed", "mixed"],
    )
    ap.add_argument("--data", default="train_data/NekoQA_10k/NekoQA-10K.json")
    ap.add_argument("--target-len", type=int, default=128)
    ap.add_argument("--short-len", type=int, default=50)
    ap.add_argument("--long-len", type=int, default=250)
    ap.add_argument("--long-ratio", type=float, default=0.05)
    ap.add_argument(
        "--trace-every",
        type=int,
        default=1,
        help="逐样本打点间隔(1=每步)",
    )
    args = ap.parse_args()

    import random

    random.seed(args.seed)

    if args.cache_limit_gb is not None:
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", file=sys.stderr, flush=True)

    log("loading model...")
    mdl, tok = load_model(args.model, patch=True)
    mdl.freeze()

    # 构造数据
    if args.data_kind == "fixed":
        samples = build_fixed_length_samples(
            tok, args.target_len, args.n_samples, args.data
        )
    elif args.data_kind == "mixed":
        samples = build_mixed_samples(
            tok, args.short_len, args.long_len, args.long_ratio,
            args.n_samples, args.data,
        )
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

    trace = []  # 每步: {step, token_len, cache, peak, compress, ms}
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
            rec = {
                "step": step,
                "token_len": tlen,
                "cache_gb": mx.get_cache_memory() / 1e9,
                "peak_gb": mx.get_peak_memory() / 1e9,
                "compress_gb": compressed_gb(),
                "ms": round(ms, 1),
            }
            trace.append(rec)
            if step % 10 == 0 or step == args.max_steps - 1:
                log(
                    f"step {step:3d} tlen={tlen:3d} "
                    f"cache={rec['cache_gb']:.2f}G peak={rec['peak_gb']:.2f}G "
                    f"compress={rec['compress_gb']:.2f}G {ms:.0f}ms"
                )
        step += 1

    # 稳态判定: 末 20 步 cache 极差 < 0.2G
    tail = [r["cache_gb"] for r in trace[-20:]]
    stable = (max(tail) - min(tail)) < 0.2 if len(tail) >= 5 else False
    last10_avg = sum(last10_ms) / len(last10_ms) if last10_ms else 0

    result = {
        "label": args.label,
        "model": args.model,
        "data_kind": args.data_kind,
        "target_len": args.target_len,
        "cache_limit_gb": args.cache_limit_gb,
        "data_stats": ds,
        "stable": stable,
        "stable_cache_gb": round(trace[-1]["cache_gb"], 2) if trace else 0,
        "stable_peak_gb": round(max(r["peak_gb"] for r in trace), 2) if trace else 0,
        "stable_compress_gb": round(max(r["compress_gb"] for r in trace), 2) if trace else 0,
        "ms_per_step_last10": round(last10_avg, 1),
        "ms_per_step_mean": round(sum(r["ms"] for r in trace) / len(trace), 1) if trace else 0,
        "trace": trace if args.trace_every == 1 else f"(trace_every={args.trace_every}, omit)",
    }
    log(
        f"DONE stable={stable} cache={result['stable_cache_gb']}G "
        f"compress={result['stable_compress_gb']}G ms={last10_avg:.0f}"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
