"""红线标定探针 — bf16 + c4G 逐桶上探,找 16GB 的样本长度实测上界。

方法:对每个分桶数据集(L450/500/550...),bf16 + c4G 跑 30 步,报:
  三口径稳态(active/cache/active+cache)+ step_peak 最大值 + compressor + ms/step
逐桶上探,直到出现换页/削顶为止。

断点定义(预写,跑完不许改):
  - step_peak 顶到削顶线(95% working_set)= 视为到达上界(安全红线)
  - compressor 持续 >8G = 换页(危险)
  - 单步 >30s = 失控
  满足任一即记"到达断点",该桶长度 = bf16 实测上界。

全绿(所有桶都没到断点)→ 加更长桶继续,直到测到断点。
fp32 不陪跑(红线 ~350 已有实据)。
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from statetuner.core import forward_with_state, make_state_params
from statetuner.data import load_qa_dataset
from statetuner.templates import NEKO_QA
from statetuner.train import TrainConfig, _to_mx_batch, cosine_lr


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


def run_bucket(mdl, samples, args, cap_line, ws_gb):
    """单桶跑 max_steps 步,返回 (max_step_peak, max_active_cache, max_compressor, max_ms, hit_break)。"""
    states = make_state_params(mdl, dtype=mx.float32)
    mx.eval(*states.values())
    opt = optim.Adam(learning_rate=args.lr, betas=[0.9, 0.99], eps=1e-8)
    cfg = TrainConfig(lr=args.lr, warmup=10, ctx_len=args.ctx_len)
    total = cfg.total_steps(len(samples))
    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)

    max_step_peak = max_sum = max_comp = max_ms = 0.0
    comp_sustained_count = 0
    hit_break = None

    for step, si in enumerate(order):
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

        opt.learning_rate = cosine_lr(step, total, cfg)
        mx.reset_peak_memory()
        ts = time.time()
        loss, grads = mx.value_and_grad(_loss_fn)(states)
        grads = {k: mx.clip(g, -cfg.grad_clip, cfg.grad_clip) for k, g in grads.items()}
        states = opt.apply_gradients(grads, states)
        mx.eval(states, loss)
        te = time.time()

        step_peak = mx.get_peak_memory() / 1e9
        cache = mx.get_cache_memory() / 1e9
        active = mx.get_active_memory() / 1e9
        comp = compressed_gb()
        ms = (te - ts) * 1000
        max_step_peak = max(max_step_peak, step_peak)
        max_sum = max(max_sum, active + cache)
        max_comp = max(max_comp, comp)
        max_ms = max(max_ms, ms)
        if comp > 8:
            comp_sustained_count += 1

        # 断点检测(跑完不许改)
        if step_peak > cap_line and hit_break is None:
            hit_break = f"step_peak {step_peak:.2f}G 顶到削顶线 {cap_line:.2f}G(安全红线)"
        if ms > 30000 and hit_break is None:
            hit_break = f"单步 {ms:.0f}ms >30s(失控)"
        # compressor 持续>8G:连续 5 步
        if comp_sustained_count >= 5 and hit_break is None:
            hit_break = f"compressor 持续 >8G(换页)"

        if step % 5 == 0 or step == args.max_steps - 1:
            print(f"    step {step:2d} tlen={inp.shape[1]:3d} peak={step_peak:.2f}G "
                  f"sum={active+cache:.2f}G comp={comp:.2f}G {ms:.0f}ms",
                  file=sys.stderr, flush=True)

    return {"max_step_peak_gb": round(max_step_peak, 3),
            "max_active_plus_cache_gb": round(max_sum, 3),
            "max_compress_gb": round(max_comp, 3),
            "max_ms": round(max_ms, 1),
            "hit_break": hit_break}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--buckets-dir", default="experiments/mixed_precision/data/redline_buckets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--cache-limit-gb", type=float, default=4.0)
    ap.add_argument("--ctx-len", type=int, default=600, help="要容纳最长桶,设大点")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    if args.cache_limit_gb is not None:
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    dh = device_info_gb()
    ch = commit_hash()
    ws_gb = dh["max_recommended_working_set_size_gb"]
    cap_line = ws_gb * 0.95

    t0 = time.time()
    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", file=sys.stderr, flush=True)

    log(f"═══ redline_probe ═══")
    log(f"working_set={ws_gb}G 削顶线(95%)={cap_line:.2f}G commit={ch}")
    log(f"cache_limit={args.cache_limit_gb}G ctx_len={args.ctx_len} precision=bf16")

    # 加载 bf16 模型一次,所有桶共用
    from bf16_patch import load_model_bf16
    log("loading bf16 model...")
    mdl, tok = load_model_bf16(args.model)
    mdl.freeze()

    # 找所有桶文件,按 target 排序(从短到长逐桶上探)
    bdir = Path(args.buckets_dir)
    bucket_files = sorted(bdir.glob("L*.json"))
    if not bucket_files:
        log(f"✗ 没找到桶文件({bdir}/L*.json),先跑 build_buckets.py")
        sys.exit(1)

    results = []
    first_break = None
    for bf in bucket_files:
        log(f"── 桶 {bf.stem} ──")
        samples = load_qa_dataset(str(bf), tok, max_len=args.ctx_len)
        lens = sorted(s.length for s in samples)
        log(f"  {len(samples)} 条,len min={lens[0]} mean={sum(lens)/len(lens):.0f} max={lens[-1]}")
        if len(samples) == 0:
            log(f"  ⚠️ 空桶,跳过")
            continue

        mx.clear_cache()
        r = run_bucket(mdl, samples, args, cap_line, ws_gb)
        r["bucket"] = bf.stem
        r["n_samples"] = len(samples)
        r["token_mean"] = round(sum(lens)/len(lens))
        r["token_max"] = lens[-1]
        results.append(r)
        log(f"  → max_peak={r['max_step_peak_gb']}G max_sum={r['max_active_plus_cache_gb']}G "
            f"max_comp={r['max_compress_gb']}G max_ms={r['max_ms']}ms")
        if r["hit_break"]:
            log(f"  ⚠️ 断点:{r['hit_break']}")
            if first_break is None:
                first_break = r
            log(f"  (已测到断点,继续跑剩余桶记录完整曲线)")

    out = {
        "experiment": "redline_probe", "precision": "bf16",
        "model": args.model, "cache_limit_gb": args.cache_limit_gb,
        "ctx_len": args.ctx_len, "max_steps": args.max_steps,
        "device": dh, "cap_line_gb": round(cap_line, 2), "commit": ch,
        "first_break_bucket": first_break["bucket"] if first_break else None,
        "buckets": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"写入 {args.out}")

    # 结论
    log(f"━━━ 红线标定结论 ━━━")
    if first_break:
        log(f"bf16 + c4G 在 16GB 的实测上界:{first_break['bucket']}")
        log(f"  断点:{first_break['hit_break']}")
    else:
        log(f"⚠️ 所有桶({[r['bucket'] for r in results]})都没测到断点")
        log(f"  需要加更长桶继续上探(全绿不算完成)")


if __name__ == "__main__":
    main()
