"""任务1:数值等价性冒烟 — 判定 bf16 递归是否发散。

方法(需求单任务1):
  固定输入,跑单层 _wkv7_step_ops fp32 vs bf16 各 L=273 步(NekoQA smoke200 的 max token),
  逐步记录 y 和 state 的相对误差。报告误差随步数的增长曲线——是平稳还是发散。
  发散则选项 D 直接判死,后续任务不做。

设计:
  不加载完整模型(单层 ops 直接算),避免内存/时间开销。构造确定性输入:
    r,w,k,v ∈ (1, L, H, D),a,b ∈ (1, L, H, D),state ∈ (1, H, D, D)
  用真实量级的随机值(参考推理时的分布:小值,|·| < 1)。
  fp32 路径:全部 fp32(state 也 fp32,模拟当前主干行为)。
  bf16 路径:state 每步进 _wkv7_step_ops 前 cast bf16,模拟 bf16_patch 行为。

  两条路径从同一初始 state 出发,逐步比较。注意:两条路径是各自独立递归
  (各自用自己的 state 更新),不是 bf16 去追 fp32 的轨迹——因为训练时 bf16 版
  从头就用低精度,我们要看的是"它自己的轨迹偏离 fp32 轨迹多远"。

判据:
  - 平稳(误差在某量级震荡或缓慢增长,末值 <1e-1):数值风险可控,过。
  - 缓慢线性增长(末值 1e-1~1):边界,报裁决。
  - 发散(单调增长到 >1,或 nan/inf):D 判死。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.rwkv7 import _wkv7_step_ops


def _rel_err(a: "mx.array", b: "mx.array") -> tuple[float, float, float]:
    """返回 (worst, global, frac_large),三个互补的相对误差口径。

    fp32 参考 b 为准。三个口径各有用途:
      global = ||a-b||_1 / ||b||_1      整体偏离(分母是范数,稳定,主指标)
      worst  = max(|a_i-b_i| / |b_i|)   over 参考值足够大的元素(过滤掉 |b_i|<thr,
                                        否则除以 ~0 是噪声放大不是真误差)
      frac_large = mean(rel_i > 1e-2)   相对误差超 1% 的元素占比(发散早期信号)
    """
    a32 = a.astype(mx.float32)
    b32 = b.astype(mx.float32)
    diff = mx.abs(a32 - b32)
    abs_b = mx.abs(b32)
    # global: L1 范数比(稳定)
    global_rel = float(diff.sum()) / float(mx.maximum(abs_b.sum(), 1e-30).sum())
    # worst: 只看 |b| 足够大的元素(> 其 max 的 1%),避免除 ~0 噪声
    thr = float(abs_b.max()) * 1e-2
    mask = abs_b > thr
    if int(mx.sum(mask)) > 0:
        masked_rel = mx.where(mask, diff / mx.maximum(abs_b, 1e-30), mx.zeros_like(diff))
        worst = float(masked_rel.max())
    else:
        worst = global_rel
    # frac_large: 相对误差 >1% 的元素占比
    rel = diff / mx.maximum(abs_b, 1e-30)
    frac_large = float(mx.sum(rel > 1e-2)) / float(rel.size)
    return worst, global_rel, frac_large


def run_path(L, H, D, dtype_state_fn, seed, r, w, k, v, a, b, init_state):
    """跑一条递归路径 L 步,返回 (ys 列表, final_state, state_hist 列表)。

    dtype_state_fn: callable(state) -> state,对每步进 ops 前的 state 做 dtype 处理。
      fp32 路径:identity(state)
      bf16 路径:lambda s: s.astype(bf16)
    r/w/k/v/a/b: (1, L, H, D),输入张量(本函数不重新生成,保证两条路径同输入)。
    init_state: (1, H, D, D),初始 state。
    """
    state = init_state
    ys = []
    state_hist = []
    for t in range(L):
        s_in = dtype_state_fn(state)
        y, state = _wkv7_step_ops(
            r[:, t], w[:, t], k[:, t], v[:, t], a[:, t], b[:, t], s_in
        )
        ys.append(y)
        state_hist.append(state)
    return ys, state, state_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=273, help="递归步数(NekoQA smoke200 max token)")
    ap.add_argument("--heads", type=int, default=16, help="H(0.4B=16, 1.5B=32)")
    ap.add_argument("--head-dim", type=int, default=64, help="D")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="输出 JSON 路径(默认 stdout)")
    ap.add_argument("--record-every", type=int, default=1, help="记录间隔(1=每步)")
    ap.add_argument("--real-w-model", default=None,
                    help="用真实模型的 w 分布(路径)。不传则用合成 w。"
                         "真实 w 模式下,从该模型第0层 dump w,按真实分布跑数值等价性,"
                         "并按 w 分桶拆误差(裁决:高w通道误差是否线性走阔)")
    ap.add_argument("--data", default="train_data/NekoQA_10k/nekoqa_smoke_200.json",
                    help="真实 w 模式下的数据源(挑最长样本 dump w)。"
                         "要跑 >273 步需换含长样本的数据(如 NekoQA_30k)")
    args = ap.parse_args()

    L, H, D = args.steps, args.heads, args.head_dim
    mx.random.seed(args.seed)

    if args.real_w_model:
        # 真实 w 模式:从模型第0层 dump w(fp32 路径),r/k/v/a/b 仍合成
        print(f"[smoke] 加载真实 w 分布 from {args.real_w_model}", file=sys.stderr, flush=True)
        from statetuner.core import load_model, patch_rwkv7_for_train
        from statetuner.data import load_qa_dataset
        patch_rwkv7_for_train()
        rmdl, rtok = load_model(args.real_w_model, patch=True)
        rmdl.freeze()
        rmdl.set_dtype(mx.float32)
        rsamples = load_qa_dataset(
            args.data, rtok, max_len=4096)
        rsamples.sort(key=lambda s: -s.length)
        # 取真实样本的前 L token 的 w
        rin = mx.array([rsamples[0].input_ids[:L]])
        ATM = type(rmdl.layers[0]["attn"])
        rorig = ATM._wkv7
        rcap = {}
        def rhook(self, r_, w_, k_, v_, a_, b_, state_):
            if 0 not in rcap:
                rcap[0] = np.array(w_.astype(mx.float32))  # (1, L_real, H, D)
            return rorig(self, r_, w_, k_, v_, a_, b_, state_)
        ATM._wkv7 = rhook
        _ = rmdl(rin, rmdl.make_cache())
        ATM._wkv7 = rorig
        del rmdl
        real_w = rcap[0]  # (1, L_real, H, D)
        actual_L = min(L, real_w.shape[1])
        w = mx.array(real_w[:, :actual_L, :, :])  # 用真实 w,截到 actual_L 步
        print(f"[smoke] 真实 w shape={w.shape}, 实际步数={actual_L}", file=sys.stderr, flush=True)
        print(f"[smoke] 真实 w 分布:p50={float(np.percentile(real_w,50)):.4f} "
              f"p95={float(np.percentile(real_w,95)):.6f} "
              f">0.999 占 {np.mean(real_w>0.999)*100:.1f}%",
              file=sys.stderr, flush=True)
        L = actual_L  # 调整 L 到真实可用的步数
        scale = 0.1
        r = mx.random.normal((1, L, H, D)) * scale
        k = mx.random.normal((1, L, H, D)) * scale
        v = mx.random.normal((1, L, H, D)) * scale
        a = mx.random.normal((1, L, H, D)) * scale
        b = mx.random.normal((1, L, H, D)) * scale
        init_state = mx.random.normal((1, H, D, D)) * 0.05
    else:
        # 合成 w 模式(原行为)
        scale = 0.1
        r = mx.random.normal((1, L, H, D)) * scale
        w = 0.95 + mx.random.normal((1, L, H, D)) * 0.02  # 衰减系数,接近1
        k = mx.random.normal((1, L, H, D)) * scale
        v = mx.random.normal((1, L, H, D)) * scale
        a = mx.random.normal((1, L, H, D)) * scale
        b = mx.random.normal((1, L, H, D)) * scale
        init_state = mx.random.normal((1, H, D, D)) * 0.05  # 初始 S₀ 训练初期接近0

    # 两条路径输入全部 cast fp32(同输入),只 state 的 dtype 处理不同
    r32, w32, k32, v32, a32, b32 = (x.astype(mx.float32) for x in (r, w, k, v, a, b))
    init32 = init_state.astype(mx.float32)

    t0 = time.time()
    print(f"[smoke] L={L} H={H} D={D} seed={args.seed}", file=sys.stderr, flush=True)
    print(f"[smoke] 跑 fp32 路径...", file=sys.stderr, flush=True)
    ys_fp, state_fp, sh_fp = run_path(
        L, H, D, lambda s: s, args.seed,
        r32, w32, k32, v32, a32, b32, init32,
    )
    mx.eval(ys_fp, state_fp, sh_fp)
    print(f"[smoke] 跑 bf16 路径... ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)
    ys_bf, state_bf, sh_bf = run_path(
        L, H, D, lambda s: s.astype(mx.bfloat16), args.seed,
        r32.astype(mx.bfloat16), w32.astype(mx.bfloat16), k32.astype(mx.bfloat16),
        v32.astype(mx.bfloat16), a32.astype(mx.bfloat16), b32.astype(mx.bfloat16),
        init32,
    )
    mx.eval(ys_bf, state_bf, sh_bf)
    print(f"[smoke] 计算逐步误差... ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)

    # 逐步记录误差
    trace = []
    y_max_global = 0.0
    state_max_global = 0.0
    for t in range(L):
        y_worst, y_global, y_frac = _rel_err(ys_bf[t], ys_fp[t])
        s_worst, s_global, s_frac = _rel_err(sh_bf[t], sh_fp[t])
        y_max_global = max(y_max_global, y_global)
        state_max_global = max(state_max_global, s_global)
        if t % args.record_every == 0:
            trace.append({
                "step": t,
                "y_rel_global": y_global,
                "y_rel_worst": y_worst,
                "y_frac_gt1pct": y_frac,
                "state_rel_global": s_global,
                "state_rel_worst": s_worst,
                "state_frac_gt1pct": s_frac,
            })
            if t % 30 == 0 or t == L - 1:
                print(
                    f"[smoke] step {t:3d}/{L}  y_glob={y_global:.2e} "
                    f"(worst {y_worst:.2e}, {y_frac*100:.1f}%)  "
                    f"state_glob={s_global:.2e}",
                    file=sys.stderr, flush=True,
                )

    # ── 按 w 分桶拆误差(裁决:高 w 通道误差是否线性走阔)──
    # 谜题:21.9% 通道 w>0.999(bf16 下被舍成永不遗忘),为何全局误差只有 1% 还饱和?
    # 候选解释一:v7 的 a/b 主动擦除兜住了 w 失效 → 高 w 桶误差也饱和
    # 候选解释二:全局指标稀释 → 高 w 桶误差线性走阔,只是被平均淹了
    # 分桶粒度:(H,D) 通道。state 是 (H,D,D),state[h] 的第 d 行由 w[h,d] 衰减。
    # 把 (H,D) 通道按 w 展平分桶,对 state 的对应行算误差。
    w_np = np.array(w.astype(mx.float32))  # (1, L, H, D)
    w_chan = w_np[0].mean(axis=0)  # (H, D) 每通道时间平均 w
    w_flat = w_chan.flatten()  # (H*D,)
    high_mask = w_flat > 0.999   # 高 w 通道(bf16 盲区)
    low_mask = w_flat < 0.99     # 正常衰减通道
    n_high, n_low = int(high_mask.sum()), int(low_mask.sum())
    print(f"[smoke] 通道分桶(H*D={H*D}):高w(>0.999) {n_high} 通道, "
          f"低w(<0.99) {n_low} 通道", file=sys.stderr, flush=True)
    bucket_trace = {"high_w": [], "low_w": []}
    for t in range(L):
        sf = np.array(sh_fp[t].astype(mx.float32))[0]  # (H,D,D)
        sb = np.array(sh_bf[t].astype(mx.float32))[0]
        # reshape (H,D,D) → (H*D, D),每行对应一个 (H,D) 通道
        sf_flat = sf.reshape(H * D, D)
        sb_flat = sb.reshape(H * D, D)
        for bname, bmask in [("high_w", high_mask), ("low_w", low_mask)]:
            if bmask.sum() == 0:
                continue
            sf_b = sf_flat[bmask].flatten()
            sb_b = sb_flat[bmask].flatten()
            err = np.abs(sf_b - sb_b)
            denom = np.maximum(np.abs(sf_b), 1e-12)
            rel = float(err.sum() / denom.sum())
            bucket_trace[bname].append({"step": t, "state_rel": rel})

    # 桶误差趋势:首末值 + 是否线性走阔
    bucket_summary = {}
    for bname, bt in bucket_trace.items():
        if not bt:
            bucket_summary[bname] = "无该桶通道"
            continue
        first = bt[0]["state_rel"]
        last = bt[-1]["state_rel"]
        mid = bt[len(bt)//2]["state_rel"]
        # 线性走阔判定:末值/中值 > 2 且末值 > 首值×3
        grows = last > mid * 1.5 and last > first * 2
        bucket_summary[bname] = {
            "n_steps": len(bt),
            "first": round(first, 4), "mid": round(mid, 4), "last": round(last, 4),
            "trend": "线性走阔 ⚠️" if grows else "饱和/平稳",
        }
    print(f"[smoke] w 分桶误差(高w盲区 vs 低w正常):", file=sys.stderr, flush=True)
    for bname, bs in bucket_summary.items():
        print(f"  {bname}: {bs}", file=sys.stderr, flush=True)
    print(f"[smoke] 裁决:高w桶若'饱和'→ 解释一成立(v7主动擦除兜底,bf16边界由内存定);",
          file=sys.stderr, flush=True)
    print(f"        高w桶若'线性走阔'→ 解释二成立(全局稀释,bf16真实红线是数值的)",
          file=sys.stderr, flush=True)

    # 判据(主指标 = global 相对误差,稳定;worst 仅供发散早期诊断)
    final_y_global = trace[-1]["y_rel_global"]
    final_state_global = trace[-1]["state_rel_global"]
    # 检查 nan/inf
    has_nonfinite = any(
        math.isnan(r["y_rel_global"]) or math.isinf(r["y_rel_global"])
        or math.isnan(r["state_rel_global"]) or math.isinf(r["state_rel_global"])
        for r in trace
    )
    overall_max = max(y_max_global, state_max_global)

    if has_nonfinite:
        verdict = "DEAD"
        verdict_reason = "出现 nan/inf"
    elif overall_max > 1.0:
        verdict = "DEAD"
        verdict_reason = f"global 相对误差 {overall_max:.2e} > 1.0,递归发散"
    elif overall_max > 1e-1:
        verdict = "BORDERLINE"
        verdict_reason = f"global 相对误差 {overall_max:.2e} 落在边界区(1e-1~1),报裁决"
    else:
        verdict = "PASS"
        verdict_reason = f"global 相对误差 {overall_max:.2e} < 1e-1,数值风险可控"

    result = {
        "label": "smoke_numeric",
        "config": {"steps": L, "heads": H, "head_dim": D, "seed": args.seed,
                   "real_w_model": args.real_w_model},
        "summary": {
            "y_max_global_overall": y_max_global,
            "state_max_global_overall": state_max_global,
            "final_y_rel_global": final_y_global,
            "final_state_rel_global": final_state_global,
            "has_nonfinite": has_nonfinite,
            "elapsed_s": round(time.time() - t0, 1),
        },
        "w_bucket_analysis": bucket_summary,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "trace": trace,
    }
    print(f"[smoke] 判决: {verdict} — {verdict_reason}", file=sys.stderr, flush=True)
    out = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[smoke] 写入 {args.out}", file=sys.stderr, flush=True)
    else:
        print(out)


if __name__ == "__main__":
    main()
