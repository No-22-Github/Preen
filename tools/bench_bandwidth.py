#!/usr/bin/env python3
"""测 M5 Air 在 MLX 下的实测可达内存带宽,用于给 decode 的 roofline 定地板。"""
import time

import mlx.core as mx

N = 16384          # 16384^2 * 2B = 536.9 MB / 矩阵
NUM_MATS = 4       # 总权重 ~2.1GB,超出 SLC,保证走 DRAM
WARMUP = 3
RUNS = 10

BYTES_PER_MAT = N * N * 2
DECODE_GB_PER_TOKEN = 2.78   # 3.055GB 总权重 - 0.268GB embedding table
SYNC_COMPILE_MS = 23.7       # 同口径实测；pipeline 23.3ms 含首步预取重叠


def bench(fn, label, bytes_per_iter):
    for _ in range(WARMUP):
        mx.eval(fn())
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append(time.perf_counter() - t0)
    times.sort()
    p50 = times[len(times) // 2]
    gbps = bytes_per_iter / p50 / 1e9
    print(
        f"  {label:<28s} p50 {p50 * 1e3:7.2f} ms │ {gbps:6.1f} GB/s"
        f"  (min-max {bytes_per_iter / times[-1] / 1e9:.1f}–"
        f"{bytes_per_iter / times[0] / 1e9:.1f})"
    )
    return gbps


def main():
    mx.random.seed(0)
    print(f"[setup] 分配 {NUM_MATS} × {BYTES_PER_MAT / 1e9:.2f} GB bf16 矩阵…")
    weights = [
        mx.random.normal((N, N), dtype=mx.bfloat16) for _ in range(NUM_MATS)
    ]
    vector = mx.random.normal((N,), dtype=mx.bfloat16)
    mx.eval(weights, vector)

    # GEMV:和 decode 一样流式读整块权重；显式参数避免捕获闭包 array。
    gemv_step = mx.compile(lambda ws, x: [weight @ x for weight in ws])
    print(f"[bench] runs = {RUNS} (每轮扫 4 块矩阵,击穿缓存)")
    bw_gemv = bench(
        lambda: gemv_step(weights, vector),
        "GEMV (decode 模式)",
        NUM_MATS * BYTES_PER_MAT,
    )

    # 纯归约:无 GEMV 点积依赖,作为可达带宽上限对照。
    reduction_step = mx.compile(lambda ws: [weight.sum() for weight in ws])
    bw_reduction = bench(
        lambda: reduction_step(weights),
        "sum reduction (上限对照)",
        NUM_MATS * BYTES_PER_MAT,
    )

    print()
    print(
        f"  标称带宽 153 GB/s → GEMV 利用率 {bw_gemv / 153 * 100:.0f}%,"
        f" reduction 利用率 {bw_reduction / 153 * 100:.0f}%"
    )
    floor = DECODE_GB_PER_TOKEN / bw_gemv * 1e3
    print(
        f"  按 GEMV 实测带宽,bf16 decode 地板 = {DECODE_GB_PER_TOKEN} GB ÷ "
        f"{bw_gemv:.1f} GB/s = {floor:.1f} ms/token"
    )
    residual = SYNC_COMPILE_MS - floor
    print(
        f"  同步 compile 实测 {SYNC_COMPILE_MS:.1f} ms → roofline 残差 "
        f"{residual:+.1f} ms/token"
    )
    print("  pipeline 的 23.3 ms 含首步预取重叠,不能拿来计算负 dispatch 税")


if __name__ == "__main__":
    main()
