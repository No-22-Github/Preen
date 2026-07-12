"""cli / serve 共用的运行期辅助(T2)。

此前 serve.py 从 cli 导 _apply_cache_limit,把 typer 拖进 sidecar 进程,
且该函数出错抛 typer.Exit(在 serve 里毫无意义)。抽出此模块让两边共用,
错误走普通 ValueError(调用方自行决定怎么呈现)。

时序铁律:cache_limit 必须在 load_model 之前生效(见 AGENTS.md 内存事实 +
tools/mem_probe.py:106-117)。调用方在 load_model 前调用本函数。
"""
from __future__ import annotations

from typing import Optional


def _bad_value(message: str) -> None:
    """统一抛 ValueError(而非 typer.Exit),由 cli/serve 各自的 try 包成错误呈现。"""
    raise ValueError(message)


def apply_cache_limit(spec: Optional[str]) -> None:
    """load_model 前设 MLX buffer cache 上限(GB 口径)。

    必须在任何 MLX 加载/分配前调用才有效(mem_probe 验证过的时序)。
    spec 解析:
      "auto"     — 物理内存 × 25%(16GB 机器 ≈ 4.3G,c4G 同档)
      "<number>" — 直接当 GB,如 "4" → 4G
    全仓 GB 口径(÷1e9),禁止 /1024³。
    """
    if spec is None:
        # 调用方未传 spec(理论上不会触发:CLI/serve 默认是 "auto")。
        # 不保留"理论上不会被触发"的薛定谔分支,出现说明漏传 → 断言。
        assert False, "apply_cache_limit 收到 None(spec 缺省应为 'auto')"
    import mlx.core as mx

    if spec == "auto":
        # memory_size 用 .get 兜底:Linux MLX / 部分版本 device_info 可能缺该键,
        # 与 doctor_report 保持同一防御口径(否则裸下标 KeyError)。
        mem_bytes = mx.device_info().get("memory_size", 0)
        if mem_bytes <= 0:
            _bad_value(
                "MLX device_info 未报告 memory_size,无法走 auto;"
                "请显式 --cache-limit-gb <GB>"
            )
        gb = mem_bytes / 1e9 * 0.25
    else:
        try:
            gb = float(spec)
        except ValueError:
            _bad_value(
                f"--cache-limit-gb 只接受 'auto' 或正数, 收到 {spec!r}"
            )
    if gb <= 0:
        _bad_value("--cache-limit-gb 必须 > 0")
    mx.set_cache_limit(int(gb * 1e9))
