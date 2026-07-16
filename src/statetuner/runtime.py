"""进程运行时资源:内存口径 + MLX cache 上限。UI 中立,CLI / serve / probe 共用。

═══ 为什么需要这个模块 ═══

历史上仓库里的内存数字有三套口径,互相对不上:
  ① mx.get_peak_memory()           —— MLX allocator 视角,不含 Metal wired,严重漏报
  ② ru_maxrss / ps rss             —— 内核 resident set,**不含 IOKit 映射**,漏报 ~7x
  ③ Activity Monitor 的「内存」列   —— 真正的判断依据,但只能人眼看

②③ 对不上的根因:**Activity Monitor 的「内存」列不是 RSS**。
自 OS X 10.9 起它显示的是 `phys_footprint`(物理足迹),账本包含:
    internal + internal_compressed + iokit_mapped + page_table + alternate_accounting
而 resident_size 不含 **iokit_mapped** —— 而 Metal 的 GPU buffer 正是走 IOKit 记账的。
所以任何 RSS 系 API 都不可能对上 Activity Monitor:它们数的不是同一个账本。

本模块提供 ①②③ 的统一读取,其中 ③ 是**唯一可用于红线判断的口径**。

═══ 单位铁律 ═══

本模块所有 *_gb 一律 ÷1e9(GB),禁止 GiB。见 AGENTS.md「内存单位」。
本模块内部一律传 bytes(int),只在 report 层转 GB。
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "phys_footprint",
    "peak_phys_footprint",
    "resident_size",
    "vmmap_footprint",
    "memory_report",
    "MemorySnapshot",
    "apply_cache_limit",
]

# ── libproc: proc_pid_rusage ────────────────────────────────
# int proc_pid_rusage(int pid, int flavor, rusage_info_t *buffer);
# 同用户进程可读,无需 root / entitlement。

RUSAGE_INFO_V0 = 0
RUSAGE_INFO_V4 = 4


class _RusageInfoV4(ctypes.Structure):
    """<sys/resource.h> 的 rusage_info_v4。

    ⚠️ 字段顺序是 ABI,不可重排。ri_phys_footprint 位于偏移 16 + 7*8 = 72,
    这个位置自 10.9 起从未变过(v0 就在那儿),因此**即便后续字段布局在未来内核里
    变了,phys_footprint 仍然读得对**。ri_lifetime_max_phys_footprint 的偏移依赖
    v3 段的完整布局,风险略高 —— 所以 memory_report 会拿它跟 vmmap 交叉验证。
    """

    _fields_ = [
        # ── v0 ────────────────────────────────────────────
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),      # ← 老口径(漏 IOKit)
        ("ri_phys_footprint", ctypes.c_uint64),     # ← ★ Activity Monitor 的「内存」列
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        # ── v1 ────────────────────────────────────────────
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        # ── v2 ────────────────────────────────────────────
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        # ── v3 ────────────────────────────────────────────
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        # ── v4 ────────────────────────────────────────────
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),  # ← ★ 峰值,白送,无需轮询
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


_libc = None


def _proc_pid_rusage(pid: int) -> _RusageInfoV4:
    """读 rusage_info_v4。仅 macOS。失败抛 OSError。"""
    global _libc
    if sys.platform != "darwin":
        raise OSError("phys_footprint is available only on macOS")
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    buf = _RusageInfoV4()
    rc = _libc.proc_pid_rusage(
        ctypes.c_int(pid), ctypes.c_int(RUSAGE_INFO_V4), ctypes.byref(buf)
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"proc_pid_rusage(pid={pid}) failed: errno={err}")
    return buf


def phys_footprint(pid: Optional[int] = None) -> int:
    """当前物理足迹(bytes)。**== Activity Monitor 的「内存」列。**

    含 Metal 的 IOKit 映射内存,这是它和 RSS 的关键区别。
    pid=None → 当前进程。跨进程需同用户(sidecar 场景满足)。
    """
    return int(_proc_pid_rusage(pid if pid is not None else os.getpid()).ri_phys_footprint)


def peak_phys_footprint(pid: Optional[int] = None) -> int:
    """进程生命周期内的物理足迹峰值(bytes)。内核记账,**无需轮询采样**。

    ⚠️ 该字段偏移依赖 rusage_info_v3 段的完整布局,比 phys_footprint 脆。
    首次使用请用 memory_report(cross_check=True) 与 vmmap 的
    "Physical footprint (peak)" 对齐一次;对不上就别信它,退回轮询 phys_footprint 取 max。
    """
    return int(
        _proc_pid_rusage(pid if pid is not None else os.getpid())
        .ri_lifetime_max_phys_footprint
    )


def resident_size(pid: Optional[int] = None) -> int:
    """RSS(bytes)。**保留仅为对照,不要用它做红线判断** —— 不含 IOKit 映射。"""
    return int(_proc_pid_rusage(pid if pid is not None else os.getpid()).ri_resident_size)


def vmmap_footprint(pid: Optional[int] = None) -> tuple[Optional[int], Optional[int]]:
    """用 /usr/bin/vmmap 独立读一次 (current, peak) 物理足迹(bytes)。

    这是**独立于 ctypes 路径的第二个信息源**,用于校准上面两个函数。
    同用户进程无需 sudo。vmmap 不可用 / 解析失败 → (None, None),不抛。
    慢(数百 ms),只用于校准,不要放进采样循环。
    """
    pid = pid if pid is not None else os.getpid()
    try:
        out = subprocess.run(
            ["/usr/bin/vmmap", "--summary", str(pid)],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return (None, None)

    current = peak = None
    for line in out.splitlines():
        low = line.lower()
        if "physical footprint" not in low:
            continue
        value = line.split(":", 1)[-1].strip()
        parsed = _parse_size(value)
        if "peak" in low:
            peak = parsed
        else:
            current = parsed
    return (current, peak)


def _parse_size(text: str) -> Optional[int]:
    """把 vmmap 的 '1.2G' / '842.5M' / '1234K' 解析成 bytes。

    ⚠️ vmmap 用的是 **二进制** 单位(1G = 1024³),而本仓库汇报口径是 GB(1e9)。
    这里先老老实实按 vmmap 的二进制语义还原成 bytes,再由 report 层统一 ÷1e9。
    混用口径正是 AGENTS.md 明令禁止的坑。
    """
    text = text.strip()
    if not text:
        return None
    mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    unit = text[-1].upper()
    try:
        if unit in mult:
            return int(float(text[:-1]) * mult[unit])
        return int(float(text))
    except ValueError:
        return None


# ── 统一报告 ────────────────────────────────────────────────

@dataclass(frozen=True)
class MemorySnapshot:
    """一次内存快照。全部字段 GB(÷1e9)。None = 该口径不可用。

    判断红线只看 footprint_gb / peak_footprint_gb。其余字段是归因用的。
    """

    pid: int
    footprint_gb: Optional[float]        # ★ == Activity Monitor 内存列
    peak_footprint_gb: Optional[float]   # ★ 生命周期峰值(内核记账)
    rss_gb: Optional[float]              # 对照用,不做判断(漏 IOKit)
    mlx_active_gb: Optional[float]       # MLX allocator 视角
    mlx_cache_gb: Optional[float]
    mlx_peak_gb: Optional[float]         # = get_peak_memory(),已知漏报
    vmmap_footprint_gb: Optional[float] = None   # 交叉验证用
    vmmap_peak_gb: Optional[float] = None

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    def summary_line(self) -> str:
        def g(v):
            return f"{v:.2f}G" if v is not None else "n/a"

        return (
            f"footprint {g(self.footprint_gb)} (peak {g(self.peak_footprint_gb)}) · "
            f"mlx active {g(self.mlx_active_gb)} + cache {g(self.mlx_cache_gb)} · "
            f"rss {g(self.rss_gb)} [reference only]"
        )


def memory_report(
    pid: Optional[int] = None, *, cross_check: bool = False
) -> MemorySnapshot:
    """采一次全口径快照。

    cross_check=True 时额外调 vmmap(慢,数百 ms)做独立校验 —— 只在校准/探针里开,
    不要放进 1Hz 采样循环。
    """
    pid = pid if pid is not None else os.getpid()
    gb = lambda b: (b / 1e9 if b is not None else None)  # noqa: E731

    fp = peak = rss = None
    try:
        info = _proc_pid_rusage(pid)
        fp = int(info.ri_phys_footprint)
        peak = int(info.ri_lifetime_max_phys_footprint)
        rss = int(info.ri_resident_size)
    except OSError:
        pass

    active = cache = mlx_peak = None
    if pid == os.getpid():  # MLX 统计只对本进程有意义
        try:
            import mlx.core as mx

            active = int(mx.get_active_memory())
            cache = int(mx.get_cache_memory())
            mlx_peak = int(mx.get_peak_memory())
        except (ImportError, AttributeError):
            pass

    vm_cur = vm_peak = None
    if cross_check:
        vm_cur, vm_peak = vmmap_footprint(pid)

    return MemorySnapshot(
        pid=pid,
        footprint_gb=gb(fp),
        peak_footprint_gb=gb(peak),
        rss_gb=gb(rss),
        mlx_active_gb=gb(active),
        mlx_cache_gb=gb(cache),
        mlx_peak_gb=gb(mlx_peak),
        vmmap_footprint_gb=gb(vm_cur),
        vmmap_peak_gb=gb(vm_peak),
    )


# ── MLX cache 上限(从 cli.py 迁出,T2)──────────────────────

def apply_cache_limit(spec: Optional[str]) -> Optional[int]:
    """在任何 MLX 加载/分配**之前**设 buffer cache 上限。返回实际设置的 bytes。

    时序铁律:必须在 load_model 前调用才有效(tools/mem_probe.py:106-117 验证)。

    spec:
      None       — 不动 MLX 默认
      "auto"     — 物理内存 × 25%(16G 机 ≈ 4.3G)
      "<number>" — 直接当 GB

    ⚠️ 不抛 typer.Exit —— 这是 UI 中立层,CLI 与 serve 共用。
    非法输入抛 ValueError,由调用方翻译成各自的错误形态。
    """
    if spec is None:
        return None
    import mlx.core as mx

    if spec == "auto":
        info = mx.device_info()
        total = info.get("memory_size")
        if not total:
            raise ValueError("Unable to read physical memory (device_info has no memory_size); specify a GB value explicitly")
        gb = total / 1e9 * 0.25
    else:
        try:
            gb = float(spec)
        except ValueError as exc:
            raise ValueError(
                f"--cache-limit-gb accepts only 'auto' or a positive number; received {spec!r}"
            ) from exc
    if gb <= 0:
        raise ValueError("cache-limit-gb must be > 0")
    limit = int(gb * 1e9)
    mx.set_cache_limit(limit)
    return limit


# ── 自校验入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    # 用法: python -m statetuner.runtime [pid]
    # 把结果与 Activity Monitor 的「内存」列并排看 —— footprint 必须对得上。
    target = int(sys.argv[1]) if len(sys.argv) > 1 else os.getpid()
    snap = memory_report(target, cross_check=True)
    print(f"pid={snap.pid}")
    print(f"  footprint       {snap.footprint_gb}        <- should match Activity Monitor memory")
    print(f"  peak footprint  {snap.peak_footprint_gb}")
    print(f"  vmmap current   {snap.vmmap_footprint_gb}  <- independent source; should match footprint")
    print(f"  vmmap peak      {snap.vmmap_peak_gb}       <- independent source; should match peak footprint")
    print(f"  rss             {snap.rss_gb}              <- reference only (excludes IOKit and is lower)")
    print(f"  mlx active      {snap.mlx_active_gb}")
    print(f"  mlx cache       {snap.mlx_cache_gb}")
    print(f"  mlx peak        {snap.mlx_peak_gb}         <- known undercount; do not use for decisions")
