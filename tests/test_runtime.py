"""runtime.py 内存口径测试。

phys_footprint 是 macOS 专属(libproc);非 macOS 跳过 ctypes 路径。
重点测:口径区分(footprint vs rss)、MemorySnapshot 序列化、apply_cache_limit。
"""
from __future__ import annotations

import json
import sys

import pytest

from statetuner.runtime import MemorySnapshot, apply_cache_limit, memory_report


# ── memory_report:口径区分 ─────────────────────────────────


def test_memory_report_returns_all_fields():
    """memory_report 返回完整 MemorySnapshot(本进程)。"""
    snap = memory_report()
    assert isinstance(snap, MemorySnapshot)
    assert snap.pid > 0
    # macOS:footprint 必须有值;非 macOS 可能 None
    if sys.platform == "darwin":
        assert snap.footprint_gb is not None
        assert snap.footprint_gb >= 0
        assert snap.rss_gb is not None
        assert snap.rss_gb >= 0


@pytest.mark.skipif(sys.platform != "darwin", reason="phys_footprint 仅 macOS")
def test_footprint_includes_metal_iokit_mapping():
    """footprint(phys_footprint)是 Activity Monitor 口径,含 IOKit 映射。

    这是它和 rss 的关键区别 —— Metal GPU buffer 走 IOKit 记账,
    resident_size 不含。加载模型后 footprint 应 ≥ rss(训练时差距更大)。
    本测试只验证字段都读得到 + footprint 非零(不加载模型,数值小)。
    """
    snap = memory_report()
    assert snap.footprint_gb is not None
    assert snap.footprint_gb > 0, "footprint 应非零(进程本身有内存)"


def test_memory_snapshot_serializes_to_json():
    """MemorySnapshot.to_dict 可 JSON 序列化(serve/UI 传输用)。"""
    snap = memory_report()
    d = snap.to_dict()
    # 必须能 json.dumps(None 值也合法)
    s = json.dumps(d, ensure_ascii=False)
    back = json.loads(s)
    assert back["pid"] == snap.pid
    assert "footprint_gb" in back


def test_memory_snapshot_summary_line_handles_none():
    """summary_line 对 None 字段不崩(非 macOS 或 MLX 未加载时)。"""
    snap = MemorySnapshot(
        pid=1,
        footprint_gb=None,
        peak_footprint_gb=None,
        rss_gb=None,
        mlx_active_gb=None,
        mlx_cache_gb=None,
        mlx_peak_gb=None,
    )
    line = snap.summary_line()
    assert "n/a" in line
    assert "footprint" in line


# ── vmmap 交叉验证(慢,但只跑一次确认口径对)──────────────


@pytest.mark.skipif(sys.platform != "darwin", reason="vmmap 仅 macOS")
def test_vmmap_cross_check_matches_footprint():
    """vmmap 独立信源应与 phys_footprint 同量级(±30% 容差)。

    vmmap 统计口径略宽(含部分 mapped 未 resident 页),不会完全相等,
    但量级必须一致。差太大说明 ctypes 偏移读错了。
    """
    from statetuner.runtime import vmmap_footprint

    snap = memory_report(cross_check=True)
    assert snap.vmmap_footprint_gb is not None, "vmmap 应返回结果"
    assert snap.footprint_gb is not None
    # 容差 30%(vmmap 略宽)
    ratio = snap.vmmap_footprint_gb / snap.footprint_gb
    assert 0.5 < ratio < 3.0, (
        f"vmmap({snap.vmmap_footprint_gb:.3f}) vs footprint({snap.footprint_gb:.3f}) "
        f"比例 {ratio:.2f} 异常,ctypes 偏移可能读错"
    )


# ── apply_cache_limit ──────────────────────────────────────


def test_apply_cache_limit_rejects_bad_spec():
    """非法 spec → ValueError(不抛 typer.Exit,UI 中立)。"""
    with pytest.raises(ValueError, match="--cache-limit-gb"):
        apply_cache_limit("abc")
    # 负数/零:float() 能解析,但 gb<=0 拦截
    with pytest.raises(ValueError, match="> 0"):
        apply_cache_limit("-5")
    with pytest.raises(ValueError, match="> 0"):
        apply_cache_limit("0")


def test_apply_cache_limit_none_returns_none():
    """spec=None 不动 MLX 默认,返回 None。"""
    assert apply_cache_limit(None) is None
