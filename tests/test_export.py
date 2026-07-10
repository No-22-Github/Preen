"""导出器回归测试(快,~5s,不加载模型)。

验证 .pth 导出器的正确性——P1 任务1的核心。
覆盖:
  - round-trip:导出 pth → load(原样) → allclose 原始 state (x070: Runner不转置)
  - 键名:全是 blocks.{i}.att.time_state
  - 形状/dtype:(H,D,D) fp32
  - 方向正确性:x070 文件里存的是训练方向原样(不做swapaxes)
  - npz 读写
"""
import numpy as np
import pytest

from conftest import STATE_PATH
from statetuner.export import (
    export_pth,
    load_npz_as_numpy,
    load_pth_as_numpy,
    verify_roundtrip,
)


@pytest.fixture
def sample_states():
    """构造一个确定性小 state(3 层, 4 头, 8 维),数值可预测。"""
    rng = np.random.RandomState(42)
    return {i: rng.randn(4, 8, 8).astype(np.float32) * 0.1 for i in range(3)}


@pytest.fixture
def real_states():
    """真实 P0 state(24 层 16 头 64 维)。缺失时 skip。"""
    if not STATE_PATH.exists():
        pytest.skip(f"state 不存在: {STATE_PATH}")
    return load_npz_as_numpy(STATE_PATH)


# ── round-trip:导出 → 加载还原 == 原始 ──────────────────────

def test_roundtrip_synthetic(sample_states, tmp_pth):
    """合成 state round-trip:导出后经 Runner 逻辑加载回来 == 原始。"""
    export_pth(sample_states, tmp_pth)
    ok, msg = verify_roundtrip(sample_states, tmp_pth)
    assert ok, msg


def test_roundtrip_real(real_states, tmp_pth):
    """真实 state round-trip。"""
    export_pth(real_states, tmp_pth)
    ok, msg = verify_roundtrip(real_states, tmp_pth)
    assert ok, msg


def test_roundtrip_exact_zero():
    """全零 state round-trip(边界情况)。"""
    states = {i: np.zeros((4, 8, 8), dtype=np.float32) for i in range(2)}
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        path = f.name
    try:
        export_pth(states, path)
        ok, msg = verify_roundtrip(states, path)
        assert ok, msg
    finally:
        os.unlink(path)


# ── 键名 / 形状 / dtype(对照 RWKV-PEFT 源码约定)──────────────

def test_key_names(real_states, tmp_pth):
    """所有 key 必须是 blocks.{i}.att.time_state(RWKV-PEFT/Runner 约定)。"""
    export_pth(real_states, tmp_pth)
    import torch

    raw = torch.load(tmp_pth, map_location="cpu", weights_only=True)
    keys = list(raw.keys())
    assert len(keys) == 24  # 0.4B 是 24 层
    for k in keys:
        assert k.startswith("blocks."), f"键名格式错: {k}"
        assert k.endswith(".att.time_state"), f"键名后缀错: {k}"
        # 层号连续 0..23
        layer = int(k.split(".")[1])
        assert 0 <= layer < 24


def test_shapes_and_dtype(real_states, tmp_pth):
    """每层 shape (H, D, D) = (16, 64, 64),dtype fp32。"""
    export_pth(real_states, tmp_pth)
    import torch

    raw = torch.load(tmp_pth, map_location="cpu", weights_only=True)
    for k, v in raw.items():
        assert tuple(v.shape) == (16, 64, 64), f"{k} shape {v.shape}"
        assert v.dtype == torch.float32, f"{k} dtype {v.dtype}"


# ── 方向正确性(x070: Runner 不转置, 文件存原样)──────────────

def test_x070_no_swapaxes(sample_states, tmp_pth):
    """x070 (RWKV-7, 默认): 文件里存的是训练方向原样, 不做 swapaxes。

    RWKV-Runner rwkv.py:843 version>=7 分支加载时不 transpose,
    所以导出器必须存原样 S, 而非 swapaxes(S)。
    """
    export_pth(sample_states, tmp_pth)  # 默认 x070=True
    import torch

    raw = torch.load(tmp_pth, map_location="cpu", weights_only=True)
    stored = raw["blocks.0.att.time_state"].numpy()
    original = sample_states[0]
    # 文件里应该 == 原始训练方向 (x070 不转置)
    np.testing.assert_allclose(stored, original, atol=1e-7)


def test_v56_legacy_swapaxes(sample_states, tmp_pth):
    """v5/v6 兼容 (x070=False): 文件里存 swapaxes(S), Runner 转(1,2) 还原。"""
    export_pth(sample_states, tmp_pth, x070=False)
    import torch

    raw = torch.load(tmp_pth, map_location="cpu", weights_only=True)
    stored = raw["blocks.0.att.time_state"].numpy()
    original = sample_states[0]
    expected = np.swapaxes(original, -2, -1)
    np.testing.assert_allclose(stored, expected, atol=1e-7)
    assert not np.allclose(stored, original), "v5/v6 导出应做转置!"


# ── npz 读写 ──────────────────────────────────────────────

def test_npz_roundtrip(sample_states, tmp_path):
    """npz 写读 round-trip。"""
    npz_path = tmp_path / "s.npz"
    arrays = {f"layer_{k}": v for k, v in sample_states.items()}
    np.savez(npz_path, **arrays)
    loaded = load_npz_as_numpy(npz_path)
    assert set(loaded) == set(sample_states)
    for i in sample_states:
        np.testing.assert_allclose(loaded[i], sample_states[i])
