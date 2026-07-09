"""导出器回归测试(快,~5s,不加载模型)。

验证 .pth 导出器的正确性——P1 任务1的核心。
覆盖:
  - round-trip:导出 pth → load + transpose → allclose 原始 state
  - 键名:全是 blocks.{i}.att.time_state
  - 形状/dtype:(H,D,D) fp32
  - 转置正确性:文件里的 = transpose(S_mlx)(模拟 Runner 加载逻辑)
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


# ── 转置方向正确性(暗坑验证)──────────────────────────────

def test_transpose_direction(sample_states, tmp_pth):
    """验证转置暗坑:文件里存的是 swapaxes(S,-2,-1),不是 S 本身。

    这确保 Runner 的 transpose(1,2) 恰好还原成 MLX 训练方向。
    """
    export_pth(sample_states, tmp_pth)
    import torch

    raw = torch.load(tmp_pth, map_location="cpu", weights_only=True)
    # 取 layer 0
    stored = raw["blocks.0.att.time_state"].numpy()
    original = sample_states[0]
    # 文件里应该 == swapaxes(原始, -2, -1), 而不是原始本身
    expected = np.swapaxes(original, -2, -1)
    np.testing.assert_allclose(stored, expected, atol=1e-7)
    # 且不等于原始(证明转置确实发生了;除非原始恰好对称)
    assert not np.allclose(stored, original), "导出未做转置!"


def test_load_reverse_transpose_restores_original(sample_states, tmp_pth):
    """load_pth(reverse_transpose=True) 应还原成 MLX 原始方向。"""
    export_pth(sample_states, tmp_pth)
    loaded = load_pth_as_numpy(tmp_pth, reverse_transpose=True)
    for i in sample_states:
        np.testing.assert_allclose(loaded[i], sample_states[i], atol=1e-7)


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
