"""pth_io 纯 Python torch .pth 读写回归测试(快,不加载模型/不需 torch)。

覆盖 write_pth → read_pth 的 round-trip 一致性,含 bf16(RWKV 原始权重的 dtype)。
外部 oracle(torch.load)在 test_export.py::test_torch_can_load,torch 在场时才跑。
"""
import numpy as np
import ml_dtypes
import pytest

from statetuner.pth_io import read_pth, write_pth


def test_roundtrip_fp32(tmp_path):
    p = tmp_path / "s.pth"
    tensors = {
        "blocks.0.att.time_state": (np.random.rand(4, 8, 8) * 2 - 1).astype(np.float32),
        "misc.weight": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    }
    write_pth(tensors, p)
    back = read_pth(p)
    assert set(back) == set(tensors)
    for k, v in tensors.items():
        assert back[k].dtype == v.dtype
        assert back[k].shape == v.shape
        assert back[k].tobytes() == v.tobytes(), f"{k} 非逐字节等价"


def test_roundtrip_bf16(tmp_path):
    """bf16 是 RWKV 原始权重的 dtype,numpy 靠 ml_dtypes 支持。"""
    p = tmp_path / "w.pth"
    arr = (np.random.rand(64, 128).astype(np.float32) * 2 - 1).astype(ml_dtypes.bfloat16)
    write_pth({"emb.weight": arr}, p)
    back = read_pth(p)["emb.weight"]
    assert back.dtype == ml_dtypes.bfloat16
    assert back.shape == (64, 128)
    assert back.tobytes() == arr.tobytes()


def test_1d_and_scalar_like(tmp_path):
    p = tmp_path / "v.pth"
    tensors = {"bias": np.arange(1024, dtype=np.float32), "single": np.array([3.5], dtype=np.float32)}
    write_pth(tensors, p)
    back = read_pth(p)
    for k, v in tensors.items():
        assert back[k].tobytes() == v.tobytes()


def test_torch_reads_bf16(tmp_path):
    """torch 在场时确认 bf16 文件 torch.load 可读(RWKV Runner 场景)。torch 缺席则 skip。"""
    torch = pytest.importorskip("torch")
    p = tmp_path / "bf16.pth"
    arr = (np.random.rand(32, 32).astype(np.float32)).astype(ml_dtypes.bfloat16)
    write_pth({"x": arr}, p)
    t = torch.load(p, map_location="cpu", weights_only=True)["x"]
    assert t.dtype == torch.bfloat16
    assert tuple(t.shape) == (32, 32)
    assert t.view(torch.uint8).numpy().tobytes() == arr.tobytes()
