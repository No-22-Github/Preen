"""纯 Python 的 torch `.pth` 读写(无 torch 依赖)。

RWKV 官方权重与我们导出的 state 都是 torch 用 zip+pickle 存的 `.pth`:
zip 里 `data.pkl` 描述张量(靠 `_rebuild_tensor_v2` 引用 storage 的切片),
`data/<key>` 是裸 storage 字节。torch 反序列化只需要这套约定,不需要 torch 本体。

- `read_pth`  : 读 → {name: np.ndarray},与 `torch.load` 逐字节等价。
- `write_pth` : {name: np.ndarray} → `.pth`,`torch.load(weights_only=True)` 可读回,
                RWKV Runner(走 `torch.load`)可挂载。

bf16 借 `ml_dtypes.bfloat16` 补上 numpy 缺失的类型。仅支持 C-contiguous 张量
(RWKV 权重与 state 均如此);遇到非平凡 stride 会显式物化,不静默出错。
"""
from __future__ import annotations

import io
import pickle
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Union

import numpy as np
import ml_dtypes

PathLike = Union[str, Path]

# torch storage 类型名 ↔ numpy dtype
_STORAGE_TO_NP = {
    "BFloat16Storage": ml_dtypes.bfloat16,
    "HalfStorage": np.float16,
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ByteStorage": np.uint8,
}
_NP_TO_STORAGE = {
    "bfloat16": "BFloat16Storage",
    "float16": "HalfStorage",
    "float32": "FloatStorage",
    "float64": "DoubleStorage",
    "int64": "LongStorage",
    "int32": "IntStorage",
    "uint8": "ByteStorage",
}


def _c_stride(size):
    if not size:
        return ()
    stride = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        stride[i] = stride[i + 1] * size[i + 1]
    return tuple(stride)


# ────────────────────────── 读 ──────────────────────────

def read_pth(path: PathLike) -> Dict[str, np.ndarray]:
    """读 torch zip-pickle `.pth` → {name: np.ndarray}。纯 Python,与 torch.load 等价。"""
    zf = zipfile.ZipFile(str(path))
    pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("data.pkl")]
    bo_entry = prefix + "byteorder"
    byteorder = zf.read(bo_entry).decode() if bo_entry in zf.namelist() else "little"

    storages: Dict[str, tuple] = {}  # key -> (np_dtype, numel)

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            _, stype, key, _loc, numel = pid
            name = stype.__name__ if hasattr(stype, "__name__") else str(stype)
            storages[str(key)] = (_STORAGE_TO_NP[name], numel)
            return ("STORAGE", str(key))

        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                def _rec(storage, offset, size, stride, *rest):
                    return ("TENSOR", storage[1], offset, tuple(size), tuple(stride))
                return _rec
            if name == "OrderedDict":
                return OrderedDict
            try:
                return super().find_class(module, name)
            except Exception:
                return type(name, (), {})

    obj = _Unpickler(io.BytesIO(zf.read(pkl_name))).load()

    raw: Dict[str, np.ndarray] = {}
    for key, (dtype, numel) in storages.items():
        buf = zf.read(f"{prefix}data/{key}")
        arr = np.frombuffer(buf, dtype=dtype, count=numel)
        if byteorder != sys.byteorder:
            arr = arr.byteswap()
        raw[key] = arr

    out: Dict[str, np.ndarray] = {}
    for name, val in obj.items():
        _, skey, offset, size, stride = val
        flat = raw[skey]
        numel = int(np.prod(size)) if size else 1
        if stride == _c_stride(size):
            sub = flat[offset : offset + numel]
            out[name] = sub.reshape(size) if size else sub.reshape(())
        else:
            strided = np.lib.stride_tricks.as_strided(
                flat[offset:], shape=size,
                strides=tuple(s * flat.itemsize for s in stride),
            )
            out[name] = np.ascontiguousarray(strided)
    return out


# ────────────────────────── 写 ──────────────────────────

class _Global:
    """pickle 成真正的 GLOBAL opcode(module.name),torch.load 侧解析。
    需可调用以过 pickle save_reduce 的 callable 检查(实际不被调用)。"""
    def __init__(self, module, name):
        self.module, self.name = module, name

    def __call__(self, *a, **k):  # pragma: no cover
        raise RuntimeError("marker only")


class _StorageRef:
    def __init__(self, dtype_name, key, numel):
        self.dtype_name, self.key, self.numel = dtype_name, key, numel


class _Tensor:
    def __init__(self, storage, size, stride):
        self.storage, self.size, self.stride = storage, size, stride


class _Pickler(pickle._Pickler):  # 纯 Python 版:才能覆写 save 手写 GLOBAL opcode
    def save(self, obj, save_persistent_id=True):
        if isinstance(obj, _Global):
            self.write(pickle.GLOBAL + f"{obj.module}\n{obj.name}\n".encode("utf-8"))
            self.memoize(obj)
            return
        return super().save(obj, save_persistent_id)

    def persistent_id(self, obj):
        if isinstance(obj, _StorageRef):
            return ("storage", _Global("torch", obj.dtype_name), obj.key, "cpu", obj.numel)
        return None

    def reducer_override(self, obj):
        if isinstance(obj, _Tensor):
            rebuild = _Global("torch._utils", "_rebuild_tensor_v2")
            return (rebuild, (obj.storage, 0, obj.size, obj.stride, False, OrderedDict()))
        return NotImplemented


def write_pth(tensors: Dict[str, np.ndarray], path: PathLike) -> Path:
    """{name: np.ndarray} → torch 格式 `.pth`。torch.load / RWKV Runner 可读。"""
    prefix = "archive"
    storages = []  # (key, raw_bytes)
    top = OrderedDict()
    for i, (name, arr) in enumerate(tensors.items()):
        arr = np.ascontiguousarray(arr)
        stype = _NP_TO_STORAGE[arr.dtype.name]
        key = str(i)
        storages.append((key, arr.tobytes()))
        size = tuple(int(x) for x in arr.shape)
        top[name] = _Tensor(_StorageRef(stype, key, int(arr.size)), size, _c_stride(size))

    buf = io.BytesIO()
    _Pickler(buf, protocol=2).dump(top)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(path), "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(f"{prefix}/data.pkl", buf.getvalue())
        zf.writestr(f"{prefix}/byteorder", sys.byteorder)
        zf.writestr(f"{prefix}/version", "3\n")
        for key, raw in storages:
            zf.writestr(f"{prefix}/data/{key}", raw)
    return path
