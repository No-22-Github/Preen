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
import math
import mmap
import os
import pickle
import sys
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple, Union

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


@dataclass(frozen=True)
class PthTensorInfo:
    """不触碰 storage 数据即可取得的 tensor 元数据。"""

    name: str
    dtype: np.dtype
    shape: Tuple[int, ...]


def _read_pth_metadata(zf: zipfile.ZipFile):
    """白名单反序列化 data.pkl，返回 prefix / byteorder / storage / tensor 描述。"""
    pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("data.pkl")]
    bo_entry = prefix + "byteorder"
    byteorder = zf.read(bo_entry).decode() if bo_entry in zf.namelist() else "little"
    storages: Dict[str, tuple] = {}

    def _rebuild_tensor_v2(storage, offset, size, stride, *rest):
        return ("TENSOR", storage[1], offset, tuple(size), tuple(stride))

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            _, stype, key, _loc, numel = pid
            name = stype.__name__ if hasattr(stype, "__name__") else str(stype)
            storages[str(key)] = (_STORAGE_TO_NP[name], int(numel))
            return ("STORAGE", str(key))

        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if name == "OrderedDict":
                return OrderedDict
            if name in _STORAGE_TO_NP:
                return type(name, (), {})
            raise pickle.UnpicklingError(
                f"pth_io refused to deserialize {module}.{name} (not allowlisted)"
            )

    obj = _Unpickler(io.BytesIO(zf.read(pkl_name))).load()
    if not isinstance(obj, (dict, OrderedDict)):
        raise ValueError(".pth data.pkl must contain a state_dict at the top level")
    return prefix, byteorder, storages, obj


def _validate_tensor_layout(
    name: str,
    storage_numel: int,
    offset: int,
    size: Tuple[int, ...],
    stride: Tuple[int, ...],
) -> None:
    """拒绝损坏或恶意的 storage 越界描述，避免 mmap 读到相邻 ZIP 数据。"""
    if offset < 0 or len(size) != len(stride):
        raise ValueError(f"Invalid tensor layout for {name}")
    if any(dim < 0 for dim in size) or any(step < 0 for step in stride):
        raise ValueError(f"Negative shape/stride is unsupported for {name}")
    if any(dim == 0 for dim in size):
        if offset > storage_numel:
            raise ValueError(f"Tensor storage offset out of bounds for {name}")
        return
    max_index = offset + sum((dim - 1) * step for dim, step in zip(size, stride))
    if max_index >= storage_numel:
        raise ValueError(
            f"Tensor storage range out of bounds for {name}: "
            f"max_index={max_index}, storage_numel={storage_numel}"
        )


def peek_pth_tensors(
    path: PathLike,
    *,
    require_stored: bool = False,
) -> list[PthTensorInfo]:
    """只读元数据并校验 storage/tensor 边界；不读取任何权重字节。

    `require_stored=True` 用于 mmap 转换入口：压缩 storage 会在创建输出目录前被拒绝。
    """
    with zipfile.ZipFile(str(path)) as zf:
        prefix, _byteorder, storages, obj = _read_pth_metadata(zf)
        for skey, (dtype, numel) in storages.items():
            entry = zf.getinfo(f"{prefix}data/{skey}")
            expected = int(numel) * np.dtype(dtype).itemsize
            if entry.file_size != expected:
                raise ValueError(
                    f"Storage size mismatch for {skey}: "
                    f"metadata={expected}, zip={entry.file_size}"
                )
            if require_stored and entry.compress_type != zipfile.ZIP_STORED:
                raise ValueError(
                    "Streaming model conversion requires a ZIP_STORED "
                    f"(uncompressed) .pth, but {entry.filename} has "
                    f"compress_type={entry.compress_type}. Re-save the weights "
                    "as an uncompressed torch .pth first."
                )

        result = []
        for name, val in obj.items():
            if not isinstance(val, tuple) or len(val) != 5 or val[0] != "TENSOR":
                raise ValueError(f"Unsupported state_dict value for {name}")
            _, skey, offset, size, stride = val
            if skey not in storages:
                raise ValueError(f"Tensor {name} references missing storage {skey}")
            dtype, storage_numel = storages[skey]
            _validate_tensor_layout(name, storage_numel, offset, size, stride)
            result.append(PthTensorInfo(name, np.dtype(dtype), tuple(size)))
        return result


def peek_pth_keys(path: PathLike) -> Tuple[list, int]:
    """只读 `data.pkl` 元数据,返回 (tensor 名称按出现顺序的列表, tensor 总数)。
    **不读取任何 storage 字节** —— 用于在流式转换前预知键名/层数/总数,
    避免把 storage 解压两遍。反序列化白名单与 read_pth/iter_pth 完全一致。
    """
    keys = [item.name for item in peek_pth_tensors(path)]
    return keys, len(keys)


def read_pth(path: PathLike) -> Dict[str, np.ndarray]:
    """读 torch zip-pickle `.pth` → {name: np.ndarray}。纯 Python,与 torch.load 等价。"""
    with zipfile.ZipFile(str(path)) as zf:
        prefix, byteorder, storages, obj = _read_pth_metadata(zf)
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


def _zip_entry_data_offset(zf: zipfile.ZipFile, entry_name: str, path: PathLike) -> int:
    """返回 zip entry 数据段在文件中的**绝对偏移**(local header 之后)。

    `ZipInfo.header_offset` 是 local file header 起始;数据紧跟在
    `[30字节固定头 + 文件名 + extra]` 之后。ZIP_STORED(未压缩)下,从该偏移
    mmap 出来的字节就是 entry 的原始数据,可直接建 numpy 视图。
    """
    info = zf.getinfo(entry_name)
    # 解析 local header 拿 filename/extra 长度(central directory 里的可能不一致)
    with open(path, "rb") as f:
        f.seek(info.header_offset + 26)
        fname_len = int.from_bytes(f.read(2), "little")
        extra_len = int.from_bytes(f.read(2), "little")
    return info.header_offset + 30 + fname_len + extra_len


def iter_pth(
    path: PathLike,
    names: Optional[Iterable[str]] = None,
) -> Iterator[Tuple[str, np.ndarray]]:
    """流式读 torch `.pth` → 按 `data.pkl` 原始顺序逐个 yield (name, ndarray)。

    **mmap 实现**:整文件只读 mmap 映射(虚拟地址,不占物理内存),每个 tensor 是
    mmap 内存上的 numpy 视图(零拷贝)。只有被访问的页才 page in 进物理内存;
    调用方(model_converter)astype 产生独立拷贝后,源视图可被 OS 自动回收。

    优势(相比 read_pth 的「全量 zf.read + reshape」):
      - 物理内存峰值 ≈ 几 MB(实测 1.5B 完整转换 ~3MB),而非 ~3-6GB;
      - 按 pkl 顺序遍历,而非按 storage 分组——产物可与旧版**逐字节一致**(同样的
        tensor 写出顺序),消除了「内存」与「pkl 顺序」的取舍;
      - 1.5B 的 pkl 顺序里 storage 有 98 次交错切换,mmap 下这不再消耗物理内存
        (未访问的页不常驻)。

    前提:torch `.pth` 是 `ZIP_STORED`(未压缩,RWKV 官方权重均如此)。压缩 storage
    无法直接 mmap，会在映射前抛出带修复建议的 ValueError。

    `dict(iter_pth(p))` 与 `read_pth(p)` 逐张量逐字节相等,且**键顺序也一致**。
    """
    path = str(path)
    # 先做完整元数据/边界/压缩校验；任何错误都发生在 mmap 与输出写入之前。
    peek_pth_tensors(path, require_stored=True)
    with zipfile.ZipFile(path) as zf:
        prefix, byteorder, storages, obj = _read_pth_metadata(zf)
        need_byteswap = byteorder != sys.byteorder
        storage_offsets = {
            skey: _zip_entry_data_offset(zf, f"{prefix}data/{skey}", path)
            for skey in storages
        }

    # 整文件 mmap(只读)。mmap 只是建立虚拟地址映射,不立即 page in。
    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, os.fstat(fd).st_size, prot=mmap.PROT_READ)
    finally:
        os.close(fd)  # mmap 建立后可关闭 fd(mmap 持有自己的引用)

    # 为每个 storage 建指向 mmap 的 frombuffer 视图(零拷贝,不占额外物理内存)
    storage_views: Dict[str, np.ndarray] = {}
    for skey, (dtype, numel) in storages.items():
        abs_off = storage_offsets[skey]
        storage_views[skey] = np.frombuffer(mm, dtype=dtype, count=numel, offset=abs_off)

    # 按 pkl 原始顺序遍历 tensor,逐个 yield。
    # C-contiguous 路径 yield mmap 视图(零拷贝);非 C-contiguous(as_strided)yield
    # 独立拷贝(必须,否则 strides 跨页不安全)。byteswap 路径也 yield 拷贝。
    # 注意:不显式 mm.close()——yield 出去的 C-contiguous 视图持有 mmap 缓冲区指针,
    # 强行 close 会抛 BufferError。mm 作为本生成器的局部变量,随生成器 GC 自动 unmap;
    # 调用方持有视图期间 mmap 钉住也无所谓(只是虚拟地址,物理页按需 page in)。
    ordered_names = list(obj) if names is None else list(names)
    if len(ordered_names) != len(set(ordered_names)):
        raise ValueError("iter_pth names contains duplicates")
    unknown = set(ordered_names) - set(obj)
    if unknown:
        raise KeyError(f"Unknown pth tensor names: {sorted(unknown)}")
    for name in ordered_names:
        val = obj[name]
        _, skey, offset, size, stride = val
        flat = storage_views[skey]
        if need_byteswap:
            flat = flat.byteswap()  # 产生新数组(脱离 mmap)
        numel_t = math.prod(size) if size else 1
        if stride == _c_stride(size):
            sub = flat[offset : offset + numel_t]
            yield name, (sub.reshape(size) if size else sub.reshape(()))
        else:
            strided = np.lib.stride_tricks.as_strided(
                flat[offset:], shape=size,
                strides=tuple(s * flat.itemsize for s in stride),
            )
            yield name, np.ascontiguousarray(strided)


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
