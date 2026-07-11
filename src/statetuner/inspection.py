"""CLI/未来 UI 共用的环境、数据与 state 检查。"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DataInspection:
    path: str
    total: int
    valid: int
    skipped_empty_question: int
    skipped_empty_answer: int
    truncated: int
    target_fully_truncated: int
    min_tokens: int
    mean_tokens: float
    p95_tokens: float
    max_tokens: int
    ctx_len: int
    template: str = "nekoqa"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StateInspection:
    path: str
    format: str
    layers: int
    layer_indices: list[int]
    continuous_layers: bool
    shapes: list[list[int]]
    dtypes: list[str]
    std_min: float
    std_mean: float
    std_max: float
    rwkv7_compatible: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _read_items(path: Path) -> list[Any]:
    if not path.exists():
        raise ValueError(f"数据文件不存在: {path}")
    try:
        if path.suffix.lower() == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, list) else [loaded]
        items = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {lineno} 行解析失败: {exc.msg}") from exc
        return items
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败: {exc.msg}") from exc


def load_qa_pairs(path: Path, *, require_answer: bool = False) -> list[tuple[str, str]]:
    """读取并校验 instruction/output，供 eval 等非训练入口复用。"""
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(_read_items(path)):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index + 1} 条必须是 JSON 对象")
        q_raw = item.get("instruction")
        a_raw = item.get("output")
        if not isinstance(q_raw, str) or not q_raw.strip():
            raise ValueError(f"第 {index + 1} 条 instruction 必须是非空字符串")
        if a_raw is not None and not isinstance(a_raw, str):
            raise ValueError(f"第 {index + 1} 条 output 必须是字符串或 null")
        answer = (a_raw or "").strip()
        if require_answer and not answer:
            raise ValueError(f"第 {index + 1} 条 output 不能为空")
        pairs.append((q_raw.strip(), answer))
    if not pairs:
        raise ValueError("数据中没有可评估样本")
    return pairs


def inspect_data(path: Path, tokenizer, *, ctx_len: int = 512) -> DataInspection:
    """检查 NekoQA instruction/output 数据并统计真实 tokenizer 长度。"""
    from .templates import NEKO_QA

    if ctx_len <= 0:
        raise ValueError("ctx_len 必须 > 0")
    items = _read_items(path)
    lengths: list[int] = []
    empty_q = empty_a = truncated = target_lost = 0

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index + 1} 条必须是 JSON 对象")
        q_raw = item.get("instruction")
        a_raw = item.get("output")
        if q_raw is not None and not isinstance(q_raw, str):
            raise ValueError(f"第 {index + 1} 条 instruction 必须是字符串")
        if a_raw is not None and not isinstance(a_raw, str):
            raise ValueError(f"第 {index + 1} 条 output 必须是字符串")
        q = (q_raw or "").strip()
        a = (a_raw or "").strip()
        if not q:
            empty_q += 1
            continue
        if not a:
            empty_a += 1
            continue

        prefix_len = len(tokenizer.encode(NEKO_QA.format_prefix(q=q)))
        target_len = len(tokenizer.encode(NEKO_QA.format_target(a=a)))
        length = prefix_len + target_len  # input_ids 长度(full 还会追加 eos)
        lengths.append(length)
        if length > ctx_len:
            truncated += 1
        # mask 从 prefix_len-1 开始；ctx 至少要覆盖该预测位置。
        if ctx_len < prefix_len:
            target_lost += 1

    if not lengths:
        raise ValueError("没有有效训练样本")
    arr = np.asarray(lengths, dtype=np.float64)
    return DataInspection(
        path=str(path),
        total=len(items),
        valid=len(lengths),
        skipped_empty_question=empty_q,
        skipped_empty_answer=empty_a,
        truncated=truncated,
        target_fully_truncated=target_lost,
        min_tokens=int(arr.min()),
        mean_tokens=round(float(arr.mean()), 1),
        p95_tokens=round(float(np.percentile(arr, 95)), 1),
        max_tokens=int(arr.max()),
        ctx_len=ctx_len,
    )


def inspect_state(path: Path) -> StateInspection:
    """读取 npz/pth 并检查层号、shape、dtype 与数值摘要。"""
    from .export import load_npz_as_numpy, load_pth_as_numpy

    if not path.exists():
        raise ValueError(f"state 文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        states = load_npz_as_numpy(path)
        fmt = "npz"
    elif suffix == ".pth":
        states = load_pth_as_numpy(path)
        fmt = "pth-x070"
    else:
        raise ValueError("state 只支持 .npz / .pth")
    if not states:
        raise ValueError("state 文件不包含任何层")

    indices = sorted(states)
    arrays = [np.asarray(states[i]) for i in indices]
    shapes = [list(a.shape) for a in arrays]
    dtypes = sorted({str(a.dtype) for a in arrays})
    stds = np.asarray([float(a.astype(np.float32).std()) for a in arrays])
    continuous = indices == list(range(len(indices)))
    compatible = continuous and all(a.ndim == 3 and a.shape[-2:] == (64, 64) for a in arrays)
    return StateInspection(
        path=str(path),
        format=fmt,
        layers=len(arrays),
        layer_indices=indices,
        continuous_layers=continuous,
        shapes=shapes,
        dtypes=dtypes,
        std_min=round(float(stds.min()), 6),
        std_mean=round(float(stds.mean()), 6),
        std_max=round(float(stds.max()), 6),
        rwkv7_compatible=compatible,
    )


def doctor_report() -> dict:
    """返回轻量环境报告；单个可选组件失败不会让整个 doctor 崩溃。"""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
    }
    for module_name in ("numpy", "torch", "mlx", "mlx_lm"):
        try:
            module = __import__(module_name)
            report[module_name] = {"ok": True, "version": getattr(module, "__version__", "unknown")}
        except BaseException as exc:
            report[module_name] = {"ok": False, "error": str(exc)}
    try:
        import mlx.core as mx

        report["metal_available"] = bool(mx.metal.is_available())
        if report["metal_available"]:
            info = mx.device_info()
            report["memory_size_gb"] = round(info.get("memory_size", 0) / 1e9, 2)
            report["working_set_gb"] = round(
                info.get("max_recommended_working_set_size", 0) / 1e9, 2
            )
    except BaseException as exc:
        report["metal_available"] = False
        report["metal_error"] = str(exc)
    return report
