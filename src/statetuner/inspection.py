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
    template: str = "qa"

    # M4:16G 机器 bf16 训练的红线(AGENTS.md 内存事实)。
    # step_peak 均值 591 / max 644 顶到削顶线 12.07G。
    # p95 接近 580 → warn(ctx_len 默认 512 已经在红线附近,长样本会触发换页)。
    P95_WARN_THRESHOLD = 580

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def p95_near_limit(self) -> bool:
        """p95 是否逼近 16G bf16 训练红线(M4)。

        红线来自 AGENTS.md:step_peak 均值 591 / max 644,削顶线 12.07G。
        p95 ≥ 580 → 多数样本接近红线,长样本(>600)会触发换页,建议降 ctx_len。
        这是建议性 warn(不 raise),由调用方决定怎么呈现。
        """
        return self.p95_tokens >= self.P95_WARN_THRESHOLD


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
    """检查 QA instruction/output 数据并统计真实 tokenizer 长度。"""
    from .templates import QA

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

        prefix_len = len(tokenizer.encode(QA.format_prefix(q=q)))
        target_len = len(tokenizer.encode(QA.format_target(a=a)))
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


def inspect_standard_jsonl(
    path: Path, tokenizer, *, template: str = "qa", ctx_len: int = 512
) -> DataInspection:
    """检查 importer 标准产物(prompt/response 或 instruction/input/response)。

    与 inspect_data 的区别:字段名契约不同(见 data.load_standard_jsonl)。
    两条数据路径共存:遗留 instruction/output 走 inspect_data,
    importer 产物走本函数。截断/长度统计逻辑一致。
    """
    from .templates import INSTRUCTION, QA

    if ctx_len <= 0:
        raise ValueError("ctx_len 必须 > 0")
    if template not in ("qa", "instruction"):
        raise ValueError(f"标准 jsonl 检查只支持 qa / instruction, 收到 {template!r}")
    tmpl = QA if template == "qa" else INSTRUCTION

    items = _read_items(path)
    lengths: list[int] = []
    empty_q = empty_a = truncated = target_lost = 0

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index + 1} 条必须是 JSON 对象")
        if template == "qa":
            q_raw = item.get("prompt")
            a_raw = item.get("response")
            if q_raw is not None and not isinstance(q_raw, str):
                raise ValueError(f"第 {index + 1} 条 prompt 必须是字符串")
            if a_raw is not None and not isinstance(a_raw, str):
                raise ValueError(f"第 {index + 1} 条 response 必须是字符串")
            q = (q_raw or "").strip()
            a = (a_raw or "").strip()
            if not q:
                empty_q += 1
                continue
            if not a:
                empty_a += 1
                continue
            prefix_len = len(tokenizer.encode(tmpl.format_prefix(q=q)))
        else:  # instruction
            instruction_raw = item.get("instruction")
            input_raw = item.get("input")
            a_raw = item.get("response")
            if instruction_raw is not None and not isinstance(instruction_raw, str):
                raise ValueError(f"第 {index + 1} 条 instruction 必须是字符串")
            if a_raw is not None and not isinstance(a_raw, str):
                raise ValueError(f"第 {index + 1} 条 response 必须是字符串")
            q = (instruction_raw or "").strip()
            inp = input_raw or ""
            a = (a_raw or "").strip()
            if not q:
                empty_q += 1
                continue
            if not a:
                empty_a += 1
                continue
            prefix_len = len(tokenizer.encode(
                tmpl.format_prefix(instruction=q, input=inp)
            ))
        target_len = len(tokenizer.encode(tmpl.format_target(a=a)))
        length = prefix_len + target_len
        lengths.append(length)
        if length > ctx_len:
            truncated += 1
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
        template=template,
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


def validate_state_for_model(state_path: Path, model) -> dict:
    """校验 state 与目标模型匹配并返回已加载的 state dict。

    三重校验（行为与错误信息与历史 _load_checked 一致）：
      - rwkv7_compatible：必须是连续层号的 (H,64,64) RWKV-7 格式
      - 层数与模型层数一致
      - 每层 shape 与模型期望一致([H, head_dim, head_dim])

    返回 {layer_idx: mx.array}，可直接注入 InferenceEngine.generate。
    CLI 的初始加载与 ChatSession 运行中 /state 切换共用此函数。

    model 需暴露 model.args.{hidden_size,head_dim} 与 model.layers（mlx-lm 约定）。
    """
    from .core import _load_state_dict

    info = inspect_state(state_path)
    expected_layers = len(model.layers)
    expected_shape = [
        model.args.hidden_size // model.args.head_dim,
        model.args.head_dim,
        model.args.head_dim,
    ]
    if not info.rwkv7_compatible:
        raise ValueError("state 不是连续层号的 RWKV-7 (H,64,64) 格式")
    if info.layers != expected_layers:
        raise ValueError(
            f"state 层数 {info.layers} 与模型层数 {expected_layers} 不匹配"
        )
    if any(shape != expected_shape for shape in info.shapes):
        raise ValueError(
            f"state shape 与模型不匹配，期望每层 {expected_shape}"
        )
    return _load_state_dict(state_path)


def doctor_report() -> dict:
    """返回轻量环境报告；单个可选组件失败不会让整个 doctor 崩溃。"""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
    }
    for module_name in ("numpy", "ml_dtypes", "mlx", "mlx_lm"):
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
