"""CLI/未来 UI 共用的环境、数据与 state 检查。"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Optional

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
        raise ValueError(f"Data file does not exist: {path}")
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
                raise ValueError(f"Failed to parse JSONL line {lineno}: {exc.msg}") from exc
        return items
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON: {exc.msg}") from exc


def load_qa_pairs(path: Path, *, require_answer: bool = False) -> list[tuple[str, str]]:
    """读取并校验 instruction/output，供 eval 等非训练入口复用。"""
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(_read_items(path)):
        if not isinstance(item, dict):
            raise ValueError(f"Record {index + 1} must be a JSON object")
        q_raw = item.get("instruction")
        a_raw = item.get("output")
        if not isinstance(q_raw, str) or not q_raw.strip():
            raise ValueError(f"Record {index + 1} instruction must be a non-empty string")
        if a_raw is not None and not isinstance(a_raw, str):
            raise ValueError(f"Record {index + 1} output must be a string or null")
        answer = (a_raw or "").strip()
        if require_answer and not answer:
            raise ValueError(f"Record {index + 1} output cannot be empty")
        pairs.append((q_raw.strip(), answer))
    if not pairs:
        raise ValueError("The data contains no evaluable samples")
    return pairs


def inspect_data(path: Path, tokenizer, *, ctx_len: int = 512) -> DataInspection:
    """检查 QA instruction/output 数据并统计真实 tokenizer 长度。"""
    from .templates import QA

    if ctx_len <= 0:
        raise ValueError("ctx_len must be > 0")
    items = _read_items(path)
    lengths: list[int] = []
    empty_q = empty_a = truncated = target_lost = 0

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Record {index + 1} must be a JSON object")
        q_raw = item.get("instruction")
        a_raw = item.get("output")
        if q_raw is not None and not isinstance(q_raw, str):
            raise ValueError(f"Record {index + 1} instruction must be a string")
        if a_raw is not None and not isinstance(a_raw, str):
            raise ValueError(f"Record {index + 1} output must be a string")
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
        raise ValueError("No valid training samples")
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
    return inspect_standard_records(
        _read_items(path), tokenizer, template=template, ctx_len=ctx_len,
        path=str(path),
    )


def inspect_standard_records(
    items: list[Any], tokenizer, *, template: str = "qa", ctx_len: int = 512,
    path: str = "<memory>",
    on_rendered: Optional[Callable[[dict[str, Any]], None]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> DataInspection:
    """检查 importer 标准记录；可流式输出渲染样本供分页缓存使用。

    ``on_rendered`` 每发现一条有效样本就调用一次。调用方可以直接写 JSONL，
    无需把完整预览留在 Python 或 Swift 内存中。``on_progress``
    在每条输入处理完成后收到 ``(current, total)``，由上层决定上报频率。
    """
    from .templates import INSTRUCTION, QA

    if ctx_len <= 0:
        raise ValueError("ctx_len must be > 0")
    if template not in ("qa", "instruction"):
        raise ValueError(f"Standard JSONL inspection supports only qa / instruction; received {template!r}")
    tmpl = QA if template == "qa" else INSTRUCTION

    lengths: list[int] = []
    empty_q = empty_a = truncated = target_lost = 0
    total_items = len(items)

    def report_progress(current: int) -> None:
        if on_progress is not None:
            on_progress(current, total_items)

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Record {index + 1} must be a JSON object")
        if template == "qa":
            q_raw = item.get("prompt")
            a_raw = item.get("response")
            if q_raw is not None and not isinstance(q_raw, str):
                raise ValueError(f"Record {index + 1} prompt must be a string")
            if a_raw is not None and not isinstance(a_raw, str):
                raise ValueError(f"Record {index + 1} response must be a string")
            q = (q_raw or "").strip()
            a = (a_raw or "").strip()
            if not q:
                empty_q += 1
                report_progress(index + 1)
                continue
            if not a:
                empty_a += 1
                report_progress(index + 1)
                continue
            prefix = tmpl.format_prefix(q=q)
        else:  # instruction
            instruction_raw = item.get("instruction")
            input_raw = item.get("input")
            a_raw = item.get("response")
            if instruction_raw is not None and not isinstance(instruction_raw, str):
                raise ValueError(f"Record {index + 1} instruction must be a string")
            if a_raw is not None and not isinstance(a_raw, str):
                raise ValueError(f"Record {index + 1} response must be a string")
            q = (instruction_raw or "").strip()
            inp = input_raw or ""
            a = (a_raw or "").strip()
            if not q:
                empty_q += 1
                report_progress(index + 1)
                continue
            if not a:
                empty_a += 1
                report_progress(index + 1)
                continue
            prefix = tmpl.format_prefix(instruction=q, input=inp)
        target = tmpl.format_target(a=a)
        prefix_len = len(tokenizer.encode(prefix))
        target_len = len(tokenizer.encode(target))
        length = prefix_len + target_len
        lengths.append(length)
        if length > ctx_len:
            truncated += 1
        if ctx_len < prefix_len:
            target_lost += 1
        if on_rendered is not None:
            on_rendered({
                "full_text": prefix + target,
                "prefix_text": prefix,
                "target_text": target,
                "prefix_len": prefix_len,
                "token_count": length,
                "prompt_text": q,
                "response_text": a,
                "truncated": length > ctx_len,
            })
        report_progress(index + 1)

    if not lengths:
        raise ValueError("No valid training samples")
    arr = np.asarray(lengths, dtype=np.float64)
    return DataInspection(
        path=path,
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
        raise ValueError(f"State file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        states = load_npz_as_numpy(path)
        fmt = "npz"
    elif suffix == ".pth":
        states = load_pth_as_numpy(path)
        fmt = "pth-x070"
    else:
        raise ValueError("State supports only .npz / .pth")
    if not states:
        raise ValueError("State file contains no layers")

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
        raise ValueError("State is not in contiguous-layer RWKV-7 (H,64,64) format")
    if info.layers != expected_layers:
        raise ValueError(
            f"State has {info.layers} layers, but the model has {expected_layers}"
        )
    if any(shape != expected_shape for shape in info.shapes):
        raise ValueError(
            f"State shape does not match the model; expected {expected_shape} per layer"
        )
    return _load_state_dict(state_path)


def _module_version(module_name: str, module: Any) -> str:
    """优先读模块版本；namespace package 则回退到安装分发元数据。"""
    module_version = getattr(module, "__version__", None)
    if module_version not in (None, "", "unknown"):
        return str(module_version)
    try:
        return metadata.version(module_name.replace("_", "-"))
    except metadata.PackageNotFoundError:
        return "unknown"


def _sysctl_text(name: str) -> Optional[str]:
    """读取不含隐私的 macOS sysctl 文本字段；失败时静默降级。"""
    if sys.platform != "darwin":
        return None
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _hardware_report() -> dict[str, str]:
    """返回可安全展示/复制的设备字段，不读取序列号或硬件 UUID。"""
    fields = {
        "chip_name": _sysctl_text("machdep.cpu.brand_string"),
        "hardware_model": _sysctl_text("hw.model"),
        "os_build": _sysctl_text("kern.osversion"),
    }
    return {key: value for key, value in fields.items() if value}


def _metal_memory_report(info: dict[str, Any]) -> dict[str, float]:
    """将 MLX device_info 的字节数转换为后端统一的十进制 GB 字段。

    macOS App 若需界面口径，自行从 GB 还原 bytes 后换算为 GiB；后端、CLI、
    事件、日志、训练、缓存和削顶判据不产出或记录 GiB。
    """
    result: dict[str, float] = {}
    memory_size = info.get("memory_size", 0)
    if isinstance(memory_size, (int, float)) and memory_size > 0:
        result["memory_size_gb"] = round(memory_size / 1e9, 2)
    working_set = info.get("max_recommended_working_set_size", 0)
    if isinstance(working_set, (int, float)) and working_set > 0:
        result["working_set_gb"] = round(working_set / 1e9, 2)
    return result


def doctor_report() -> dict:
    """返回轻量环境报告；单个可选组件失败不会让整个 doctor 崩溃。"""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
    }
    os_version = platform.mac_ver()[0] if sys.platform == "darwin" else platform.release()
    if os_version:
        report["os_version"] = os_version
    report.update(_hardware_report())
    for module_name in ("numpy", "ml_dtypes", "mlx", "mlx_lm"):
        try:
            module = __import__(module_name)
            report[module_name] = {
                "ok": True,
                "version": _module_version(module_name, module),
            }
        except BaseException as exc:
            report[module_name] = {"ok": False, "error": str(exc)}
    try:
        import mlx.core as mx

        report["metal_available"] = bool(mx.metal.is_available())
        if report["metal_available"]:
            info = mx.device_info()
            report.update(_metal_memory_report(info))
    except BaseException as exc:
        report["metal_available"] = False
        report["metal_error"] = str(exc)
    return report
