"""应用用例层：编排训练、保存和导出，不感知 Typer/SwiftUI。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .events import EventEmitter

StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class TrainingRequest:
    model: Path
    data: Path
    out: Path
    train_config: object
    template: str = "nekoqa"
    test_data: Optional[Path] = None
    test_ratio: float = 0.1
    export_pth: bool = False
    pth_out: Optional[Path] = None


@dataclass(frozen=True)
class TrainingJobResult:
    state_path: Path
    metadata_path: Path
    pth_path: Optional[Path]
    epochs_run: int
    final_loss: float
    final_state_std: float
    elapsed: float


def validate_training_request(request: TrainingRequest) -> None:
    """应用层校验；sidecar/其他客户端不依赖 CLI 也能得到同样保护。"""
    cfg = request.train_config
    if request.template != "nekoqa":
        raise ValueError(f"当前只支持 nekoqa 模板，收到 {request.template!r}")
    if not request.model.is_dir():
        raise ValueError(f"模型目录不存在: {request.model}")
    if not request.data.is_file():
        raise ValueError(f"训练数据不存在: {request.data}")
    if request.test_data is not None and not request.test_data.is_file():
        raise ValueError(f"held-out 数据不存在: {request.test_data}")
    if cfg.lr <= 0 or cfg.lr_floor <= 0 or cfg.lr_floor > cfg.lr:
        raise ValueError("lr/lr_floor 参数范围非法")
    if cfg.warmup < 0 or cfg.ctx_len <= 0 or cfg.epochs <= 0 or cfg.grad_clip <= 0:
        raise ValueError("warmup/ctx_len/epochs/grad_clip 参数范围非法")
    if not 0 < request.test_ratio < 1:
        raise ValueError("test_ratio 必须在 (0, 1) 范围内")
    if request.pth_out is not None and not request.export_pth:
        raise ValueError("pth_out 必须配合 export_pth")


def run_training(
    request: TrainingRequest,
    emitter: EventEmitter,
    *,
    status: Optional[StatusCallback] = None,
) -> TrainingJobResult:
    """执行完整训练 job；所有必需产物落盘后才发 completed。"""
    validate_training_request(request)
    from . import events
    from .core import load_model
    from .data import load_qa_dataset, train_test_split
    from .inspection import inspect_data
    from .metadata import write_state_metadata
    from .train import Trainer, save_state_npz

    notify = status or (lambda _: None)
    cfg = request.train_config
    notify(f"加载模型 {request.model} (patch ops 路径, template={request.template})")
    model, tokenizer = load_model(str(request.model), patch=True)
    model.freeze()

    data_summary = inspect_data(request.data, tokenizer, ctx_len=cfg.ctx_len)
    if data_summary.target_fully_truncated:
        raise ValueError(
            f"有 {data_summary.target_fully_truncated} 条样本的 target 被 ctx_len 完全截断"
        )
    samples = load_qa_dataset(request.data, tokenizer, max_len=cfg.ctx_len)
    notify(f"训练样本: {len(samples)} 条 (template={request.template})")

    held_out = None
    if cfg.early_stop:
        if request.test_data is not None:
            test_summary = inspect_data(request.test_data, tokenizer, ctx_len=cfg.ctx_len)
            if test_summary.target_fully_truncated:
                raise ValueError(
                    f"held-out 有 {test_summary.target_fully_truncated} 条 target 被完全截断"
                )
            held_out = load_qa_dataset(request.test_data, tokenizer, max_len=cfg.ctx_len)
            notify(f"held-out: {len(held_out)} 条 (来自 {request.test_data})")
        else:
            if len(samples) < 2:
                raise ValueError("自动 held-out 至少需要 2 条有效样本")
            samples, held_out = train_test_split(
                samples, test_ratio=request.test_ratio, seed=cfg.seed
            )
            if not samples:
                raise ValueError("held-out 划分后训练集为空，请降低 test_ratio")
            notify(
                f"held-out: {len(held_out)} 条 (从 train 划分 {request.test_ratio:.0%})"
            )

    result = Trainer(model, cfg, emitter).train(samples, held_out)

    request.out.parent.mkdir(parents=True, exist_ok=True)
    save_state_npz(result.states, request.out)
    notify(f"state → {request.out} (std={result.final_state_std:.4f})")

    pth_path = None
    if request.export_pth:
        from .export import export_pth, verify_roundtrip

        pth_path = request.pth_out or request.out.with_suffix(".pth")
        pth_path.parent.mkdir(parents=True, exist_ok=True)
        export_pth({i: result.states[i] for i in result.states}, pth_path)
        ok, message = verify_roundtrip(
            {i: result.states[i] for i in result.states}, pth_path
        )
        notify(f"pth → {pth_path} ({'OK' if ok else 'FAIL'}: {message})")
        if not ok:
            raise RuntimeError(f"pth round-trip 验证失败: {message}")

    metadata_path = write_state_metadata(
        request.out,
        model_path=request.model,
        data_path=request.data,
        template=request.template,
        config=cfg.to_dict(),
        data_stats=data_summary.to_dict(),
        result={
            "epochs_run": result.epochs_run,
            "final_loss": result.final_loss,
            "final_state_std": result.final_state_std,
            "best_held_out_loss": result.best_held_out_loss,
            "elapsed": result.elapsed,
        },
        pth_path=pth_path,
    )
    emitter.emit(
        events.completed(
            str(request.out), result.elapsed, message=f"metadata={metadata_path}"
        )
    )
    return TrainingJobResult(
        state_path=request.out,
        metadata_path=metadata_path,
        pth_path=pth_path,
        epochs_run=result.epochs_run,
        final_loss=result.final_loss,
        final_state_std=result.final_state_std,
        elapsed=result.elapsed,
    )
