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
    template: str = "qa"
    test_data: Optional[Path] = None
    test_ratio: float = 0.1
    export_pth: bool = False
    pth_out: Optional[Path] = None
    drop_truncated: bool = False  # True = 丢弃截头样本;False(默认)= 截头保尾继续训练


@dataclass(frozen=True)
class TrainingJobResult:
    state_path: Path
    metadata_path: Path
    pth_path: Optional[Path]
    epochs_run: int
    final_loss: float
    final_state_std: float
    elapsed: float


def _has_import_sidecar(data_path: Path) -> bool:
    """数据文件旁是否有 <name>.import.json(importer 产物标记)。

    拆出来便于单测,且 run_training 与 validate 共用同一判定。
    """
    sidecar = data_path.with_name(data_path.stem + data_path.suffix + ".import.json")
    return sidecar.exists()


def validate_training_request(request: TrainingRequest) -> None:
    """应用层校验；sidecar/其他客户端不依赖 CLI 也能得到同样保护。"""
    cfg = request.train_config
    if request.template not in ("qa", "instruction"):
        raise ValueError(
            f"训练模板只支持 qa / instruction，收到 {request.template!r}"
        )
    if not request.model.is_dir():
        raise ValueError(f"模型目录不存在: {request.model}")
    if not request.data.is_file():
        raise ValueError(f"训练数据不存在: {request.data}")
    if request.test_data is not None and not request.test_data.is_file():
        raise ValueError(f"held-out 数据不存在: {request.test_data}")
    # S1:遗留数据(无 importer sidecar)只能走 qa 模板。load_qa_dataset 无条件
    # 按 QA 编码,放行 instruction 会在 metadata 里写假话却按 QA 训练。
    # 在校验阶段就拦下(早于模型加载),便于 Q5 单测且避免无谓 load_model。
    if request.template == "instruction" and not _has_import_sidecar(request.data):
        raise ValueError(
            "遗留数据(无 .import.json sidecar)不支持 instruction 模板;"
            "请先用 `statetuner import` 转标准 jsonl,或显式 --template qa"
        )
    if cfg.lr <= 0 or cfg.lr_floor <= 0 or cfg.lr_floor > cfg.lr:
        raise ValueError("lr/lr_floor 参数范围非法")
    if cfg.warmup < 0 or cfg.ctx_len <= 0 or cfg.epochs <= 0 or cfg.grad_clip <= 0:
        raise ValueError("warmup/ctx-len/epochs/grad_clip 参数范围非法")
    if not 0 < request.test_ratio < 1:
        raise ValueError("test_ratio 必须在 (0, 1) 范围内")
    if request.pth_out is not None and not request.export_pth:
        raise ValueError("pth_out 必须配合 export_pth")
    # 训练精度契约:权重 bf16 + state fp32(docs/decision-precision.md)。
    # int8 量化模型是推理专用产物(state tuning 的 S₀ 叠加到量化权重上会破坏
    # 精度契约)→ 读 config.json 的 quantization 字段早期拦下。
    _config_path = request.model / "config.json"
    if _config_path.is_file():
        import json

        _cfg = json.loads(_config_path.read_text())
        if _cfg.get("quantization") or _cfg.get("quantization_config"):
            raise ValueError(
                "检测到量化模型(config 含 quantization 字段),训练要求 bf16 权重。"
                "请用未量化的模型目录训练(量化模型仅用于推理)。"
            )


def _check_data_warnings(summary, ctx_len: int, notify: StatusCallback) -> None:
    """数据检查的建议性 warn(M4,不 raise)。

    p95 逼近 16G bf16 训练红线(均值 591 / max 644)→ notify 建议:
      - 若 ctx_len < summary.p95_tokens:部分样本被截断(可能丢 target)
      - 若 p95 ≥ 580:即便没截断也在红线附近,长样本会触发换页,建议降 ctx
    """
    if summary.p95_near_limit:
        notify(
            f"⚠ p95_tokens={summary.p95_tokens:.0f} 接近 16G bf16 训练红线"
            f"(均值 591 / max 644,见 AGENTS.md);长样本可能触发换页。"
        )
        if ctx_len >= 580:
            notify(
                f"  建议:--ctx-len 降到 ~480(当前 {ctx_len}),或减小 --cache-limit-gb。"
            )
    if summary.truncated > 0 and ctx_len < summary.p95_tokens:
        notify(
            f"⚠ {summary.truncated} 条样本 > ctx_len={ctx_len} 将被截断"
            f"(p95={summary.p95_tokens:.0f});截头部保尾部 stop(S3),但 target 前段会丢。"
        )


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
    # WKV7 kernel 模式选择(见 docs/decision-fast-wkv7.md):
    #   metal(默认)= Metal checkpoint kernel,6.67× 加速(长序列);梯度经三轮实验验证数值等价 ops。
    #   ops = Python _wkv7_step_ops 循环,慢但可微基线;排查 Metal 路径问题时用 --no-fast-wkv 回退。
    # load_model(patch=True) 内部调 patch_rwkv7_for_train(ops);metal 模式需在 load 前
    # 先 monkeypatch,且传 patch=False 跳过 load_model 的默认 ops patch(否则会覆盖)。
    use_metal = cfg.wkv_mode == "metal"
    if use_metal:
        from .core import patch_rwkv7_for_train_fast
        notify(
            f"加载模型 {request.model} | WKV7: Metal checkpoint kernel "
            f"(chunk={cfg.wkv_chunk}, template={request.template})"
        )
        patch_rwkv7_for_train_fast(chunk=cfg.wkv_chunk)
        model, tokenizer = load_model(str(request.model), patch=False)
    else:
        notify(
            f"加载模型 {request.model} | WKV7: ops 循环 [慢速基线] "
            f"(template={request.template})"
        )
        model, tokenizer = load_model(str(request.model), patch=True)
    model.freeze()

    # 数据分流(§4.3):importer 产物(带 .import.json sidecar)走标准 loader,
    # 遗留数据(instruction/output 字段)走 load_qa_dataset。两条路径都有截断检查。
    # S1:遗留数据只支持 qa 模板已在 validate_training_request 拦下。
    if _has_import_sidecar(request.data):
        from .data import load_standard_jsonl
        from .inspection import inspect_standard_jsonl as _inspect_std
        data_summary = _inspect_std(request.data, tokenizer, template=request.template, ctx_len=cfg.ctx_len)
        _check_data_warnings(data_summary, cfg.ctx_len, notify)
        # 完全截断只警告不阻断:截头保尾仍保留 stop_token,可训练(用户决定要不要练)。
        # drop_truncated 时这些会被丢弃,无需警告。
        if data_summary.target_fully_truncated and not request.drop_truncated:
            notify(
                f"⚠ {data_summary.target_fully_truncated} 条样本 target 被 ctx_len 完全截断"
                f"(target 前段丢失,建议增大 ctx_len 或勾选丢弃超长样本)"
            )
        samples = load_standard_jsonl(
            request.data, tokenizer, template=request.template,
            max_len=cfg.ctx_len, drop_truncated=request.drop_truncated,
        )
        notify(f"训练样本: {len(samples)} 条 (importer 产物, template={request.template})")
    else:
        data_summary = inspect_data(request.data, tokenizer, ctx_len=cfg.ctx_len)
        _check_data_warnings(data_summary, cfg.ctx_len, notify)
        # 完全截断只警告不阻断:截头保尾仍保留 stop_token,可训练(用户决定要不要练)。
        # drop_truncated 时这些会被丢弃,无需警告。
        if data_summary.target_fully_truncated and not request.drop_truncated:
            notify(
                f"⚠ {data_summary.target_fully_truncated} 条样本 target 被 ctx_len 完全截断"
                f"(target 前段丢失,建议增大 ctx_len 或勾选丢弃超长样本)"
            )
        samples = load_qa_dataset(
            request.data, tokenizer, max_len=cfg.ctx_len,
            drop_truncated=request.drop_truncated,
        )
        notify(f"训练样本: {len(samples)} 条 (template=qa)")

    # 有效样本为 0 直接拦下:否则关早停时训练循环一步不跑,静默产出未训练的 state。
    # 常见成因:数据全空 response、导入 0 记录、或 drop_truncated 把样本全丢光。
    if not samples:
        hint = "（勾选了丢弃超长样本，可能已全部丢弃；可增大 ctx_len 或取消丢弃）" \
            if request.drop_truncated else "（请检查数据是否为空或 response 全空）"
        raise ValueError(f"没有有效训练样本{hint}")

    held_out = None
    if cfg.early_stop:
        if request.test_data is not None:
            test_summary = inspect_data(request.test_data, tokenizer, ctx_len=cfg.ctx_len)
            # 只警告不阻断:held-out 截断只影响早停判据精度,不破坏训练本身。
            if test_summary.target_fully_truncated:
                notify(
                    f"⚠ held-out 有 {test_summary.target_fully_truncated} 条 target 被完全截断"
                    f"(早停判据可能略失真,建议增大 ctx_len)"
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


# ── 评估用例 ────────────────────────────────────────────────

# 缺省示例 prompt：无 --data 时的内置演示用例。
# 属于用例（非 CLI），故放在 service 层；猫娘风格默认对齐 qa 训练分布。
# 注：模板渲染交给 run_evaluation 按传入 template + reasoning + think 统一处理，
# 这里只存原始问题。
DEFAULT_EVAL_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("你好呀，宝宝！今天想做什么？", ""),
    ("主人要出门了，你会怎么做？", ""),
    ("能教我用尾巴写字吗？", ""),
)


@dataclass(frozen=True)
class EvaluationRequest:
    """单次评估 job 的全部输入。engine 由调用方注入（模型已加载）。"""

    engine: object  # InferenceEngine，type 用字符串避免 import 循环
    state: object  # StateInput（str 路径 / dict / None）
    template: str
    config: object  # GenerationConfig
    data: Optional[Path] = None
    limit: int = 5
    # 推理方言(Phase 3 §1.1)：reasoning 模型前缀 bos + think 档位。
    # 仅与 qa 模板组合;instruction/raw 传 True 会在 render_prompt 报错。
    reasoning: bool = False
    think: str = "off"


@dataclass(frozen=True)
class EvaluationItem:
    """单条评估结果（结构化，供文本/JSON 双输出派生）。"""

    index: int
    question: str
    reference: str
    generation: dict  # GenerationResult.to_dict()
    text: str  # 已 strip 的输出文本

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "question": self.question,
            "reference": self.reference,
            **self.generation,
            "text": self.text,
        }


@dataclass(frozen=True)
class EvaluationResult:
    items: list[EvaluationItem]

    def to_dict(self) -> dict:
        return {"results": [item.to_dict() for item in self.items]}


def validate_evaluation_request(request: EvaluationRequest) -> None:
    """应用层校验；sidecar/其他客户端不依赖 CLI 也能得到同样保护。"""
    if request.template not in ("qa", "instruction", "raw"):
        raise ValueError(
            f"评估模板只支持 qa / instruction / raw，收到 {request.template!r}"
        )
    if request.think not in ("off", "fast", "on"):
        raise ValueError(f"think 档位只支持 off / fast / on，收到 {request.think!r}")
    if request.think != "off" and not request.reasoning:
        raise ValueError("think 仅在 reasoning 模型上生效(reasoning=False 时必须 off)")
    if request.data is not None and not request.data.is_file():
        raise ValueError(f"评估数据不存在: {request.data}")
    if request.limit <= 0:
        raise ValueError("limit 必须 > 0")
    if request.state is None:
        raise ValueError("评估必须提供 state")


def run_evaluation(request: EvaluationRequest) -> EvaluationResult:
    """对数据集（或内置示例）逐条生成，返回结构化结果。

    编排不感知 Typer：QA 加载、limit 截断、prompt 渲染（按 template）、
    逐条生成循环、结构化结果组装全部在此完成。CLI/sidecar 只负责模型加载、
    engine 构造与输出格式化。
    """
    validate_evaluation_request(request)
    from .inference import render_prompt
    from .inspection import load_qa_pairs

    if request.data is not None:
        pairs = load_qa_pairs(request.data)
    else:
        pairs = list(DEFAULT_EVAL_QUESTIONS)

    items: list[EvaluationItem] = []
    # 同一 template 同时驱动 prompt 渲染与 stop sequences（已在 config 构造时注入），
    # 保证渲染与 stops 同源（历史 bug：render_prompt 硬编码某模板）。
    # reasoning/think 透传给 render_prompt(仅 qa 模板合法)。
    for i, (question, reference) in enumerate(pairs[: request.limit]):
        prompt = render_prompt(
            question,
            request.template,
            reasoning=request.reasoning,
            think=request.think,
        )
        generated = request.engine.generate(
            prompt, state=request.state, config=request.config
        )
        items.append(
            EvaluationItem(
                index=i + 1,
                question=question,
                reference=reference,
                generation=generated.to_dict(),
                text=generated.text.strip(),
            )
        )
    return EvaluationResult(items=items)
