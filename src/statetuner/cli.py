"""statetuner CLI — 训练、推理、导出与输入检查入口。

这是 P1 的对外入口,也是未来 sidecar IPC 的雏形(每个子命令对应一个 IPC handler)。

事件流:train 把结构化事件以 JSON lines 输出到 stdout(--events-file 可同时写文件),
供人类阅读、管道处理,以及未来的 SwiftUI 进度面板消费。

用法:
  statetuner train --model MODELS --data DATA.json --out state.npz --export-pth
  statetuner eval --model MODELS --state state.npz --data test.json
  statetuner export --state state.npz --out state.pth
  statetuner preview --model MODELS --state state.npz --prompt "你好" --ab
  statetuner doctor
  statetuner data-info --model MODELS --data DATA.json
  statetuner state-info --state state.npz
  statetuner chat --model MODELS --state state.npz
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import sys

import typer

app = typer.Typer(
    name="statetuner",
    help="RWKV-7 state tuning for Mac — train, export, and preview init states.",
    no_args_is_help=True,
    add_completion=False,
)


def _bad_input(exc: Exception) -> None:
    """把可预期的输入错误转成简洁 CLI 错误。"""
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(2)


def _apply_cache_limit(spec: Optional[str]) -> None:
    """CLI 包装(T2):runtime.apply_cache_limit 出错 → _bad_input(typer.Exit 2)。

    逻辑下沉到 runtime.py,cli/serve 共用;本函数只负责把 ValueError 翻译成
    CLI 的退出码(serve 走 ProtocolError,不经过这里)。
    """
    from .runtime import apply_cache_limit

    try:
        apply_cache_limit(spec)
    except ValueError as exc:
        _bad_input(exc)


@app.command("doctor")
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Write structured JSON to stdout"),
):
    """Check Python, MLX, Metal, and export dependencies."""
    from .inspection import doctor_report

    report = doctor_report()
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False))
        return
    typer.echo(f"Python: {report['python']}")
    typer.echo(f"Platform: {report['platform']} ({report['machine']})")
    if report.get("chip_name"):
        typer.echo(
            f"Chip: {report['chip_name']}"
            + (f" ({report['hardware_model']})" if report.get("hardware_model") else "")
        )
    typer.echo(f"Apple Silicon: {'✓' if report['apple_silicon'] else '✗'}")
    for name in ("mlx", "mlx_lm", "ml_dtypes", "numpy"):
        info = report[name]
        typer.echo(f"{name}: {'✓ ' + info.get('version', '') if info['ok'] else '✗ ' + info['error']}")
    typer.echo(f"Metal: {'✓' if report.get('metal_available') else '✗'}")
    if report.get("metal_available"):
        typer.echo(
            f"Memory: physical={report.get('memory_size_gb', 0):g}GB "
            f"working_set={report.get('working_set_gb', 0):.2f}GB"
        )


@app.command("data-info")
def data_info(
    model: Path = typer.Option(..., "--model", "-m", help="HF model directory (tokenizer source)"),
    data: Path = typer.Option(..., "--data", "-d", help="NekoQA JSON/JSONL"),
    ctx_len: int = typer.Option(512, "--ctx-len"),
    template: str = typer.Option(
        "auto", "--template", help="auto (from sidecar) | qa | instruction",
    ),
    json_output: bool = typer.Option(False, "--json", help="Write structured JSON to stdout"),
):
    """Inspect data fields, token lengths, and truncation with the real tokenizer."""
    if not model.is_dir():
        _bad_input(ValueError(f"Model directory does not exist: {model}"))
    if not data.is_file():
        _bad_input(ValueError(f"Data file does not exist: {data}"))
    if ctx_len <= 0:
        _bad_input(ValueError("--ctx-len must be > 0"))
    if template not in ("auto", "qa", "instruction"):
        _bad_input(ValueError("--template supports only auto / qa / instruction"))

    from .inspection import inspect_data, inspect_standard_jsonl
    from .service import _has_import_sidecar
    from mlx_lm.utils import load_tokenizer

    typer.echo(f"# Loading tokenizer: {model}", err=True)
    try:
        tok = load_tokenizer(
            str(model), tokenizer_config_extra={"trust_remote_code": True}
        )
        if _has_import_sidecar(data):
            selected_template = template
            if selected_template == "auto":
                sidecar = data.with_name(data.stem + data.suffix + ".import.json")
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                selected_template = payload.get("result", {}).get("template", "qa")
            result = inspect_standard_jsonl(
                data, tok, template=selected_template, ctx_len=ctx_len,
            )
        else:
            if template == "instruction":
                raise ValueError(
                    "Legacy data without an .import.json sidecar can be inspected only with the qa template"
                )
            result = inspect_data(data, tok, ctx_len=ctx_len)
    except (OSError, ValueError, TypeError) as exc:
        _bad_input(exc)
    payload = result.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
        return
    typer.echo(f"Data: {result.path}")
    typer.echo(
        f"Samples: total={result.total} valid={result.valid} "
        f"empty_q={result.skipped_empty_question} empty_a={result.skipped_empty_answer}"
    )
    typer.echo(
        f"tokens: min={result.min_tokens} mean={result.mean_tokens:.1f} "
        f"p95={result.p95_tokens:.1f} max={result.max_tokens}"
    )
    typer.echo(
        f"ctx={result.ctx_len}: truncated={result.truncated} "
        f"target_fully_truncated={result.target_fully_truncated}"
    )


@app.command("state-info")
def state_info(
    state: Path = typer.Option(..., "--state", "-s", help="state npz/pth"),
    json_output: bool = typer.Option(False, "--json", help="Write structured JSON to stdout"),
):
    """Inspect state format, layer indices, shapes, dtypes, and standard deviation."""
    from .inspection import inspect_state

    try:
        result = inspect_state(state)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        _bad_input(exc)
    payload = result.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
        return
    typer.echo(f"state: {result.path} ({result.format})")
    typer.echo(
        f"layers={result.layers} continuous={'yes' if result.continuous_layers else 'no'} "
        f"rwkv7_compatible={'yes' if result.rwkv7_compatible else 'no'}"
    )
    typer.echo(f"shapes={result.shapes} dtypes={result.dtypes}")
    typer.echo(
        f"std: min={result.std_min:.6f} mean={result.std_mean:.6f} max={result.std_max:.6f}"
    )


@app.command()
def train(
    model: Path = typer.Option(..., "--model", "-m", help="Converted HF model directory"),
    data: Path = typer.Option(..., "--data", "-d", help="Training data in JSON/JSONL format"),
    test_data: Optional[Path] = typer.Option(
        None, "--test-data", help="Held-out data for early stopping; defaults to a split from training data"
    ),
    out: Path = typer.Option(
        Path("state.npz"), "--out", "-o", help="Output training state (.npz)"
    ),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate (default 0.0001; 1.0 diverges)"),
    lr_floor: float = typer.Option(1e-5, "--lr-floor", help="Cosine-decay floor (default 0.00001)"),
    warmup: int = typer.Option(50, "--warmup", help="Warmup steps (default 50)"),
    ctx_len: int = typer.Option(512, "--ctx-len", help="Context length"),
    epochs: int = typer.Option(5, "--epochs", help="Epoch count (default 5; maximum when early stopping is enabled)"),
    grad_clip: float = typer.Option(1.0, "--grad-clip"),
    log_every: int = typer.Option(
        1, "--log-every",
        help="Emit a step event every N steps (default 1; event overhead is negligible)",
    ),
    early_stop: bool = typer.Option(True, "--early-stop/--no-early-stop", help="Held-out early stopping"),
    drop_truncated: bool = typer.Option(
        False, "--drop-truncated/--keep-truncated",
        help="Overlength samples: drop them, or trim the start and keep the end (default)",
    ),
    patience: int = typer.Option(3, "--patience", help="Early-stopping patience (stop after N checks without improvement)"),
    test_ratio: float = typer.Option(0.1, "--test-ratio", help="Fraction split from training data when --test-data is absent"),
    checkpoint_dir: Optional[Path] = typer.Option(None, "--checkpoint-dir", help="Checkpoint directory"),
    checkpoint_every: int = typer.Option(2, "--checkpoint-every", help="Save every N epochs"),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Resume from checkpoint"),
    events_file: Optional[Path] = typer.Option(
        None, "--events-file", help="Write training event JSON Lines to a file (also emitted to stdout)"
    ),
    export_pth: bool = typer.Option(False, "--export-pth", help="Export .pth after training"),
    pth_out: Optional[Path] = typer.Option(None, "--pth-out", help="Exported PTH path (defaults to the output name with .pth)"),
    template: str = typer.Option(
        "qa", "--template",
        help="Task template: qa (role-play QA, default) | instruction",
    ),
    seed: int = typer.Option(42, "--seed"),
    cache_limit_gb: Optional[str] = typer.Option(
        "auto", "--cache-limit-gb",
        help="MLX buffer-cache limit; auto=25% of physical memory (default), or a GB value. Applied before model loading.",
    ),
    fast_wkv: bool = typer.Option(
        True, "--fast-wkv/--no-fast-wkv",
        help="Use the WKV7 Metal checkpoint kernel (default; 6.67x faster on long sequences; see "
             "docs/decision-fast-wkv7.md). --no-fast-wkv falls back to the slow differentiable Python ops loop.",
    ),
    fast_wkv_chunk: int = typer.Option(
        16, "--fast-wkv-chunk",
        help="Backward chunk size for the Metal checkpoint kernel (32/16/8; default 16). Smaller is more accurate but reconstructs more often.",
    ),
):
    """Train an initial state and write JSON Lines events to stdout.

    Training accepts only the qa and instruction templates. The reasoning dialect and
    think level are inference-only concepts, so training targets never contain thinking
    markup.
    """
    if not model.is_dir():
        _bad_input(ValueError(f"Model directory does not exist: {model}"))
    if not data.is_file():
        _bad_input(ValueError(f"Training data does not exist: {data}"))
    if test_data is not None and not test_data.is_file():
        _bad_input(ValueError(f"Held-out data does not exist: {test_data}"))
    if lr <= 0:
        _bad_input(ValueError("--lr must be > 0"))
    if lr_floor <= 0 or lr_floor > lr:
        _bad_input(ValueError("--lr-floor must be > 0 and <= --lr"))
    if warmup < 0 or ctx_len <= 0 or epochs <= 0 or grad_clip <= 0:
        _bad_input(ValueError("warmup/ctx-len/epochs/grad-clip parameters are out of range"))
    if not 0 < test_ratio < 1:
        _bad_input(ValueError("--test-ratio must be in the range (0, 1)"))
    if patience <= 0 or checkpoint_every <= 0:
        _bad_input(ValueError("--patience and --checkpoint-every must be > 0"))
    if log_every < 1:
        _bad_input(ValueError("--log-every must be >= 1"))
    if pth_out is not None and not export_pth:
        _bad_input(ValueError("--pth-out requires --export-pth"))
    if template not in ("qa", "instruction"):
        _bad_input(ValueError(
            f"--template supports only qa / instruction; received {template!r}"
            " (reasoning/think are inference-only parameters)"
        ))

    # cache_limit 必须在 load_model 前生效(mem_probe 验证过的时序)。
    _apply_cache_limit(cache_limit_gb)

    import mlx.core as mx

    from . import events
    from .service import TrainingRequest, run_training
    from .train import TrainConfig

    cfg = TrainConfig(
        lr=lr, lr_floor=lr_floor, warmup=warmup, ctx_len=ctx_len, epochs=epochs,
        # std 健康区间未标定：只记录，不使用旧 1.0 阈值报警。
        grad_clip=grad_clip, max_state_std=None,
        log_every=log_every,
        early_stop=early_stop, early_stop_patience=patience,
        checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every,
        resume=resume, seed=seed,
        wkv_mode="metal" if fast_wkv else "ops",
        wkv_chunk=fast_wkv_chunk,
    )

    request = TrainingRequest(
        model=model,
        data=data,
        out=out,
        train_config=cfg,
        template=template,
        test_data=test_data,
        test_ratio=test_ratio,
        export_pth=export_pth,
        pth_out=pth_out,
        drop_truncated=drop_truncated,
    )

    def _status(message: str) -> None:
        typer.echo(f"# {message}", err=True)

    with events.EventEmitter(file=events_file) as em:
        try:
            result = run_training(request, em, status=_status)
        except KeyboardInterrupt:
            em.emit(events.cancelled())
            raise typer.Exit(130)
        except typer.Exit:
            raise
        except Exception as exc:
            em.emit(events.failed(str(exc), path=str(out)))
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from None

    # 内存口径(runtime.py):用 phys_footprint 峰值(== Activity Monitor「内存」列,
    # 含 Metal 的 IOKit 映射)。旧实现用 ru_maxrss 漏报 ~7x(不含 IOKit);
    # mx.get_peak_memory() 更不准(只算 MLX allocator)。见 runtime.py docstring。
    from .runtime import memory_report

    snap = memory_report()
    peak_gb = snap.peak_footprint_gb or snap.rss_gb or 0.0
    typer.echo(
        f"# Complete: epochs={result.epochs_run} loss={result.final_loss:.4f} "
        f"std={result.final_state_std:.4f} peak_mem={peak_gb:.2f}GB "
        f"elapsed={result.elapsed:.1f}s",
        err=True,
    )


@app.command()
def eval(
    model: Path = typer.Option(..., "--model", "-m", help="HF model directory"),
    state: Path = typer.Option(..., "--state", "-s", help="State file (npz or pth)"),
    data: Optional[Path] = typer.Option(
        None, "--data", "-d", help="Evaluation data (JSONL or JSON array); defaults to built-in examples"
    ),
    max_tokens: int = typer.Option(70, "--max-tokens"),
    temperature: float = typer.Option(0.8, "--temperature", help="Sampling temperature; 0=greedy"),
    top_p: float = typer.Option(0.9, "--top-p", help="Nucleus-sampling threshold"),
    seed: int = typer.Option(42, "--seed", help="Sampling seed"),
    template: str = typer.Option(
        "qa", "--template",
        help="Task template: qa (role-play QA, default) | instruction | raw",
    ),
    reasoning: bool = typer.Option(
        False, "--reasoning",
        help="Add the reasoning-model BOS prefix and --think tags. Valid only with the qa template.",
    ),
    think: str = typer.Option(
        "off", "--think",
        help="Thinking mode (only with --reasoning): off | fast (skip thinking) | on (full thinking)",
    ),
    limit: int = typer.Option(5, "--limit", help="Maximum number of outputs (default 5)"),
    json_output: bool = typer.Option(False, "--json", help="Write structured JSON to stdout"),
    cache_limit_gb: Optional[str] = typer.Option(
        "auto", "--cache-limit-gb",
        help="MLX buffer-cache limit; auto=25% of physical memory (default), or a GB value. Applied before model loading.",
    ),
):
    """Evaluate a state by generating one result per dataset record.

    Prompts and stop sequences come from the selected training template. --reasoning and
    --think are valid only with the qa template.
    """
    if template not in ("qa", "instruction", "raw"):
        raise typer.BadParameter(
            f"--template supports only qa / instruction / raw; received {template!r}"
        )
    if think not in ("off", "fast", "on"):
        raise typer.BadParameter(
            f"--think supports only off / fast / on; received {think!r}"
        )
    if think != "off" and not reasoning:
        raise typer.BadParameter("--think is valid only with --reasoning")
    if not model.is_dir():
        _bad_input(ValueError(f"Model directory does not exist: {model}"))
    if not state.is_file():
        _bad_input(ValueError(f"State file does not exist: {state}"))
    if data is not None and not data.is_file():
        _bad_input(ValueError(f"Evaluation data does not exist: {data}"))
    if max_tokens <= 0 or temperature < 0 or not 0 < top_p <= 1:
        _bad_input(ValueError("max-tokens/temperature/top-p parameters are out of range"))
    if limit <= 0:
        raise typer.BadParameter("--limit must be > 0")

    # cache_limit 必须在 load_model 前生效(mem_probe 验证过的时序)。
    _apply_cache_limit(cache_limit_gb)

    from .core import load_model
    from .inference import GenerationConfig, InferenceEngine, with_template_stops
    from .service import EvaluationRequest, run_evaluation

    cfg = with_template_stops(
        GenerationConfig(
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed
        ),
        template,
    )

    typer.echo(
        f"# Loading model {model} (kernel path, template={template}"
        f"{' reasoning=' + think if reasoning else ''})", err=True
    )
    mdl, tok = load_model(model, patch=False)
    engine = InferenceEngine(mdl, tok)

    request = EvaluationRequest(
        engine=engine,
        state=str(state),
        template=template,
        config=cfg,
        data=data,
        limit=limit,
        reasoning=reasoning,
        think=think,
    )
    try:
        result = run_evaluation(request)
    except (OSError, ValueError, TypeError) as exc:
        _bad_input(exc)

    if json_output:
        typer.echo(json.dumps(result.to_dict(), ensure_ascii=False))
        return
    for item in result.items:
        typer.echo(f"[{item.index}] {item.question}")
        if item.reference:
            typer.echo(f"    REF: {item.reference[:100]}")
        typer.echo(f"    OUT: {item.text[:300]}")
        typer.echo()


@app.command()
def export(
    state: Path = typer.Option(..., "--state", "-s", help="State npz (internal P0 format)"),
    out: Path = typer.Option(..., "--out", "-o", help="Output .pth path"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify round-trip after export"),
    deep: bool = typer.Option(
        False, "--deep",
        help="Deep validation: verify that MLX generation with mounted PTH matches direct state injection. Requires --model.",
    ),
    model: Optional[Path] = typer.Option(
        None, "--model", "-m",
        help="Model directory for deep validation (required with --deep; must match training/inference).",
    ),
    prompt: str = typer.Option(
        "User: Hello\n\nAssistant:", "--prompt",
        help="Probe prompt for deep validation (defaults to a QA prefix).",
    ),
):
    """Export an NPZ state as a .pth file that RWKV Runner can mount.

    --deep verifies end-to-end mount equivalence and requires a model plus tokenizer.
    """
    if not state.is_file():
        _bad_input(ValueError(f"State file does not exist: {state}"))
    if state.suffix.lower() != ".npz":
        _bad_input(ValueError("export --state must be an .npz file"))
    if deep and not model:
        _bad_input(ValueError("--deep requires --model for end-to-end validation"))
    if deep and model is not None and not model.is_dir():
        _bad_input(ValueError(f"Model directory does not exist: {model}"))
    out.parent.mkdir(parents=True, exist_ok=True)

    from .export import export_pth as _export, load_npz_as_numpy, verify_mount_equivalence, verify_roundtrip

    typer.echo(f"# Reading {state}", err=True)
    states = load_npz_as_numpy(state)
    typer.echo(f"# Exporting {len(states)} layers -> {out}", err=True)
    _export(states, out)

    if verify:
        ok, msg = verify_roundtrip(states, out)
        typer.echo(f"# round-trip: {'✓' if ok else '✗'} {msg}", err=True)
        if not ok:
            raise typer.Exit(1)

    if deep:
        # T6:接通此前无调用方的 verify_mount_equivalence(本仓库最强的正确性证明)。
        # 清死代码 + 给 State 库的"深度校验"勾选项一个 CLI 对应物。
        _apply_cache_limit("auto")  # 模型加载前生效(时序铁律)
        from .core import load_model

        typer.echo(f"# Deep validation: loading {model} (end-to-end mount equivalence)", err=True)
        mdl, tok = load_model(model, patch=False)
        ok, msg = verify_mount_equivalence(mdl, tok, states, out, prompt)
        typer.echo(f"# deep mount: {'✓' if ok else '✗'} {msg}", err=True)
        if not ok:
            raise typer.Exit(1)

    typer.echo(f"✓ {out}")


def _render_reply_lines(ui_console, lines: list[str]) -> None:
    """渲染 ChatReply.lines:A/B 标题行原样,内容行走 markdown。

    A/B 输出格式("=== 有 state ===" / text / summary / "=== 无 state ===" / ...):
      - 以 === 开头的分隔行 → dim 原样打印(不渲染 markdown)。
      - 以 [stop= 开头的摘要行 → dim。
      - 其余 → markdown 渲染(助手回复正文)。
    """
    from . import console as ui

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("===") or stripped.startswith("[stop="):
            ui_console.print(line, style="dim")
        else:
            ui_console.print(ui.render_markdown(line))


@app.command()
def chat(
    model: Path = typer.Option(..., "--model", "-m", help="HF model directory"),
    state: Optional[Path] = typer.Option(None, "--state", "-s", help="Initial state (npz/pth)"),
    max_tokens: int = typer.Option(300, "--max-tokens", help="Maximum generated tokens per turn"),
    temperature: float = typer.Option(1.2, "--temperature", help="Sampling temperature; 0=greedy"),
    top_p: float = typer.Option(0.5, "--top-p"),
    seed: int = typer.Option(42, "--seed"),
    presence_penalty: float = typer.Option(
        0.4, "--presence",
        help="Fixed penalty for previously seen tokens (ChatRWKV default 0.4); 0 disables it.",
    ),
    frequency_penalty: float = typer.Option(
        0.4, "--frequency",
        help="Penalty accumulated by token frequency (ChatRWKV default 0.4); 0 disables it.",
    ),
    penalty_decay: float = typer.Option(
        0.996, "--penalty-decay",
        help="Exponential decay for repetition-history counts (ChatRWKV default 0.996).",
    ),
    ab: bool = typer.Option(False, "--ab", help="Enable A/B at startup"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream generated text"),
    template: str = typer.Option(
        "qa", "--template",
        help="Task template: qa (role-play QA, default) | instruction | raw",
    ),
    reasoning: bool = typer.Option(
        False, "--reasoning",
        help="Add the reasoning-model BOS prefix and --think tags. Valid only with qa; required for G1 reasoning models.",
    ),
    think: str = typer.Option(
        "off", "--think",
        help="Thinking mode with --reasoning: off (direct), fast (skip thinking), or on (full thinking). G1 recommends fast.",
    ),
    cache_limit_gb: Optional[str] = typer.Option(
        "auto", "--cache-limit-gb",
        help="MLX buffer-cache limit; auto=25% of physical memory (default), or a GB value. Applied before model loading.",
    ),
):
    """Run an interactive model session with live state switching."""
    if not model.is_dir():
        _bad_input(ValueError(f"Model directory does not exist: {model}"))
    if state is not None and not state.is_file():
        _bad_input(ValueError(f"State file does not exist: {state}"))
    if template not in ("raw", "qa", "instruction"):
        _bad_input(ValueError("--template supports only raw / qa / instruction"))
    if think not in ("off", "fast", "on"):
        _bad_input(ValueError("--think supports only off / fast / on"))
    if think != "off" and not reasoning:
        _bad_input(ValueError("--think is valid only with --reasoning"))
    if max_tokens <= 0 or temperature < 0 or not 0 < top_p <= 1:
        _bad_input(ValueError("max-tokens/temperature/top-p parameters are out of range"))
    if ab and state is None:
        _bad_input(ValueError("--ab requires --state"))

    # cache_limit 必须在 load_model 前生效(mem_probe 验证过的时序)。
    _apply_cache_limit(cache_limit_gb)

    from .chat import ChatSession
    from .core import load_model
    from .inference import GenerationConfig, InferenceEngine
    from .inspection import validate_state_for_model

    typer.echo(f"# Loading model {model} (kept resident for this session)", err=True)
    mdl, tok = load_model(model, patch=False)
    engine = InferenceEngine(mdl, tok)

    # state 校验+加载下沉到 inspection.validate_state_for_model；
    # CLI 与 ChatSession 运行中 /state 切换共用同一校验逻辑。
    def _load_checked(path: Path):
        return validate_state_for_model(path, mdl)

    loaded_state = None
    if state is not None:
        try:
            loaded_state = _load_checked(state)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            _bad_input(exc)

    # think=on 时 G1 系列思考段实测 ~400 token,默认 300 撑不到 </think> 闭合
    # 就被截断(渲染兜底会把已生成内容当思考 dim 显示,但 answer 会缺失)。
    # 启发式:think=on 且用户未显式调过 max_tokens(仍是默认 300)→ 抬到 800。
    # 用户显式传 --max-tokens(即便传 300)即视为知情,不覆盖。
    effective_max_tokens = max_tokens
    if reasoning and think == "on" and max_tokens == 300:
        effective_max_tokens = 800

    session = ChatSession(
        engine,
        config=GenerationConfig(
            max_tokens=effective_max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            penalty_decay=penalty_decay,
        ),
        template=template,
        reasoning=reasoning,
        think=think,
        state=loaded_state,
        state_label=str(state) if state else None,
        state_loader=_load_checked,
        ab=ab,
    )

    from . import console as ui

    ui_console = ui.make_console()
    ui_console.print(
        "Interactive mode started. Multi-turn qa uses cache continuation; enter [bold]/help[/bold] for commands."
    )
    if not reasoning and template == "qa":
        ui_console.print(
            "[dim]# Tip: for G1 reasoning models, use --reasoning --think fast "
            "(see docs/g1g-decode-alignment.md).[/dim]",
            style=None,
        )
    # 启动横幅用紧凑两行(· 分隔);/config 命令仍走详尽表格。
    ui_console.print(ui.render_config_compact(session.config_groups(brief=True)))

    while True:
        try:
            ui_console.print(ui.user_prompt_label(), end="")
            line = input()
        except (EOFError, KeyboardInterrupt):
            ui_console.print("\nSession ended.")
            break
        use_stream = stream and not session.ab and not line.lstrip().startswith("/")

        if use_stream:
            # 流式:Live 内增量显示纯文本(不逐 token 解析 markdown,避免闪烁),
            # 结束后用 payload 里的完整文本一次性渲染 markdown 面板 + dim 摘要。
            # 前导换行清洗(reasoning 方言)由 ChatSession._wrap_stream_callback 负责。
            from rich.live import Live
            from rich.text import Text

            think_on = reasoning and think == "on"

            if think_on:
                # think=on phase 状态机:思考段实时以 dim italic 流,
                # </think> 闭合后切正常风格继续流 answer。
                # 用累积 buffer + split_thinking 判断当前是否已越过闭合标签;
                # tokenizer 可能把 </think> 拆多 token,基于累积文本判断保证正确
                # (切换最多延迟 1-2 token,可接受)。
                view = Text()
                accum = [""]  # 已生成文本总累积(含 think 段)

                def _on_text(chunk: str) -> None:
                    accum[0] += chunk
                    thinking, answer = ui.split_thinking(accum[0])
                    if "</think>" in accum[0]:
                        # 已越过闭合:think 段已全部确定,显示 thinking(dim)+ answer 增量
                        # 重建 view(thinking 一次 + answer 当前累积),简单且无 phase 漏切。
                        view.truncate(0)
                        if thinking:
                            view.append(thinking, style="dim italic")
                            view.append("\n\n", style="dim italic")
                        view.append(answer)
                    else:
                        # 还在 think 段:增量全以 dim italic 流
                        view.append(chunk, style="dim italic")
                    live.update(view, refresh=True)

                with Live(
                    Text(""), console=ui_console, refresh_per_second=15, transient=True
                ) as live:
                    reply = session.handle(line, on_text=_on_text)
                full_text = (reply.payload or {}).get("text", accum[0])
                thinking, answer = ui.split_thinking(full_text)
                if thinking:
                    ui_console.print(ui.render_thinking_panel(thinking))
                if answer:
                    ui_console.print(ui.render_assistant_panel(answer))
                elif thinking:
                    # think 段未闭合(被 max_tokens 截断):思考显示出来了但没有正式回答。
                    ui_console.print(
                        "[dim]Warning: thinking was truncated by max_tokens before </think>; "
                        "increase /max-tokens and retry.[/dim]"
                    )
            else:
                buf = [""]

                def _on_text(chunk: str) -> None:
                    buf[0] += chunk
                    live.update(Text(buf[0]), refresh=True)

                # transient=True:流式预览在 Live 退出时清掉,避免和最终 markdown 面板重复。
                with Live(
                    Text(""), console=ui_console, refresh_per_second=15, transient=True
                ) as live:
                    reply = session.handle(line, on_text=_on_text)
                full_text = (reply.payload or {}).get("text", buf[0])
                ui_console.print(ui.render_assistant_panel(full_text))
            for output_line in reply.lines:
                ui_console.print(ui.dim_summary(output_line))
        else:
            reply = session.handle(line)
            # 命令输出:/help → 表格;/config → 表格;A/B → markdown 分段;其他 → 原样
            if line.lstrip().startswith("/help"):
                table = ui.render_help_table(reply.lines)
                if table is not None:
                    ui_console.print(table)
                else:
                    for output_line in reply.lines:
                        ui_console.print(output_line)
            elif line.lstrip().startswith("/config"):
                ui_console.print(ui.render_config_table(session.config_groups()))
            else:
                # A/B 或普通命令:逐行渲染(A/B 标题行原样,内容走 markdown)
                _render_reply_lines(ui_console, reply.lines)
        if reply.exit:
            break


@app.command()
def preview(
    model: Path = typer.Option(..., "--model", "-m", help="HF model directory"),
    state: Optional[Path] = typer.Option(
        None, "--state", "-s", help="State file (npz/pth); defaults to no-state baseline"
    ),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Input text or question"),
    max_tokens: int = typer.Option(80, "--max-tokens"),
    temperature: float = typer.Option(0.0, "--temperature", help="Sampling temperature; 0=greedy"),
    top_p: float = typer.Option(0.9, "--top-p", help="Nucleus-sampling threshold"),
    seed: int = typer.Option(42, "--seed", help="Sampling seed"),
    ab: bool = typer.Option(False, "--ab", help="A/B comparison: with state vs no-state baseline"),
    template: str = typer.Option(
        "raw", "--template",
        help="Prompt wrapper: raw (unchanged, default) | qa | instruction",
    ),
    reasoning: bool = typer.Option(
        False, "--reasoning",
        help="Add the reasoning-model BOS prefix and --think tags. Valid only with qa.",
    ),
    think: str = typer.Option(
        "off", "--think",
        help="Thinking mode with --reasoning: off (direct), fast (skip thinking), or on (full thinking)",
    ),
    json_output: bool = typer.Option(False, "--json", help="Write structured JSON to stdout"),
    stream: bool = typer.Option(False, "--stream", help="Stream single-generation text"),
):
    """Generate a preview with an optional state; --ab compares state and baseline.

    --template controls both prompt rendering and stop sequences. --reasoning and --think
    are valid only with the qa template. A temperature of 0 uses greedy decoding.
    """
    if template not in ("raw", "qa", "instruction"):
        raise typer.BadParameter(
            f"--template supports only raw / qa / instruction; received {template!r}"
        )
    if think not in ("off", "fast", "on"):
        raise typer.BadParameter(
            f"--think supports only off / fast / on; received {think!r}"
        )
    if think != "off" and not reasoning:
        raise typer.BadParameter("--think is valid only with --reasoning")
    if ab and state is None:
        raise typer.BadParameter("--ab requires --state")
    if stream and (ab or json_output):
        raise typer.BadParameter("--stream cannot be combined with --ab or --json")
    if not model.is_dir():
        _bad_input(ValueError(f"Model directory does not exist: {model}"))
    if state is not None and not state.is_file():
        _bad_input(ValueError(f"State file does not exist: {state}"))
    if not prompt:
        _bad_input(ValueError("--prompt cannot be empty"))
    if max_tokens <= 0 or temperature < 0 or not 0 < top_p <= 1:
        _bad_input(ValueError("max-tokens/temperature/top-p parameters are out of range"))

    from .core import load_model
    from .inference import (
        GenerationConfig, InferenceEngine, render_prompt, with_template_stops,
    )

    cfg = with_template_stops(
        GenerationConfig(
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed
        ),
        template,
    )

    # 按模板包装 prompt(与训练/eval 同源)
    wrapped = render_prompt(prompt, template, reasoning=reasoning, think=think)

    typer.echo(
        f"# Loading model {model} (template={template}"
        f"{' reasoning=' + think if reasoning else ''})", err=True
    )
    mdl, tok = load_model(model, patch=False)
    engine = InferenceEngine(mdl, tok)

    if ab:
        result = engine.compare(wrapped, state=str(state), config=cfg)
        if json_output:
            typer.echo(json.dumps(result.to_dict(), ensure_ascii=False))
            return
        typer.echo("=== With state ===")
        typer.echo(result.with_state.text)
        typer.echo(
            f"[stop={result.with_state.stop_reason}, tokens={result.with_state.token_count}]"
        )
        typer.echo("=== Without state (baseline) ===")
        typer.echo(result.baseline.text)
        typer.echo(
            f"[stop={result.baseline.stop_reason}, tokens={result.baseline.token_count}]"
        )
    else:
        callback = None
        if stream:
            def callback(chunk: str) -> None:
                typer.echo(chunk, nl=False)

        result = engine.generate(
            wrapped,
            state=(str(state) if state else None),
            config=cfg,
            on_text=callback,
        )
        if json_output:
            typer.echo(json.dumps(result.to_dict(), ensure_ascii=False))
        elif stream:
            typer.echo()
            typer.echo(result.summary_line())
        else:
            typer.echo(result.text)
            typer.echo(result.summary_line())


@app.command("convert-model")
def convert_model(
    rwkv7: Path = typer.Option(..., "--rwkv7", help="Native BlinkDL RWKV-7 .pth"),
    out: Path = typer.Option(..., "--out", "-o", help="Output HF model directory"),
    precision: str = typer.Option("bf16", "--precision", help="bf16 (recommended) | fp16 | fp32"),
    reference: Optional[Path] = typer.Option(
        None, "--reference", help="Optional live safetensors validation template with the same architecture",
    ),
    tokenizer_src: Optional[Path] = typer.Option(
        None, "--tokenizer-src", help="Optional custom World tokenizer directory",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow writing to a non-empty output directory"),
):
    """Convert a native RWKV-7 .pth file to an HF safetensors model directory.

    stdout contains only tool-task JSON Lines; human-readable logs go to stderr.
    """
    if not rwkv7.is_file():
        _bad_input(ValueError(f"Native model file does not exist: {rwkv7}"))
    if rwkv7.suffix.lower() != ".pth":
        _bad_input(ValueError("--rwkv7 must be a .pth file"))
    if precision not in ("bf16", "fp16", "fp32"):
        _bad_input(ValueError("--precision supports only bf16 / fp16 / fp32"))
    if reference is not None and not reference.is_file():
        _bad_input(ValueError(f"Reference does not exist: {reference}"))
    if tokenizer_src is not None and not tokenizer_src.is_dir():
        _bad_input(ValueError(f"Tokenizer directory does not exist: {tokenizer_src}"))
    if out.exists() and not out.is_dir():
        _bad_input(ValueError(f"Output path exists and is not a directory: {out}"))
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        _bad_input(ValueError(f"Output directory is not empty: {out}; pass --overwrite to replace it"))

    import time
    from .model_converter import convert
    from .tool_events import (
        ToolEventEmitter, cancelled as tool_cancelled, completed as tool_completed,
        failed as tool_failed, progress as tool_progress, started as tool_started,
    )

    tool = "model_conversion"
    emitter = ToolEventEmitter()
    emitter.emit(tool_started(tool, f"Converting {rwkv7.name}"))
    began = time.monotonic()

    def _progress(phase: str, message: str, current: Optional[int], total: Optional[int]) -> None:
        fraction = 0.02 if phase == "read" else 0.9 if phase == "write" else None
        if phase == "convert" and current is not None and total:
            fraction = 0.05 + 0.8 * (current / total)
        emitter.emit(tool_progress(
            tool, phase, message, current=current, total=total, fraction=fraction,
        ))

    try:
        result = convert(
            rwkv7, out, ref_path=reference, tokenizer_src=tokenizer_src,
            precision=precision, overwrite=overwrite, progress_callback=_progress,
            log=lambda message: typer.echo(f"# {message}", err=True),
        )
    except KeyboardInterrupt:
        emitter.emit(tool_cancelled(tool))
        raise typer.Exit(130)
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
        emitter.emit(tool_failed(tool, str(exc)))
        raise typer.Exit(1) from None

    result["elapsed"] = round(time.monotonic() - began, 3)
    emitter.emit(tool_completed(tool, result, path=str(out)))


@app.command("quantize")
def quantize_cmd(
    model: Path = typer.Option(..., "--model", "-m", help="Source BF16 HF model directory"),
    out: Path = typer.Option(..., "--out", "-o", help="Output INT8 quantized model directory"),
    bits: int = typer.Option(8, "--bits", help="Quantization bits (default 8 = INT8)"),
    group_size: int = typer.Option(64, "--group-size", help="Quantization group size (default 64)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow writing to a non-empty output directory"),
):
    """Quantize a BF16 HF model directory to the standard MLX INT8 format.

    Inference commands load the resulting directory without extra flags. Quantized models
    are inference-only; state training requires BF16 weights. stdout contains tool-task
    JSON Lines and human-readable logs go to stderr.
    """
    if not model.is_dir():
        _bad_input(ValueError(f"Source model directory does not exist: {model}"))
    if not (model / "config.json").is_file():
        _bad_input(ValueError(f"Source directory is missing config.json: {model}"))
    if out.exists() and not out.is_dir():
        _bad_input(ValueError(f"Output path exists and is not a directory: {out}"))
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        _bad_input(ValueError(f"Output directory is not empty: {out}; pass --overwrite to replace it"))
    if overwrite and out.is_dir():
        import shutil

        shutil.rmtree(out)

    import time
    from .quantizer import quantize
    from .tool_events import (
        ToolEventEmitter, cancelled as tool_cancelled, completed as tool_completed,
        failed as tool_failed, progress as tool_progress, started as tool_started,
    )

    tool = "quantization"
    emitter = ToolEventEmitter()
    emitter.emit(tool_started(tool, f"Quantizing {model.name} (bits={bits})"))
    began = time.monotonic()

    def _progress(phase: str, message: str, current: Optional[int], total: Optional[int]) -> None:
        fraction = {"load": 0.15, "quantize": 0.6, "save": 0.9}.get(phase, None)
        emitter.emit(tool_progress(
            tool, phase, message, current=current, total=total, fraction=fraction,
        ))

    try:
        result = quantize(
            model, out, bits=bits, group_size=group_size,
            progress_callback=_progress,
            log=lambda message: typer.echo(f"# {message}", err=True),
        )
    except KeyboardInterrupt:
        emitter.emit(tool_cancelled(tool))
        raise typer.Exit(130)
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
        emitter.emit(tool_failed(tool, str(exc)))
        raise typer.Exit(1) from None

    result["elapsed"] = round(time.monotonic() - began, 3)
    emitter.emit(tool_completed(tool, result, path=str(out)))


@app.command("dataset-preview")
def dataset_preview(
    model: Path = typer.Option(..., "--model", "-m", help="HF model directory (tokenizer source)"),
    data: Path = typer.Option(..., "--data", "-d", help="Source data in JSON/JSONL/CSV format"),
    ctx_len: int = typer.Option(512, "--ctx-len"),
    turn_policy: str = typer.Option("first", "--turn-policy", help="first | all"),
    prompt_key: Optional[str] = typer.Option(None, "--prompt-key"),
    response_key: Optional[str] = typer.Option(None, "--response-key"),
    cache_out: Optional[Path] = typer.Option(
        None, "--cache-out", help="Optional JSONL page cache for the complete rendered preview",
    ),
    page_size: int = typer.Option(20, "--page-size", min=1, max=200),
):
    """Detect a dataset format, inspect all records, and return the first preview page.

    --cache-out streams the complete preview to disk for dataset-preview-page. This loads
    only the tokenizer, not model weights; stdout contains tool-task JSON Lines.
    """
    if not model.is_dir():
        _bad_input(ValueError(f"Model directory does not exist: {model}"))
    if not data.is_file():
        _bad_input(ValueError(f"Data file does not exist: {data}"))
    if ctx_len <= 0:
        _bad_input(ValueError("--ctx-len must be > 0"))
    if turn_policy not in ("first", "all"):
        _bad_input(ValueError("--turn-policy supports only first / all"))
    if (prompt_key is None) != (response_key is None):
        _bad_input(ValueError("--prompt-key and --response-key must be provided together"))
    if not 1 <= page_size <= 200:
        _bad_input(ValueError("--page-size must be between 1 and 200"))

    from .importer import (
        convert, detect_schema, detection_for_fields, read_records,
    )
    from .inspection import inspect_standard_records
    from .preview_cache import PreviewCacheWriter
    from .tool_events import (
        ToolEventEmitter, cancelled as tool_cancelled, completed as tool_completed,
        failed as tool_failed, progress as tool_progress, started as tool_started,
        warning as tool_warning,
    )
    from mlx_lm.utils import load_tokenizer

    tool = "dataset_preview"
    emitter = ToolEventEmitter()
    emitter.emit(tool_started(tool, f"Checking {data.name}"))
    try:
        emitter.emit(tool_progress(tool, "tokenizer", "Loading tokenizer"))
        tokenizer = load_tokenizer(
            str(model), tokenizer_config_extra={"trust_remote_code": True},
        )
        items = read_records(data)
        detection = detect_schema(items)
        if prompt_key is not None:
            detection = detection_for_fields(items, prompt_key, response_key or "")

        result_payload = None
        rendered_payload: list[dict] = []
        inspection_payload = None
        pagination_payload = None
        if detection.schema != "unknown":
            converted = convert(items, detection, turn_policy=turn_policy)  # type: ignore[arg-type]
            result_payload = converted.to_dict()
            emitter.emit(tool_progress(
                tool, "inspect", f"Preparing to check {len(converted.records)} samples",
            ))

            def report_inspection_progress(current: int, total: int) -> None:
                if current == total or current % 100 == 0:
                    emitter.emit(tool_progress(
                        tool,
                        "render",
                        f"Checking and rendering {current:,} / {total:,} samples",
                        current=current,
                        total=total,
                    ))

            if cache_out is not None:
                with PreviewCacheWriter(cache_out, page_size=page_size) as cache:
                    inspected = inspect_standard_records(
                        converted.records, tokenizer, template=converted.template,
                        ctx_len=ctx_len, path=str(data), on_rendered=cache.append,
                        on_progress=report_inspection_progress,
                    )
                    meta = cache.commit(template=converted.template, ctx_len=ctx_len)
                    rendered_payload = cache.first_page
                pagination_payload = {
                    "cache_path": str(cache_out),
                    "total": meta["total"],
                    "page_size": meta["page_size"],
                    "page_count": meta["page_count"],
                }
            else:
                def collect_first_page(sample: dict) -> None:
                    if len(rendered_payload) < page_size:
                        rendered_payload.append(sample)

                inspected = inspect_standard_records(
                    converted.records, tokenizer, template=converted.template,
                    ctx_len=ctx_len, path=str(data), on_rendered=collect_first_page,
                    on_progress=report_inspection_progress,
                )
            inspection_payload = inspected.to_dict()
            if converted.dropped_system:
                emitter.emit(tool_warning(
                    tool, f"Ignored {converted.dropped_system} system messages",
                ))
            if converted.dropped_other:
                emitter.emit(tool_warning(
                    tool, f"Ignored {converted.dropped_other} unpaired records",
                ))
            if inspected.truncated:
                emitter.emit(tool_warning(
                    tool, f"ctx_len={ctx_len} will truncate {inspected.truncated} samples",
                ))

        available_keys = sorted({
            str(key) for item in detection.sample if isinstance(item, dict) for key in item.keys()
        })
        payload = {
            "detection": detection.to_dict(),
            "result": result_payload,
            "preview": rendered_payload,
            "inspection": inspection_payload,
            "pagination": pagination_payload,
            "available_keys": available_keys,
            "turn_policy": turn_policy,
        }
        emitter.emit(tool_completed(tool, payload, path=str(data)))
    except KeyboardInterrupt:
        emitter.emit(tool_cancelled(tool))
        raise typer.Exit(130)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        emitter.emit(tool_failed(tool, str(exc)))
        raise typer.Exit(1) from None


@app.command("dataset-preview-page")
def dataset_preview_page(
    cache: Path = typer.Option(..., "--cache", help="Cache generated by dataset-preview"),
    page: int = typer.Option(..., "--page", min=1),
    page_size: Optional[int] = typer.Option(None, "--page-size", min=1, max=200),
):
    """Read one page from a preview cache without loading a dataset, tokenizer, or model."""
    from .preview_cache import read_preview_cache_page
    from .tool_events import (
        ToolEventEmitter, completed as tool_completed, failed as tool_failed,
        started as tool_started,
    )

    tool = "dataset_preview_page"
    emitter = ToolEventEmitter()
    emitter.emit(tool_started(tool, f"Loading page {page}"))
    try:
        payload = read_preview_cache_page(cache, page=page, page_size=page_size)
        emitter.emit(tool_completed(tool, payload, path=str(cache)))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        emitter.emit(tool_failed(tool, str(exc)))
        raise typer.Exit(1) from None


@app.command("import")
def import_data(
    data: Path = typer.Option(..., "--data", "-d", help="Source data file (JSONL/JSON/CSV)"),
    out: Path = typer.Option(..., "--out", "-o", help="Output standard JSONL path"),
    turn_policy: str = typer.Option(
        "first", "--turn-policy",
        help="Multi-turn split policy (ShareGPT/Messages): first | all (each pair becomes a sample)",
    ),
    prompt_key: Optional[str] = typer.Option(
        None, "--prompt-key", help="Manual prompt field when automatic detection fails",
    ),
    response_key: Optional[str] = typer.Option(
        None, "--response-key", help="Manual response field when automatic detection fails",
    ),
    json_output: bool = typer.Option(False, "--json", help="Write structured JSON to stdout"),
    events: bool = typer.Option(False, "--events", help="Write tool-task JSON Lines to stdout"),
):
    """Detect and convert an external dataset to Preen's standard JSONL format.

    Automatically detects Alpaca, ShareGPT, Messages/ChatML, and plain QA. The output is a
    standard JSONL file plus a traceable <name>.import.json sidecar. Parquet is unsupported.
    """
    if not data.is_file():
        _bad_input(ValueError(f"Data file does not exist: {data}"))
    if turn_policy not in ("first", "all"):
        _bad_input(ValueError("--turn-policy supports only first / all"))
    if (prompt_key is None) != (response_key is None):
        _bad_input(ValueError("--prompt-key and --response-key must be provided together"))
    if json_output and events:
        _bad_input(ValueError("--json and --events cannot be used together"))

    from .importer import import_dataset
    emitter = None
    if events:
        from .tool_events import ToolEventEmitter, started as tool_started
        emitter = ToolEventEmitter()
        emitter.emit(tool_started("dataset_import", f"Converting {data.name}"))

    try:
        artifact, result = import_dataset(
            data, out, turn_policy=turn_policy,  # type: ignore[arg-type]
            prompt_key=prompt_key, response_key=response_key,
        )
    except KeyboardInterrupt:
        if emitter is not None:
            from .tool_events import cancelled as tool_cancelled
            emitter.emit(tool_cancelled("dataset_import"))
        raise typer.Exit(130)
    except (OSError, ValueError, TypeError) as exc:
        if emitter is not None:
            from .tool_events import failed as tool_failed
            emitter.emit(tool_failed("dataset_import", str(exc)))
            raise typer.Exit(1) from None
        _bad_input(exc)

    payload = {
        "jsonl_path": str(artifact.jsonl_path),
        "sidecar_path": str(artifact.sidecar_path),
        "sha256": artifact.sha256,
        "record_count": artifact.record_count,
        "result": result.to_dict(),
    }

    if emitter is not None:
        from .tool_events import completed as tool_completed
        emitter.emit(tool_completed("dataset_import", payload, path=str(artifact.jsonl_path)))
        return

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
        return

    typer.echo(f"# Detection: {result.detection.schema} (confidence={result.detection.confidence:.2f})", err=True)
    typer.echo(f"# Template: {result.template}, policy: {turn_policy}", err=True)
    typer.echo(f"# Output: {len(result.records)} records -> {artifact.jsonl_path}", err=True)
    typer.echo(f"# sidecar: {artifact.sidecar_path}", err=True)
    typer.echo(f"# Source SHA-256: {artifact.sha256}", err=True)
    if result.dropped_system:
        typer.echo(f"# Dropped system messages: {result.dropped_system}", err=True)
    if result.dropped_other:
        typer.echo(f"# Dropped unclassified records: {result.dropped_other}", err=True)
    if result.qa_degradation_hint:
        typer.echo(
            "# Tip: all input fields are empty; you may use the qa template instead (instruction -> prompt)",
            err=True,
        )
    typer.echo(f"✓ {artifact.jsonl_path} ({artifact.record_count} records)")


@app.command()
def serve(
    model: Path = typer.Option(..., "--model", "-m", help="Resident HF model directory"),
    cache_limit_gb: Optional[str] = typer.Option(
        "auto", "--cache-limit-gb",
        help="MLX buffer-cache limit; auto=25% of physical memory (default), or a GB value.",
    ),
):
    """Run the persistent inference JSON Lines protocol over stdin and stdout.

    One process keeps one model loaded. Switching models requires restarting serve. A ready
    event is emitted after loading, then requests are handled one line at a time.
    """
    if not model.is_dir():
        _bad_input(ValueError(f"Model directory does not exist: {model}"))

    from .serve import run_serve

    typer.echo(f"# Starting serve: {model} (stdin/stdout JSON Lines protocol)", err=True)
    run_serve(str(model), cache_limit_spec=cache_limit_gb)


if __name__ == "__main__":
    app()
