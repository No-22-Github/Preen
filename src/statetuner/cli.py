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

import typer

app = typer.Typer(
    name="statetuner",
    help="RWKV-7 state tuning for Mac — train, export, and preview init states.",
    no_args_is_help=True,
    add_completion=False,
)


def _bad_input(exc: Exception) -> None:
    """把可预期的输入错误转成简洁 CLI 错误。"""
    typer.echo(f"错误: {exc}", err=True)
    raise typer.Exit(2)


def _apply_cache_limit(spec: Optional[str]) -> None:
    """load_model 前设 MLX buffer cache 上限(GB 口径)。

    必须在任何 MLX 加载/分配前调用才有效(mem_probe 验证过的时序)。
    spec 解析:
      None       — 不动 MLX 默认(理论上不会被触发,默认是 auto)
      "auto"     — 物理内存 × 25%(16GB 机器 ≈ 4.3G,c4G 同档)
      "<number>" — 直接当 GB,如 "4" → 4G
    全仓 GB 口径(÷1e9),禁止 /1024³。
    """
    if spec is None:
        return
    import mlx.core as mx

    if spec == "auto":
        gb = mx.device_info()["memory_size"] / 1e9 * 0.25
    else:
        try:
            gb = float(spec)
        except ValueError:
            _bad_input(ValueError(
                f"--cache-limit-gb 只接受 'auto' 或正数, 收到 {spec!r}"
            ))
    if gb <= 0:
        _bad_input(ValueError("--cache-limit-gb 必须 > 0"))
    mx.set_cache_limit(int(gb * 1e9))


@app.command("doctor")
def doctor(
    json_output: bool = typer.Option(False, "--json", help="stdout 输出结构化 JSON"),
):
    """检查 Python、MLX、Metal 和导出依赖。"""
    from .inspection import doctor_report

    report = doctor_report()
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False))
        return
    typer.echo(f"Python: {report['python']}")
    typer.echo(f"平台: {report['platform']} ({report['machine']})")
    typer.echo(f"Apple Silicon: {'✓' if report['apple_silicon'] else '✗'}")
    for name in ("mlx", "mlx_lm", "ml_dtypes", "numpy"):
        info = report[name]
        typer.echo(f"{name}: {'✓ ' + info.get('version', '') if info['ok'] else '✗ ' + info['error']}")
    typer.echo(f"Metal: {'✓' if report.get('metal_available') else '✗'}")
    if report.get("metal_available"):
        typer.echo(
            f"内存: physical={report.get('memory_size_gb', 0):.2f}G "
            f"working_set={report.get('working_set_gb', 0):.2f}G"
        )


@app.command("data-info")
def data_info(
    model: Path = typer.Option(..., "--model", "-m", help="HF 模型目录(tokenizer 来源)"),
    data: Path = typer.Option(..., "--data", "-d", help="NekoQA JSON/JSONL"),
    ctx_len: int = typer.Option(512, "--ctx-len"),
    json_output: bool = typer.Option(False, "--json", help="stdout 输出结构化 JSON"),
):
    """用真实 tokenizer 检查数据字段、长度与截断情况。"""
    if not model.is_dir():
        _bad_input(ValueError(f"模型目录不存在: {model}"))
    if not data.is_file():
        _bad_input(ValueError(f"数据文件不存在: {data}"))
    if ctx_len <= 0:
        _bad_input(ValueError("--ctx-len 必须 > 0"))

    from .inspection import inspect_data
    from mlx_lm.utils import load_tokenizer

    typer.echo(f"# 加载 tokenizer: {model}", err=True)
    try:
        tok = load_tokenizer(
            str(model), tokenizer_config_extra={"trust_remote_code": True}
        )
        result = inspect_data(data, tok, ctx_len=ctx_len)
    except (OSError, ValueError, TypeError) as exc:
        _bad_input(exc)
    payload = result.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
        return
    typer.echo(f"数据: {result.path}")
    typer.echo(
        f"样本: total={result.total} valid={result.valid} "
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
    json_output: bool = typer.Option(False, "--json", help="stdout 输出结构化 JSON"),
):
    """检查 state 格式、层号、shape、dtype 与 std。"""
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
    model: Path = typer.Option(..., "--model", "-m", help="转换后的 HF 模型目录"),
    data: Path = typer.Option(..., "--data", "-d", help="训练数据 JSON/JSONL"),
    test_data: Optional[Path] = typer.Option(
        None, "--test-data", help="held-out 数据(早停用);缺省则从 train 划分"
    ),
    out: Path = typer.Option(
        Path("state.npz"), "--out", "-o", help="输出训练 state(.npz)"
    ),
    lr: float = typer.Option(0.01, "--lr", help="学习率(默认 0.01,P0 实测;1.0 会爆炸)"),
    lr_floor: float = typer.Option(1e-4, "--lr-floor", help="cosine 衰减终点"),
    warmup: int = typer.Option(10, "--warmup", help="warmup 步数"),
    ctx_len: int = typer.Option(512, "--ctx-len", help="上下文长度"),
    epochs: int = typer.Option(20, "--epochs", help="epoch 数(配早停后是上限)"),
    grad_clip: float = typer.Option(1.0, "--grad-clip"),
    early_stop: bool = typer.Option(True, "--early-stop/--no-early-stop", help="held-out 早停"),
    patience: int = typer.Option(3, "--patience", help="早停耐心(连续 N 次不改善则停)"),
    test_ratio: float = typer.Option(0.1, "--test-ratio", help="无 --test-data 时从 train 划分比例"),
    checkpoint_dir: Optional[Path] = typer.Option(None, "--checkpoint-dir", help="checkpoint 目录"),
    checkpoint_every: int = typer.Option(2, "--checkpoint-every", help="每 N epoch 存一次"),
    resume: Optional[Path] = typer.Option(None, "--resume", help="从 checkpoint 恢复"),
    events_file: Optional[Path] = typer.Option(
        None, "--events-file", help="训练事件 JSON lines 写入文件(stdout 同时输出)"
    ),
    export_pth: bool = typer.Option(False, "--export-pth", help="训完顺手导出 .pth"),
    pth_out: Optional[Path] = typer.Option(None, "--pth-out", help="导出 pth 路径(默认 out 同名 .pth)"),
    template: str = typer.Option(
        "nekoqa", "--template",
        help="任务模板: nekoqa(角色扮演 QA,默认)",
    ),
    seed: int = typer.Option(42, "--seed"),
    cache_limit_gb: Optional[str] = typer.Option(
        "auto", "--cache-limit-gb",
        help="MLX buffer cache 上限;auto=物理内存×25%(默认,16G机≈4.3G),或直接给 GB 数。设小降 RSS,必须在模型加载前生效。",
    ),
):
    """训练 state tuning。事件流输出到 stdout(JSON lines)。"""
    if not model.is_dir():
        _bad_input(ValueError(f"模型目录不存在: {model}"))
    if not data.is_file():
        _bad_input(ValueError(f"训练数据不存在: {data}"))
    if test_data is not None and not test_data.is_file():
        _bad_input(ValueError(f"held-out 数据不存在: {test_data}"))
    if lr <= 0:
        _bad_input(ValueError("--lr 必须 > 0"))
    if lr_floor <= 0 or lr_floor > lr:
        _bad_input(ValueError("--lr-floor 必须 > 0 且 <= --lr"))
    if warmup < 0 or ctx_len <= 0 or epochs <= 0 or grad_clip <= 0:
        _bad_input(ValueError("warmup/ctx-len/epochs/grad-clip 参数范围非法"))
    if not 0 < test_ratio < 1:
        _bad_input(ValueError("--test-ratio 必须在 (0, 1) 范围内"))
    if patience <= 0 or checkpoint_every <= 0:
        _bad_input(ValueError("--patience 和 --checkpoint-every 必须 > 0"))
    if pth_out is not None and not export_pth:
        _bad_input(ValueError("--pth-out 必须配合 --export-pth"))

    # cache_limit 必须在 load_model 前生效(mem_probe 验证过的时序)。
    _apply_cache_limit(cache_limit_gb)

    import mlx.core as mx

    from . import events
    from .service import TrainingRequest, run_training
    from .train import TrainConfig

    # TODO(产品决策): 是否开放 g1g 训练模板未定。g1g 是 reasoning 模板，
    # state tuning 是否能稳定注入风格/格式而非破坏推理链路，尚未验证。
    # 当前只支持 nekoqa；若开放需在 templates/service/data 全链路同步。
    if template != "nekoqa":
        raise typer.BadParameter(
            f"--template 当前只支持 nekoqa, 收到 {template!r}"
        )

    cfg = TrainConfig(
        lr=lr, lr_floor=lr_floor, warmup=warmup, ctx_len=ctx_len, epochs=epochs,
        # std 健康区间未标定：只记录，不使用旧 1.0 阈值报警。
        grad_clip=grad_clip, max_state_std=None,
        early_stop=early_stop, early_stop_patience=patience,
        checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every,
        resume=resume, seed=seed,
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
            typer.echo(f"错误: {exc}", err=True)
            raise typer.Exit(1) from None

    peak = float(mx.get_peak_memory()) / 1e9
    typer.echo(
        f"# 完成: epochs={result.epochs_run} loss={result.final_loss:.4f} "
        f"std={result.final_state_std:.4f} peak_mem={peak:.2f}GB "
        f"elapsed={result.elapsed:.1f}s",
        err=True,
    )


@app.command()
def eval(
    model: Path = typer.Option(..., "--model", "-m", help="HF 模型目录"),
    state: Path = typer.Option(..., "--state", "-s", help="state 文件(npz 或 pth)"),
    data: Optional[Path] = typer.Option(
        None, "--data", "-d", help="评估数据(jsonl 或 json 数组);缺省用内置示例"
    ),
    max_tokens: int = typer.Option(70, "--max-tokens"),
    temperature: float = typer.Option(0.8, "--temperature", help="采样温度;0=贪心"),
    top_p: float = typer.Option(0.9, "--top-p", help="nucleus sampling 阈值"),
    seed: int = typer.Option(42, "--seed", help="采样随机种子"),
    template: str = typer.Option(
        "nekoqa", "--template",
        help="任务模板: nekoqa(角色扮演 QA,默认) | g1g(RWKV7-G1 原生)",
    ),
    limit: int = typer.Option(5, "--limit", help="最多输出条数(默认 5)"),
    json_output: bool = typer.Option(False, "--json", help="stdout 输出结构化 JSON"),
    cache_limit_gb: Optional[str] = typer.Option(
        "auto", "--cache-limit-gb",
        help="MLX buffer cache 上限;auto=物理内存×25%(默认,16G机≈4.3G),或直接给 GB 数。设小降 RSS,必须在模型加载前生效。",
    ),
):
    """评估:对数据集逐条生成(state 注入),输出结果。

    prompt 从 templates 派生(与训练 encode 路径同源,保证编码同构);
    --template 同时驱动 prompt 渲染与 stop sequences(同源,不分裂)。
    """
    if template not in ("nekoqa", "g1g"):
        raise typer.BadParameter(
            f"--template 当前只支持 nekoqa / g1g, 收到 {template!r}"
        )
    if not model.is_dir():
        _bad_input(ValueError(f"模型目录不存在: {model}"))
    if not state.is_file():
        _bad_input(ValueError(f"state 文件不存在: {state}"))
    if data is not None and not data.is_file():
        _bad_input(ValueError(f"评估数据不存在: {data}"))
    if max_tokens <= 0 or temperature < 0 or not 0 < top_p <= 1:
        _bad_input(ValueError("max-tokens/temperature/top-p 参数范围非法"))
    if limit <= 0:
        raise typer.BadParameter("--limit 必须 > 0")

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

    typer.echo(f"# 加载模型 {model} (kernel 路径, template={template})", err=True)
    mdl, tok = load_model(model, patch=False)
    engine = InferenceEngine(mdl, tok)

    request = EvaluationRequest(
        engine=engine,
        state=str(state),
        template=template,
        config=cfg,
        data=data,
        limit=limit,
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
    state: Path = typer.Option(..., "--state", "-s", help="state npz(P0 内部格式)"),
    out: Path = typer.Option(..., "--out", "-o", help="输出 .pth 路径"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="导出后 round-trip 验证"),
):
    """npz → RWKV Runner 可挂载的 .pth。"""
    if not state.is_file():
        _bad_input(ValueError(f"state 文件不存在: {state}"))
    if state.suffix.lower() != ".npz":
        _bad_input(ValueError("export 的 --state 必须是 .npz"))
    out.parent.mkdir(parents=True, exist_ok=True)

    from .export import export_pth as _export, load_npz_as_numpy, verify_roundtrip

    typer.echo(f"# 读取 {state}", err=True)
    states = load_npz_as_numpy(state)
    typer.echo(f"# {len(states)} 层, 导出 → {out}", err=True)
    _export(states, out)

    if verify:
        ok, msg = verify_roundtrip(states, out)
        typer.echo(f"# round-trip: {'✓' if ok else '✗'} {msg}", err=True)
        if not ok:
            raise typer.Exit(1)
    typer.echo(f"✓ {out}")


@app.command()
def chat(
    model: Path = typer.Option(..., "--model", "-m", help="HF 模型目录"),
    state: Optional[Path] = typer.Option(None, "--state", "-s", help="初始 state(npz/pth)"),
    max_tokens: int = typer.Option(200, "--max-tokens", help="单轮最大生成 token"),
    temperature: float = typer.Option(0.8, "--temperature", help="采样温度;0=贪心"),
    top_p: float = typer.Option(0.9, "--top-p"),
    seed: int = typer.Option(42, "--seed"),
    ab: bool = typer.Option(False, "--ab", help="启动时开启 A/B"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="逐步输出生成文本"),
    template: str = typer.Option("g1g", "--template", help="g1g(RWKV7-G1 原生,默认) | raw | nekoqa"),
    cache_limit_gb: Optional[str] = typer.Option(
        "auto", "--cache-limit-gb",
        help="MLX buffer cache 上限;auto=物理内存×25%(默认,16G机≈4.3G),或直接给 GB 数。设小降 RSS,必须在模型加载前生效。",
    ),
):
    """模型常驻的交互模式；支持运行中动态切换 state。"""
    if not model.is_dir():
        _bad_input(ValueError(f"模型目录不存在: {model}"))
    if state is not None and not state.is_file():
        _bad_input(ValueError(f"state 文件不存在: {state}"))
    if template not in ("raw", "nekoqa", "g1g"):
        _bad_input(ValueError("--template 只支持 raw / nekoqa / g1g"))
    if max_tokens <= 0 or temperature < 0 or not 0 < top_p <= 1:
        _bad_input(ValueError("max-tokens/temperature/top-p 参数范围非法"))
    if ab and state is None:
        _bad_input(ValueError("--ab 必须同时提供 --state"))

    # cache_limit 必须在 load_model 前生效(mem_probe 验证过的时序)。
    _apply_cache_limit(cache_limit_gb)

    from .chat import ChatSession
    from .core import load_model
    from .inference import GenerationConfig, InferenceEngine
    from .inspection import validate_state_for_model

    typer.echo(f"# 加载模型 {model}（模型将在会话期间常驻）", err=True)
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

    session = ChatSession(
        engine,
        config=GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        ),
        template=template,
        state=loaded_state,
        state_label=str(state) if state else None,
        state_loader=_load_checked,
        ab=ab,
    )

    typer.echo("交互模式已启动。每轮从当前 S₀ 重新开始；输入 /help 查看命令。")
    typer.echo(session.config_line())
    while True:
        try:
            line = input("You> ")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\n会话结束。")
            break
        use_stream = stream and not session.ab and not line.lstrip().startswith("/")
        if use_stream:
            typer.echo("Neko> ", nl=False)

            def _on_text(chunk: str) -> None:
                typer.echo(chunk, nl=False)

            reply = session.handle(line, on_text=_on_text)
            typer.echo()
        else:
            reply = session.handle(line)
        for output_line in reply.lines:
            typer.echo(output_line)
        if reply.exit:
            break


@app.command()
def preview(
    model: Path = typer.Option(..., "--model", "-m", help="HF 模型目录"),
    state: Optional[Path] = typer.Option(
        None, "--state", "-s", help="state 文件(npz/pth);缺省=无 state 基线"
    ),
    prompt: str = typer.Option(..., "--prompt", "-p", help="输入文本(问题/中文)"),
    max_tokens: int = typer.Option(80, "--max-tokens"),
    temperature: float = typer.Option(0.0, "--temperature", help="采样温度;0=贪心"),
    top_p: float = typer.Option(0.9, "--top-p", help="nucleus sampling 阈值"),
    seed: int = typer.Option(42, "--seed", help="采样随机种子"),
    ab: bool = typer.Option(False, "--ab", help="A/B 对比:有 state vs 无 state 双输出"),
    template: str = typer.Option(
        "raw", "--template",
        help="包装 prompt 的模板: raw(原样,默认) | nekoqa | g1g(RWKV7-G1 原生)",
    ),
    json_output: bool = typer.Option(False, "--json", help="stdout 输出结构化 JSON"),
    stream: bool = typer.Option(False, "--stream", help="逐步输出单路生成文本"),
):
    """预览:注入 state 生成。--ab 做 A/B 对比，temperature=0 时为贪心。

    --template 决定 prompt 包装与 stop sequences(同源,不分裂):
      raw 原样传入;nekoqa 包成 "User: {prompt}\\n\\nAssistant:";
      g1g 包成带 bos + 空 think 标签的 RWKV7-G1 格式。
    """
    if template not in ("raw", "nekoqa", "g1g"):
        raise typer.BadParameter(
            f"--template 只支持 raw / nekoqa / g1g, 收到 {template!r}"
        )
    if ab and state is None:
        raise typer.BadParameter("--ab 必须同时提供 --state")
    if stream and (ab or json_output):
        raise typer.BadParameter("--stream 不能与 --ab / --json 同时使用")
    if not model.is_dir():
        _bad_input(ValueError(f"模型目录不存在: {model}"))
    if state is not None and not state.is_file():
        _bad_input(ValueError(f"state 文件不存在: {state}"))
    if not prompt:
        _bad_input(ValueError("--prompt 不能为空"))
    if max_tokens <= 0 or temperature < 0 or not 0 < top_p <= 1:
        _bad_input(ValueError("max-tokens/temperature/top-p 参数范围非法"))

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
    wrapped = render_prompt(prompt, template)

    typer.echo(f"# 加载模型 {model} (template={template})", err=True)
    mdl, tok = load_model(model, patch=False)
    engine = InferenceEngine(mdl, tok)

    if ab:
        result = engine.compare(wrapped, state=str(state), config=cfg)
        if json_output:
            typer.echo(json.dumps(result.to_dict(), ensure_ascii=False))
            return
        typer.echo("=== 有 state ===")
        typer.echo(result.with_state.text)
        typer.echo(
            f"[stop={result.with_state.stop_reason}, tokens={result.with_state.token_count}]"
        )
        typer.echo("=== 无 state(基线)===")
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


if __name__ == "__main__":
    app()
