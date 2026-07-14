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
    typer.echo(f"错误: {exc}", err=True)
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
    template: str = typer.Option(
        "auto", "--template", help="auto(按 sidecar) | qa | instruction",
    ),
    json_output: bool = typer.Option(False, "--json", help="stdout 输出结构化 JSON"),
):
    """用真实 tokenizer 检查数据字段、长度与截断情况。"""
    if not model.is_dir():
        _bad_input(ValueError(f"模型目录不存在: {model}"))
    if not data.is_file():
        _bad_input(ValueError(f"数据文件不存在: {data}"))
    if ctx_len <= 0:
        _bad_input(ValueError("--ctx-len 必须 > 0"))
    if template not in ("auto", "qa", "instruction"):
        _bad_input(ValueError("--template 只支持 auto / qa / instruction"))

    from .inspection import inspect_data, inspect_standard_jsonl
    from .service import _has_import_sidecar
    from mlx_lm.utils import load_tokenizer

    typer.echo(f"# 加载 tokenizer: {model}", err=True)
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
                    "无 .import.json sidecar 的遗留数据只能按 qa 模板检查"
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
    log_every: int = typer.Option(
        1, "--log-every",
        help="每 N 步发一个 step 事件(默认 1 = 每步都发;事件开销微秒级,不影响训练速度)",
    ),
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
        "qa", "--template",
        help="任务模板: qa(角色扮演 QA,默认) | instruction(指令问答)",
    ),
    seed: int = typer.Option(42, "--seed"),
    cache_limit_gb: Optional[str] = typer.Option(
        "auto", "--cache-limit-gb",
        help="MLX buffer cache 上限;auto=物理内存×25%(默认,16G机≈4.3G),或直接给 GB 数。设小降 RSS,必须在模型加载前生效。",
    ),
):
    """训练 state tuning。事件流输出到 stdout(JSON lines)。

    训练只接受 qa / instruction 模板;reasoning 方言与 think 档位是推理侧概念,
    在 train 命令中不存在(不是忽略,是没有这些参数)。训练侧 target 永远不含
    think 内容(Spec §1.2)。
    """
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
    if log_every < 1:
        _bad_input(ValueError("--log-every 必须 >= 1"))
    if pth_out is not None and not export_pth:
        _bad_input(ValueError("--pth-out 必须配合 --export-pth"))
    if template not in ("qa", "instruction"):
        _bad_input(ValueError(
            f"--template 只支持 qa / instruction, 收到 {template!r}"
            " (reasoning/think 是推理侧参数, 训练不接受)"
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

    # 内存口径(runtime.py):用 phys_footprint 峰值(== Activity Monitor「内存」列,
    # 含 Metal 的 IOKit 映射)。旧实现用 ru_maxrss 漏报 ~7x(不含 IOKit);
    # mx.get_peak_memory() 更不准(只算 MLX allocator)。见 runtime.py docstring。
    from .runtime import memory_report

    snap = memory_report()
    peak_gb = snap.peak_footprint_gb or snap.rss_gb or 0.0
    typer.echo(
        f"# 完成: epochs={result.epochs_run} loss={result.final_loss:.4f} "
        f"std={result.final_state_std:.4f} peak_mem={peak_gb:.2f}GB "
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
        "qa", "--template",
        help="任务模板: qa(角色扮演 QA,默认) | instruction(指令) | raw(原样)",
    ),
    reasoning: bool = typer.Option(
        False, "--reasoning",
        help="reasoning 模型(G1 系列)前缀加 bos + 按 --think 追加标签。仅 qa 模板合法。",
    ),
    think: str = typer.Option(
        "off", "--think",
        help="思考档位(仅 --reasoning 时合法): off(直答) | fast(跳过思考) | on(完整思考)",
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
    --reasoning + --think 仅在 qa 模板下生效(reasoning 模型前缀 bos + think 标签)。
    """
    if template not in ("qa", "instruction", "raw"):
        raise typer.BadParameter(
            f"--template 只支持 qa / instruction / raw, 收到 {template!r}"
        )
    if think not in ("off", "fast", "on"):
        raise typer.BadParameter(
            f"--think 只支持 off / fast / on, 收到 {think!r}"
        )
    if think != "off" and not reasoning:
        raise typer.BadParameter("--think 仅在 --reasoning 时合法")
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

    typer.echo(
        f"# 加载模型 {model} (kernel 路径, template={template}"
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
    state: Path = typer.Option(..., "--state", "-s", help="state npz(P0 内部格式)"),
    out: Path = typer.Option(..., "--out", "-o", help="输出 .pth 路径"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="导出后 round-trip 验证"),
    deep: bool = typer.Option(
        False, "--deep",
        help="深度校验:用 pth 挂载后 MLX generate 的输出 == 原始 state 注入的输出。"
             "端到端语义证明(强于 round-trip 的容器证明)。需 --model。",
    ),
    model: Optional[Path] = typer.Option(
        None, "--model", "-m",
        help="深度校验用的模型目录(--deep 时必需,与训练/推理同模型)。",
    ),
    prompt: str = typer.Option(
        "User: 你好\n\nAssistant:", "--prompt",
        help="深度校验用的探测 prompt(默认 QA 前缀)。",
    ),
):
    """npz → RWKV Runner 可挂载的 .pth。

    --deep(T6):接上 export.verify_mount_equivalence —— 端到端证明 pth 挂载后的
    输出 == 训练时 state 注入的输出(语义正确)。区别于 round-trip 只证明
    torch.load 能读(容器正确)。需 --model + tokenizer;会短暂加载模型。
    """
    if not state.is_file():
        _bad_input(ValueError(f"state 文件不存在: {state}"))
    if state.suffix.lower() != ".npz":
        _bad_input(ValueError("export 的 --state 必须是 .npz"))
    if deep and not model:
        _bad_input(ValueError("--deep 必须同时提供 --model(端到端校验需要加载模型)"))
    if deep and model is not None and not model.is_dir():
        _bad_input(ValueError(f"模型目录不存在: {model}"))
    out.parent.mkdir(parents=True, exist_ok=True)

    from .export import export_pth as _export, load_npz_as_numpy, verify_mount_equivalence, verify_roundtrip

    typer.echo(f"# 读取 {state}", err=True)
    states = load_npz_as_numpy(state)
    typer.echo(f"# {len(states)} 层, 导出 → {out}", err=True)
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

        typer.echo(f"# 深度校验: 加载 {model}(端到端 mount 等价)", err=True)
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
    model: Path = typer.Option(..., "--model", "-m", help="HF 模型目录"),
    state: Optional[Path] = typer.Option(None, "--state", "-s", help="初始 state(npz/pth)"),
    max_tokens: int = typer.Option(300, "--max-tokens", help="单轮最大生成 token"),
    temperature: float = typer.Option(1.2, "--temperature", help="采样温度;0=贪心"),
    top_p: float = typer.Option(0.5, "--top-p"),
    seed: int = typer.Option(42, "--seed"),
    presence_penalty: float = typer.Option(
        0.4, "--presence",
        help="重复惩罚:对已出现 token 的固定惩罚(ChatRWKV 官方默认 0.4)。0=关闭。",
    ),
    frequency_penalty: float = typer.Option(
        0.4, "--frequency",
        help="重复惩罚:按出现次数累加(ChatRWKV 官方默认 0.4)。0=关闭。",
    ),
    penalty_decay: float = typer.Option(
        0.996, "--penalty-decay",
        help="重复惩罚:历史计数指数衰减率(ChatRWKV 官方默认 0.996)。",
    ),
    ab: bool = typer.Option(False, "--ab", help="启动时开启 A/B"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="逐步输出生成文本"),
    template: str = typer.Option(
        "qa", "--template",
        help="任务模板: qa(角色扮演 QA,默认) | instruction(指令) | raw(原样)",
    ),
    reasoning: bool = typer.Option(
        False, "--reasoning",
        help="reasoning 模型(G1 系列)前缀加 bos + 按 --think 追加标签。仅 qa 模板合法。"
             "G1 系列 reasoning 模型若不加此项会降智(实测铁证)。",
    ),
    think: str = typer.Option(
        "off", "--think",
        help="思考档位(仅 --reasoning 时生效): off(直答,默认) | fast(跳过思考) | on(完整思考)。"
             "G1 系列推荐 --reasoning --think fast。",
    ),
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
    if template not in ("raw", "qa", "instruction"):
        _bad_input(ValueError("--template 只支持 raw / qa / instruction"))
    if think not in ("off", "fast", "on"):
        _bad_input(ValueError("--think 只支持 off / fast / on"))
    if think != "off" and not reasoning:
        _bad_input(ValueError("--think 仅在 --reasoning 时合法"))
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
        "交互模式已启动。qa 模板多轮走 cache 续传;输入 [bold]/help[/bold] 查看命令。"
    )
    if not reasoning and template == "qa":
        ui_console.print(
            "[dim]# 提示: 若使用 G1 系列 reasoning 模型,加 --reasoning --think fast"
            " 避免降智(详见 docs/g1g-decode-alignment.md)[/dim]",
            style=None,
        )
    # 启动横幅用紧凑两行(· 分隔);/config 命令仍走详尽表格。
    ui_console.print(ui.render_config_compact(session.config_groups(brief=True)))

    while True:
        try:
            ui_console.print(ui.user_prompt_label(), end="")
            line = input()
        except (EOFError, KeyboardInterrupt):
            ui_console.print("\n会话结束。")
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
                        "[dim]⚠ 思考未完成(被 max_tokens 截断,未输出 </think>);"
                        "可用 /max-tokens 调大后重试。[/dim]"
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
        help="包装 prompt 的模板: raw(原样,默认) | qa(角色扮演) | instruction(指令)",
    ),
    reasoning: bool = typer.Option(
        False, "--reasoning",
        help="reasoning 模型(G1 系列)前缀加 bos + 按 --think 追加标签。仅 qa 模板合法。",
    ),
    think: str = typer.Option(
        "off", "--think",
        help="思考档位(仅 --reasoning 时合法): off(直答,默认) | fast(跳过思考) | on(完整思考)",
    ),
    json_output: bool = typer.Option(False, "--json", help="stdout 输出结构化 JSON"),
    stream: bool = typer.Option(False, "--stream", help="逐步输出单路生成文本"),
):
    """预览:注入 state 生成。--ab 做 A/B 对比，temperature=0 时为贪心。

    --template 决定 prompt 包装与 stop sequences(同源,不分裂):
      raw 原样传入;qa 包成 "User: {prompt}\\n\\nAssistant:";
      instruction 包成 "Instruction: ...\\n\\nInput: ...\\n\\nResponse:"(空 input 降级)。
    --reasoning + --think 仅在 qa 模板下生效:前缀加 bos + 按 think 档位追加标签。
    """
    if template not in ("raw", "qa", "instruction"):
        raise typer.BadParameter(
            f"--template 只支持 raw / qa / instruction, 收到 {template!r}"
        )
    if think not in ("off", "fast", "on"):
        raise typer.BadParameter(
            f"--think 只支持 off / fast / on, 收到 {think!r}"
        )
    if think != "off" and not reasoning:
        raise typer.BadParameter("--think 仅在 --reasoning 时合法")
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
    wrapped = render_prompt(prompt, template, reasoning=reasoning, think=think)

    typer.echo(
        f"# 加载模型 {model} (template={template}"
        f"{' reasoning=' + think if reasoning else ''})", err=True
    )
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


@app.command("convert-model")
def convert_model(
    rwkv7: Path = typer.Option(..., "--rwkv7", help="BlinkDL 原生 RWKV-7 .pth"),
    out: Path = typer.Option(..., "--out", "-o", help="输出 HF 模型目录"),
    precision: str = typer.Option("bf16", "--precision", help="bf16(推荐) | fp16 | fp32"),
    reference: Optional[Path] = typer.Option(
        None, "--reference", help="可选:同架构 safetensors 活模型校验模板",
    ),
    tokenizer_src: Optional[Path] = typer.Option(
        None, "--tokenizer-src", help="可选:自定义 World tokenizer 目录",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="允许写入非空输出目录"),
):
    """原生 RWKV-7 .pth → Preen/MLX 可用的 HF safetensors 模型目录。

    stdout 只输出工具任务 JSON Lines；人类日志走 stderr，供 App 与 CLI 共用。
    """
    if not rwkv7.is_file():
        _bad_input(ValueError(f"原生模型文件不存在: {rwkv7}"))
    if rwkv7.suffix.lower() != ".pth":
        _bad_input(ValueError("--rwkv7 必须是 .pth 文件"))
    if precision not in ("bf16", "fp16", "fp32"):
        _bad_input(ValueError("--precision 只支持 bf16 / fp16 / fp32"))
    if reference is not None and not reference.is_file():
        _bad_input(ValueError(f"reference 不存在: {reference}"))
    if tokenizer_src is not None and not tokenizer_src.is_dir():
        _bad_input(ValueError(f"tokenizer 目录不存在: {tokenizer_src}"))
    if out.exists() and not out.is_dir():
        _bad_input(ValueError(f"输出路径已存在且不是目录: {out}"))
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        _bad_input(ValueError(f"输出目录非空: {out};如需覆盖请传 --overwrite"))

    import time
    from .model_converter import convert
    from .tool_events import (
        ToolEventEmitter, cancelled as tool_cancelled, completed as tool_completed,
        failed as tool_failed, progress as tool_progress, started as tool_started,
    )

    tool = "model_conversion"
    emitter = ToolEventEmitter()
    emitter.emit(tool_started(tool, f"开始转换 {rwkv7.name}"))
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
            precision=precision, progress_callback=_progress,
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
    model: Path = typer.Option(..., "--model", "-m", help="HF 模型目录(tokenizer 来源)"),
    data: Path = typer.Option(..., "--data", "-d", help="源数据 json/jsonl/csv"),
    ctx_len: int = typer.Option(512, "--ctx-len"),
    turn_policy: str = typer.Option("first", "--turn-policy", help="first | all"),
    prompt_key: Optional[str] = typer.Option(None, "--prompt-key"),
    response_key: Optional[str] = typer.Option(None, "--response-key"),
    cache_out: Optional[Path] = typer.Option(
        None, "--cache-out", help="可选：写入完整渲染预览 JSONL 分页缓存",
    ),
    page_size: int = typer.Option(20, "--page-size", min=1, max=200),
):
    """探测格式、全量统计，并返回首页训练文本预览。

    传 --cache-out 时把完整预览流式写入磁盘，后续通过 dataset-preview-page
    按页读取。只加载 tokenizer，不加载模型权重；stdout 为工具任务 JSON Lines。
    """
    if not model.is_dir():
        _bad_input(ValueError(f"模型目录不存在: {model}"))
    if not data.is_file():
        _bad_input(ValueError(f"数据文件不存在: {data}"))
    if ctx_len <= 0:
        _bad_input(ValueError("--ctx-len 必须 > 0"))
    if turn_policy not in ("first", "all"):
        _bad_input(ValueError("--turn-policy 只支持 first / all"))
    if (prompt_key is None) != (response_key is None):
        _bad_input(ValueError("--prompt-key 与 --response-key 必须同时提供"))
    if not 1 <= page_size <= 200:
        _bad_input(ValueError("--page-size 必须在 1...200 之间"))

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
    emitter.emit(tool_started(tool, f"检查 {data.name}"))
    try:
        emitter.emit(tool_progress(tool, "tokenizer", "加载 tokenizer"))
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
                tool, "inspect", f"准备检查 {len(converted.records)} 条样本",
            ))

            def report_inspection_progress(current: int, total: int) -> None:
                if current == total or current % 100 == 0:
                    emitter.emit(tool_progress(
                        tool,
                        "render",
                        f"正在检查并渲染 {current:,} / {total:,} 条样本",
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
                    tool, f"已忽略 {converted.dropped_system} 条 system 消息",
                ))
            if converted.dropped_other:
                emitter.emit(tool_warning(
                    tool, f"已忽略 {converted.dropped_other} 条无法配对的记录",
                ))
            if inspected.truncated:
                emitter.emit(tool_warning(
                    tool, f"ctx_len={ctx_len} 将截断 {inspected.truncated} 条样本",
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
    cache: Path = typer.Option(..., "--cache", help="dataset-preview 生成的缓存"),
    page: int = typer.Option(..., "--page", min=1),
    page_size: Optional[int] = typer.Option(None, "--page-size", min=1, max=200),
):
    """从磁盘预览缓存读取一页；不加载数据集、tokenizer 或模型。"""
    from .preview_cache import read_preview_cache_page
    from .tool_events import (
        ToolEventEmitter, completed as tool_completed, failed as tool_failed,
        started as tool_started,
    )

    tool = "dataset_preview_page"
    emitter = ToolEventEmitter()
    emitter.emit(tool_started(tool, f"加载第 {page} 页"))
    try:
        payload = read_preview_cache_page(cache, page=page, page_size=page_size)
        emitter.emit(tool_completed(tool, payload, path=str(cache)))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        emitter.emit(tool_failed(tool, str(exc)))
        raise typer.Exit(1) from None


@app.command("import")
def import_data(
    data: Path = typer.Option(..., "--data", "-d", help="源数据文件(jsonl/json/csv)"),
    out: Path = typer.Option(..., "--out", "-o", help="输出标准 jsonl 路径"),
    turn_policy: str = typer.Option(
        "first", "--turn-policy",
        help="多轮拆分策略(ShareGPT/Messages): first(只取首对) | all(每对独立成样本)",
    ),
    prompt_key: Optional[str] = typer.Option(
        None, "--prompt-key", help="自动探测失败时手动指定 prompt 字段",
    ),
    response_key: Optional[str] = typer.Option(
        None, "--response-key", help="自动探测失败时手动指定 response 字段",
    ),
    json_output: bool = typer.Option(False, "--json", help="stdout 输出结构化 JSON"),
    events: bool = typer.Option(False, "--events", help="stdout 输出工具任务 JSON Lines"),
):
    """导入外部数据集 → 探测格式 → 转内部标准 jsonl(Spec §4)。

    支持 Alpaca / ShareGPT / Messages(ChatML)/ 裸 QA 四类格式自动探测。
    parquet 不支持(pyarrow 伤 bundle),报错给转换提示。
    产物:标准 jsonl + sidecar <name>.import.json(来源 hash + 探测结果,可追溯)。

    UI 与 CLI 共用同一 service 层(statetuner.importer)。
    """
    if not data.is_file():
        _bad_input(ValueError(f"数据文件不存在: {data}"))
    if turn_policy not in ("first", "all"):
        _bad_input(ValueError("--turn-policy 只支持 first / all"))
    if (prompt_key is None) != (response_key is None):
        _bad_input(ValueError("--prompt-key 与 --response-key 必须同时提供"))
    if json_output and events:
        _bad_input(ValueError("--json 与 --events 不能同时使用"))

    from .importer import import_dataset
    emitter = None
    if events:
        from .tool_events import ToolEventEmitter, started as tool_started
        emitter = ToolEventEmitter()
        emitter.emit(tool_started("dataset_import", f"开始转换 {data.name}"))

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

    typer.echo(f"# 探测: {result.detection.schema} (confidence={result.detection.confidence:.2f})", err=True)
    typer.echo(f"# 模板: {result.template}, 策略: {turn_policy}", err=True)
    typer.echo(f"# 产物: {len(result.records)} 条 → {artifact.jsonl_path}", err=True)
    typer.echo(f"# sidecar: {artifact.sidecar_path}", err=True)
    typer.echo(f"# 源文件 sha256: {artifact.sha256}", err=True)
    if result.dropped_system:
        typer.echo(f"# 丢弃 system 消息: {result.dropped_system} 条", err=True)
    if result.dropped_other:
        typer.echo(f"# 丢弃无法归类: {result.dropped_other} 条", err=True)
    if result.qa_degradation_hint:
        typer.echo(
            "# 提示: 所有 input 字段为空,可选降级为 qa 模板(instruction→prompt)",
            err=True,
        )
    typer.echo(f"✓ {artifact.jsonl_path} ({artifact.record_count} 条)")


@app.command()
def serve(
    model: Path = typer.Option(..., "--model", "-m", help="HF 模型目录(常驻)"),
    cache_limit_gb: Optional[str] = typer.Option(
        "auto", "--cache-limit-gb",
        help="MLX buffer cache 上限;auto=物理内存×25%(默认,16G机≈4.3G),或直接给 GB 数。",
    ),
):
    """常驻推理进程:stdin 收 JSON 指令行,stdout 发 JSON 事件行(Spec §3)。

    单进程单模型,启动加载一个模型常驻。SwiftUI/SidecarClient 通过 stdin/stdout
    JSON lines 协议会话。换模型 = 重启 serve 进程。

    协议见 docs/Phase3-总体Spec.md §3(指令集 + 错误语义 + abort 机制)。
    启动后先发 ready 事件(模型加载完毕),随后逐行处理 stdin 指令。
    """
    if not model.is_dir():
        _bad_input(ValueError(f"模型目录不存在: {model}"))

    from .serve import run_serve

    typer.echo(f"# serve 启动: {model}(stdin/stdout JSON lines 协议)", err=True)
    run_serve(str(model), cache_limit_spec=cache_limit_gb)


if __name__ == "__main__":
    app()
