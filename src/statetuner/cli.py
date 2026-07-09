"""statetuner CLI — train / eval / export / preview 四子命令。

这是 P1 的对外入口,也是未来 sidecar IPC 的雏形(每个子命令对应一个 IPC handler)。

事件流:train 把结构化事件以 JSON lines 输出到 stdout(--events-file 可同时写文件),
供人类阅读、管道处理,以及未来的 SwiftUI 进度面板消费。

用法:
  statetuner train --model MODELS --data DATA.jsonl --out state.npz --export-pth
  statetuner eval --model MODELS --state state.pth --data test.jsonl
  statetuner export --state state.npz --out state.pth
  statetuner preview --model MODELS --state state.pth --prompt "你好" --ab
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="statetuner",
    help="RWKV-7 state tuning for Mac — train, export, and preview init states.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def train(
    model: Path = typer.Option(..., "--model", "-m", help="转换后的 HF 模型目录"),
    data: Path = typer.Option(..., "--data", "-d", help="训练数据 jsonl"),
    test_data: Optional[Path] = typer.Option(
        None, "--test-data", help="held-out 数据(早停用);缺省则从 train 划分"
    ),
    out: Path = typer.Option(
        Path("state.npz"), "--out", "-o", help="输出 state(npz,P0 内部格式)"
    ),
    lr: float = typer.Option(0.01, "--lr", help="学习率(默认 0.01,P0 实测;1.0 会爆炸)"),
    lr_floor: float = typer.Option(1e-4, "--lr-floor", help="cosine 衰减终点"),
    warmup: int = typer.Option(10, "--warmup", help="warmup 步数"),
    ctx_len: int = typer.Option(512, "--ctx-len", help="上下文长度"),
    epochs: int = typer.Option(20, "--epochs", help="epoch 数(配早停后是上限)"),
    grad_clip: float = typer.Option(1.0, "--grad-clip"),
    max_state_std: float = typer.Option(
        1.0, "--max-state-std", help="state std 预警阈值(>此值发 warning,不中断)"
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
    seed: int = typer.Option(42, "--seed"),
):
    """训练 state tuning。事件流输出到 stdout(JSON lines)。"""
    import mlx.core as mx

    from . import events
    from .core import load_model
    from .data import load_dataset, train_test_split
    from .train import Trainer, TrainConfig, save_state_npz

    # 加载模型(patch ops 路径,训练用)
    model_path = str(model)
    typer.echo(f"# 加载模型 {model_path} (patch ops 路径)", err=True)
    mdl, tok = load_model(model_path, patch=True)
    mdl.freeze()

    samples = load_dataset(data, tok, max_len=ctx_len)
    typer.echo(f"# 训练样本: {len(samples)} 条", err=True)

    held_out = None
    if early_stop:
        if test_data is not None:
            held_out = load_dataset(test_data, tok, max_len=ctx_len)
            typer.echo(f"# held-out: {len(held_out)} 条 (来自 {test_data})", err=True)
        else:
            samples, held_out = train_test_split(samples, test_ratio=test_ratio, seed=seed)
            typer.echo(f"# held-out: {len(held_out)} 条 (从 train 划分 {test_ratio:.0%})", err=True)

    cfg = TrainConfig(
        lr=lr, lr_floor=lr_floor, warmup=warmup, ctx_len=ctx_len, epochs=epochs,
        grad_clip=grad_clip, max_state_std=max_state_std,
        early_stop=early_stop, early_stop_patience=patience,
        checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every,
        resume=resume, seed=seed,
    )

    with events.EventEmitter(file=events_file) as em:
        trainer = Trainer(mdl, cfg, em)
        result = trainer.train(samples, held_out)

    # 存 npz
    save_state_npz(result.states, out)
    typer.echo(f"# state → {out} (std={result.final_state_std:.4f})", err=True)

    # 可选导出 pth
    if export_pth:
        from .export import export_pth as _export, verify_roundtrip

        pth_path = pth_out or out.with_suffix(".pth")
        _export({i: result.states[i] for i in result.states}, pth_path)
        ok, msg = verify_roundtrip({i: result.states[i] for i in result.states}, pth_path)
        typer.echo(f"# pth → {pth_path} ({'OK' if ok else 'WARN'}: {msg})", err=True)

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
        None, "--data", "-d", help="评估数据 jsonl;缺省用内置示例"
    ),
    max_tokens: int = typer.Option(70, "--max-tokens"),
    prefix_format: str = typer.Option(
        "{text}\n", "--prefix-format", help="评估前缀格式(须与训练对齐)"
    ),
):
    """评估:对数据集逐条生成(state 注入),输出翻译结果。"""
    import json

    from .core import generate, load_model
    from .data import extract_cn_en, load_jsonl

    typer.echo(f"# 加载模型 {model} (kernel 路径)", err=True)
    mdl, tok = load_model(model, patch=False)

    if data is not None:
        items = load_jsonl(data)
        pairs = [extract_cn_en(it) for it in items]
    else:
        pairs = [
            ("今天下午三点开会，别忘了带上项目文档。", ""),
            ("由于连续降雨，部分地区出现了轻微内涝。", ""),
            ("人工智能技术正在深刻改变各行各业的生产方式。", ""),
        ]

    for i, (cn, ref) in enumerate(pairs):
        prompt = prefix_format.format(text=cn)
        out = generate(mdl, tok, prompt, state=str(state), max_tokens=max_tokens)
        out = out.split("\n")[0].strip() if "\n" in out else out.strip()
        typer.echo(f"[{i+1}] {cn}")
        if ref:
            typer.echo(f"    REF: {ref}")
        typer.echo(f"    OUT: {out}")
        typer.echo()


@app.command()
def export(
    state: Path = typer.Option(..., "--state", "-s", help="state npz(P0 内部格式)"),
    out: Path = typer.Option(..., "--out", "-o", help="输出 .pth 路径"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="导出后 round-trip 验证"),
):
    """npz → RWKV Runner 可挂载的 .pth。"""
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
def preview(
    model: Path = typer.Option(..., "--model", "-m", help="HF 模型目录"),
    state: Optional[Path] = typer.Option(
        None, "--state", "-s", help="state 文件(npz/pth);缺省=无 state 基线"
    ),
    prompt: str = typer.Option(..., "--prompt", "-p", help="输入文本"),
    max_tokens: int = typer.Option(80, "--max-tokens"),
    ab: bool = typer.Option(False, "--ab", help="A/B 对比:有 state vs 无 state 双输出"),
):
    """预览:注入 state 贪心生成。--ab 做 A/B 对比。"""
    from .core import generate, load_model

    typer.echo(f"# 加载模型 {model}", err=True)
    mdl, tok = load_model(model, patch=False)

    if ab:
        typer.echo("=== 有 state ===")
        if state is not None:
            out_s = generate(mdl, tok, prompt, state=str(state), max_tokens=max_tokens)
            typer.echo(out_s)
        else:
            typer.echo("(未提供 --state)")
        typer.echo("=== 无 state(基线)===")
        out_n = generate(mdl, tok, prompt, state=None, max_tokens=max_tokens)
        typer.echo(out_n)
    else:
        out = generate(
            mdl, tok, prompt,
            state=(str(state) if state else None),
            max_tokens=max_tokens,
        )
        typer.echo(out)


if __name__ == "__main__":
    app()
