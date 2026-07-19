"""训练循环产品化。

P0 实验报告 §三/§七 的修正建议落地于此:
  - 默认 lr 0.0001(非 RWKV-PEFT 的 1.0;实测 1.0 导致 state 爆炸 std 7~13)
  - state std 监控:每 epoch 记录；健康区间未标定，不设产品告警阈值
  - held-out 早停:每 epoch 在 held-out 上算 loss,连续 N 次不改善则停(废除固定 epoch)
  - checkpoint/中断恢复:存 state + optimizer 状态 + 进度,可断点续训
  - 结构化事件:全程 EventEmitter 发出 start/step/epoch_end/
    checkpoint/early_stop/final,为 CLI 输出和未来 sidecar IPC 铺路

关键实现点:
  - state 是外部 dict(不在 model 里),value_and_grad 用 *state_list 展开
  - optimizer 状态(Adam 的 m/v)需手动序列化才能恢复(MLX 不自动持久化)
  - cosine lr: warmup → peak → cosine → floor
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from . import events
from .core import (
    forward_with_state,
    make_state_params,
    patch_rwkv7_for_train,
    state_std,
)
from .data import Sample

PathLike = Union[str, Path]


@dataclass
class TrainConfig:
    """训练超参。字段名与 CLI 参数一一对应(typer 直接用)。"""

    lr: float = 1e-4  # 产品默认峰值;1.0 会爆炸
    lr_floor: float = 1e-5  # cosine 衰减终点
    warmup: int = 50  # warmup 步数
    ctx_len: int = 512
    epochs: int = 5  # 配 early_stop 后是上限
    grad_clip: float = 1.0  # value clipping(mx.clip(g,-c,c));注意是逐元素 clip
                            # 而非 global norm clip —— 改变梯度方向(非等比缩放)。
                            # state tuning 实测此值合理;改语义前先读这行。
    log_every: int = 10  # 每 N 步发一个 step 事件

    # state std 监控
    # 健康区间尚未标定；None = 只记录 std，不做阈值解释。
    max_state_std: Optional[float] = None

    # held-out 早停
    early_stop: bool = True
    early_stop_patience: int = 3  # 连续 N 次不改善则停

    # checkpoint
    checkpoint_dir: Optional[PathLike] = None
    checkpoint_every: int = 2  # 每 N epoch 存一个

    # 中断恢复
    resume: Optional[PathLike] = None

    # WKV7 kernel 模式(基础设施配置,影响训练行为 + 排查)
    #   "metal" = Metal checkpoint kernel(默认,6.67× 加速,见 docs/decision-fast-wkv7.md)
    #   "ops"   = Python _wkv7_step_ops 循环(可微基线,慢;排查 Metal 路径问题时回退)
    wkv_mode: str = "metal"
    wkv_chunk: int = 16  # Metal kernel 的 checkpoint chunk(32/16/8;16 经三轮实验验证)

    seed: int = 42

    def total_steps(self, n_samples: int) -> int:
        return self.epochs * n_samples

    def to_dict(self) -> dict:
        d = asdict(self)
        # 路径转 str(JSON 友好)
        for k in ("checkpoint_dir", "resume"):
            if d.get(k) is not None:
                d[k] = str(d[k])
        return d


@dataclass
class TrainResult:
    """训练产出。"""

    states: Dict[int, "mx.array"]  # 最终(最佳)state
    best_held_out_loss: Optional[float]
    final_loss: float
    final_state_std: float
    epochs_run: int
    elapsed: float


def cosine_lr(step: int, total_steps: int, cfg: TrainConfig) -> float:
    """lr: warmup 线性升 → cosine 衰减到 lr_floor。"""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(1, cfg.warmup)
    progress = (step - cfg.warmup) / max(1, total_steps - cfg.warmup)
    progress = min(1.0, progress)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr_floor + (cfg.lr - cfg.lr_floor) * cosine


def _to_mx_batch(sample: Sample):
    """单样本 → (input_ids(1,L), labels(1,L), mask(1,L))。

    TR1:删 bsz 参数。旧实现的 bsz>1 是"把同一条样本复制 N 份"(非真 batch),
    且训练循环从没传过 cfg.bsz、CLI 也没暴露,却进了 events.start 的 config
    快照和 metadata.json —— 在产物元数据里写假话,直接删字段。

    batch 维恒为 1 的决策仍成立,但理由在 2026-07-19 性能插桩后被更新:
    旧注释写「state tuning 对 16G 是负收益」,实测推翻——batch=2 端到端是 +10%
    正收益(单样本 478→435ms),峰值内存 5.5→7.3GB(16G 机器放得下)。但 +10%
    不值得动 Trainer:wkv7 kernel 是瓶颈(占 step 35.6%),而 wkv 从 batch 维的
    收益有限(1.82×);GEMM 虽受益大但不是瓶颈。详见 AGENTS.md「训练性能画像」。
    """
    inp = sample.input_ids
    lab = sample.labels
    msk = sample.mask
    return (
        mx.array([inp]),
        mx.array([lab]),
        mx.array([[float(x) for x in msk]], dtype=mx.float32),
    )


def _make_loss_and_grad(model, *, compile_graph: bool):
    """构造训练 loss+grad callable；Metal fast path 可复用编译图。

    loss 口径保持与历史训练循环完全一致。把 ``inp/lab/msk`` 改为显式参数，
    避免每步创建捕获不同数组的闭包，也让 ``mx.compile`` 能按输入形状缓存图。
    ``model.state`` 作为 captured inputs 声明，供 MLX 正确追踪冻结模型权重。

    ops 排查路径不编译：逐 token Python 图很大，编译会显著增加冷启动与内存，
    且 ``--no-fast-wkv`` 的职责就是保留最朴素的可微基线。
    """

    def _loss_fn(sd, inp, lab, msk):
        batch_size = inp.shape[0]
        logits = forward_with_state(model, inp, sd, batch_size)
        lp = nn.log_softmax(logits, -1)
        g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
        return (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)

    loss_and_grad = mx.value_and_grad(_loss_fn)
    if compile_graph:
        loss_and_grad = mx.compile(loss_and_grad, inputs=model.state)
    return loss_and_grad


def _eval_loss(model, samples: List[Sample], states: Dict[int, "mx.array"]) -> float:
    """在样本集上算平均 masked loss(无 grad)。供 held-out 早停用。"""
    if not samples:
        return 0.0
    total, count = 0.0, 0
    for s in samples:
        inp, lab, msk = _to_mx_batch(s)
        logits = forward_with_state(model, inp, states, 1)
        lp = nn.log_softmax(logits, -1)
        g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
        per = (-g * msk).sum()
        cnt = mx.maximum(msk.sum(), 1.0)
        total += float(per)
        count += float(cnt)
    return total / max(1.0, count)


class Trainer:
    """产品化训练循环。

    用法:
        model, tok = load_model(model_path, patch=True); model.freeze()
        cfg = TrainConfig(...)
        with EventEmitter(file="train.jsonl") as em:
            trainer = Trainer(model, cfg, em)
            result = trainer.train(samples, held_out)
        # result.states 是训好的 state dict
    """

    def __init__(self, model, config: TrainConfig, emitter: Optional[events.EventEmitter] = None):
        self.model = model
        self.cfg = config
        self.emitter = emitter or events.EventEmitter(quiet=True)
        self._rng = random.Random(config.seed)

    def train(
        self,
        samples: List[Sample],
        held_out: Optional[List[Sample]] = None,
    ) -> TrainResult:
        cfg = self.cfg
        t0 = time.time()
        random.seed(cfg.seed)
        self._rng = random.Random(cfg.seed)

        # 初始化 state + optimizer
        states = make_state_params(self.model, dtype=mx.float32)
        opt = optim.Adam(learning_rate=cfg.lr, betas=[0.9, 0.99], eps=1e-8)

        start_epoch = 0
        best_held_out = math.inf
        patience_left = cfg.early_stop_patience
        best_states = {i: mx.array(s) for i, s in states.items()}

        # 中断恢复
        if cfg.resume is not None:
            start_epoch, states, opt, best_held_out, patience_left = self._load_checkpoint(
                Path(cfg.resume), states, opt
            )
            best_states = {i: mx.array(s) for i, s in states.items()}
            self.emitter.emit(
                events.Event(
                    type="resume",
                    epoch=start_epoch,
                    message=f"Resumed from epoch {start_epoch}",
                )
            )

        total_steps = cfg.total_steps(len(samples))
        self.emitter.emit(events.start({**cfg.to_dict(), "n_samples": len(samples)}))

        # fast Metal 路径把 loss+backward 编译一次并按输入形状复用。训练数据、
        # lr、梯度裁剪和 Adam 更新仍逐步执行，步数与事件进度口径不变。
        loss_and_grad = _make_loss_and_grad(
            self.model, compile_graph=(cfg.wkv_mode == "metal")
        )

        step = start_epoch * len(samples)
        global_step_offset = step

        # 预初始化 epoch/avg_loss/sstd,防止 resume 已到末尾时 for 循环一次不进
        # 导致后续引用 NameError(review TR2)。resume 到末尾时这些值应反映
        # "没有新 epoch 跑过":epoch 回退到最后一个完成的轮次,loss/std 取当前 state。
        epoch = start_epoch - 1
        avg_loss = float("nan")
        sstd = state_std(states)

        for epoch in range(start_epoch, cfg.epochs):
            self.emitter.emit(events.epoch_start(epoch))
            order = list(range(len(samples)))
            self._rng.shuffle(order)

            losses = []
            for si in order:
                batch = _to_mx_batch(samples[si])
                inp, lab, msk = batch

                lr = cosine_lr(step, total_steps, cfg)
                opt.learning_rate = lr
                loss, grads = loss_and_grad(states, inp, lab, msk)
                # TR4: value clipping(逐元素 clip 到 [-c, c]),非 global-norm。
                # 改方向不等比缩放;state tuning 实测此阈值可用,不要按 norm clip 常识调参。
                grads = {k: mx.clip(g, -cfg.grad_clip, cfg.grad_clip) for k, g in grads.items()}
                states = opt.apply_gradients(grads, states)
                mx.eval(states, loss)
                loss_f = float(loss)
                losses.append(loss_f)

                if cfg.log_every and (step - global_step_offset) % cfg.log_every == 0:
                    self.emitter.emit(
                        events.step(step, total_steps, loss_f, lr, epoch=epoch)
                    )
                step += 1

            avg_loss = sum(losses) / len(losses)
            sstd = state_std(states)

            # held-out 评估 + 早停
            held_out_loss = None
            if cfg.early_stop and held_out:
                # M5:held-out eval 前清 MLX buffer cache。训练 step 累积了大量
                # 中间张量(softmax / VJP 图),eval 再叠一层会顶高峰值;近乎零成本。
                mx.clear_cache()
                held_out_loss = _eval_loss(self.model, held_out, states)
                if held_out_loss < best_held_out - 1e-6:
                    best_held_out = held_out_loss
                    patience_left = cfg.early_stop_patience
                    best_states = {i: mx.array(s) for i, s in states.items()}
                else:
                    patience_left -= 1

            self.emitter.emit(
                events.epoch_end(
                    epoch,
                    avg_loss,
                    sstd,
                    lr,
                    held_out_loss=held_out_loss,
                    best=(best_held_out if held_out else None),
                    patience_left=(patience_left if cfg.early_stop and held_out else None),
                )
            )
            # M5:epoch 结束后清一次 MLX buffer cache(近乎零成本的保险,
            # 防止跨 epoch 的中间张量累积顶高峰值,尤其在 16G 红线机器上)。
            mx.clear_cache()

            # state std 预警
            if cfg.max_state_std is not None and sstd > cfg.max_state_std:
                self.emitter.emit(events.std_warning(epoch, sstd, cfg.max_state_std))

            # checkpoint
            if cfg.checkpoint_dir and (epoch + 1) % cfg.checkpoint_every == 0:
                ckpt_path = self._save_checkpoint(
                    Path(cfg.checkpoint_dir),
                    epoch,
                    states,
                    opt,
                    best_held_out,
                    patience_left,
                )
                self.emitter.emit(events.checkpoint(epoch, str(ckpt_path)))

            # 早停判定(epoch 结束后)
            if cfg.early_stop and held_out and patience_left <= 0:
                self.emitter.emit(events.early_stop(epoch, best_held_out, held_out_loss))
                break

        # 用最佳 state(若用了 held-out),否则用最后
        final_states = best_states if (cfg.early_stop and held_out) else states
        # TR3:final_state_std 必须描述 final_states,而不是最后一轮的 states。
        # 早停时 final_states == best_states(可能是第 5 轮),而 sstd 是第 8 轮的;
        # 三个字段并排出现在 metadata 里会误导读者以为描述同一对象。
        final_state_std = state_std(final_states)
        elapsed = time.time() - t0

        out_path = None
        if cfg.checkpoint_dir:
            out_path = self._save_checkpoint(
                Path(cfg.checkpoint_dir),
                epoch,
                final_states,
                opt,
                best_held_out,
                patience_left,
                name="final.npz",
            )
        self.emitter.emit(
            events.final(
                str(out_path) if out_path else "(in-memory)",
                elapsed,
                best=(best_held_out if (held_out and best_held_out != math.inf) else None),
            )
        )

        return TrainResult(
            states=final_states,
            best_held_out_loss=(best_held_out if best_held_out != math.inf else None),
            final_loss=avg_loss,
            final_state_std=final_state_std,
            epochs_run=epoch + 1 - start_epoch,
            elapsed=elapsed,
        )

    # ── checkpoint 持久化 ──────────────────────────────────────
    def _save_checkpoint(
        self,
        ckpt_dir: Path,
        epoch: int,
        states: Dict[int, "mx.array"],
        opt: optim.Optimizer,
        best: float,
        patience_left: int,
        name: Optional[str] = None,
    ) -> Path:
        """存 checkpoint: state(npz) + optimizer 状态(npz) + meta(json)。

        原子写(TR5):npz/meta 先写 .tmp 再 replace,中途被打断不留半截文件。

        optimizer 状态(P0-1 修复):
          MLX Adam._state 的层键是 **int**(层号),不是 tuple —— 旧实现的
          `isinstance(k, tuple)` 永假,导致 m/v 从没存进 npz,resume 拿到的是
          刚构造、未 apply_gradients 的空 opt。
          现在显式存 _optm_{i} / _optv_{i} / _optstep(step 漏存会让续训第一步
          的 Adam bias correction 按 step=1 算 → 一个超大步)。
        """
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fname = name or f"epoch{epoch:03d}.npz"
        npz_path = ckpt_dir / fname

        arrays: dict = {f"layer_{k}": np.array(states[k]) for k in sorted(states)}

        # optimizer state(P0-1):层键是 int。step 必存(否则续训第一步爆冲)。
        if hasattr(opt, "_state"):
            st = opt._state
            for k, layer_st in st.items():
                if isinstance(k, int):
                    if "m" in layer_st:
                        arrays[f"_optm_{k}"] = np.array(layer_st["m"])
                    if "v" in layer_st:
                        arrays[f"_optv_{k}"] = np.array(layer_st["v"])
            if "step" in st:
                arrays["_optstep"] = np.array(np.uint64(st["step"]))

        # TR5: tmp + Replace 原子写(对齐 save_state_npz)。
        # tmp 路径必须以 .npz 结尾 —— np.savez 会在非 .npz 路径后再加 .npz,
        # 导致 replace 找不到刚写的文件(save_state_npz 用 ".tmp.npz" 同此理)。
        tmp_npz = npz_path.with_name(npz_path.name + ".tmp.npz")
        np.savez(tmp_npz, **arrays)
        tmp_npz.replace(npz_path)

        # meta
        meta = {
            "epoch": epoch,
            "best_held_out": (best if best != math.inf else None),
            "patience_left": patience_left,
            "lr": self.cfg.lr,
            "lr_floor": self.cfg.lr_floor,
            "warmup": self.cfg.warmup,
            "seed": self.cfg.seed,
            "n_layers": len(states),
        }
        meta_path = npz_path.with_suffix(".meta.json")
        tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_meta.replace(meta_path)

        return npz_path

    def _load_checkpoint(
        self, path: Path, states: Dict[int, "mx.array"], opt: optim.Optimizer
    ) -> Tuple[int, Dict[int, "mx.array"], optim.Optimizer, float, int]:
        """从 checkpoint 恢复: state + optimizer(含 m/v/step)+ 进度。

        P0-1 关键:必须先 opt.init(states) 让 _state 长出层键,再灌 m/v/step。
        旧实现的 opt 是刚构造、未 apply_gradients 的,_state 里只有
        {'step','learning_rate'},连层键都不存在 —— 键判断改对了也是空转。
        """
        data = np.load(path)
        n_layers = len(states)
        for i in range(n_layers):
            states[i] = mx.array(data[f"layer_{i}"])

        # 恢复 optimizer 状态(P0-1):先 init 让层键出现,再逐层灌 m/v + step。
        if hasattr(opt, "_state"):
            opt.init(states)
            for i in range(n_layers):
                m_key = f"_optm_{i}"
                v_key = f"_optv_{i}"
                if m_key in data and i in opt._state:
                    opt._state[i]["m"] = mx.array(data[m_key])
                if v_key in data and i in opt._state:
                    opt._state[i]["v"] = mx.array(data[v_key])
            if "_optstep" in data:
                opt._state["step"] = mx.array(np.uint64(data["_optstep"]))

        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        epoch = meta.get("epoch", 0) + 1  # 从下一个 epoch 开始
        best = meta.get("best_held_out")
        best = best if best is not None else math.inf
        patience = meta.get("patience_left", self.cfg.early_stop_patience)
        return epoch, states, opt, best, patience


def save_state_npz(states: Dict[int, "mx.array"], path: PathLike) -> Path:
    """把 state dict 存为 npz(P0 内部格式 layer_{i}),generate 可直接读。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"layer_{k}": np.array(states[k]) for k in sorted(states)}
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return path
