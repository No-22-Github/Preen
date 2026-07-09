"""训练循环产品化。

P0 实验报告 §三/§七 的修正建议落地于此:
  - 默认 lr 0.01(非 RWKV-PEFT 的 1.0;实测 1.0 导致 state 爆炸 std 7~13)
  - state std 监控:>1.0 预警(可能爆炸),但不中断训练
  - held-out 早停:每 epoch 在 held-out 上算 loss,连续 N 次不改善则停(废除固定 epoch)
  - checkpoint/中断恢复:存 state + optimizer 状态 + 进度,可断点续训
  - 结构化事件:全程 EventEmitter 发出 start/step/epoch_end/std_warning/
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
    compute_loss,
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

    lr: float = 0.01  # 默认 0.01(尊重 P0 实测;1.0 会爆炸)
    lr_floor: float = 1e-4  # cosine 衰减终点
    warmup: int = 10  # warmup 步数
    ctx_len: int = 512
    bsz: int = 1
    epochs: int = 20  # 配 early_stop 后是上限
    grad_clip: float = 1.0  # state tuning 梯度可能大,裁剪
    log_every: int = 10  # 每 N 步发一个 step 事件

    # state std 监控
    max_state_std: float = 1.0  # > 此值发 std_warning(不中断)

    # held-out 早停
    early_stop: bool = True
    early_stop_patience: int = 3  # 连续 N 次不改善则停

    # checkpoint
    checkpoint_dir: Optional[PathLike] = None
    checkpoint_every: int = 2  # 每 N epoch 存一个

    # 中断恢复
    resume: Optional[PathLike] = None

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


def _to_mx_batch(sample: Sample, bsz: int = 1):
    """单样本 → (input_ids(B,L), labels(B,L), mask(B,L))。bsz>1 时重复。"""
    inp = sample.input_ids
    lab = sample.labels
    msk = sample.mask
    if bsz == 1:
        return (
            mx.array([inp]),
            mx.array([lab]),
            mx.array([[float(x) for x in msk]], dtype=mx.float32),
        )
    return (
        mx.array([inp] * bsz),
        mx.array([lab] * bsz),
        mx.array([[float(x) for x in msk]] * bsz, dtype=mx.float32),
    )


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
        cfg = TrainConfig(lr=0.01, ...)
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
                    message=f"从 epoch {start_epoch} 恢复",
                )
            )

        total_steps = cfg.total_steps(len(samples))
        self.emitter.emit(events.start({**cfg.to_dict(), "n_samples": len(samples)}))

        step = start_epoch * len(samples)
        global_step_offset = step

        for epoch in range(start_epoch, cfg.epochs):
            self.emitter.emit(events.epoch_start(epoch))
            order = list(range(len(samples)))
            self._rng.shuffle(order)

            losses = []
            for si in order:
                batch = _to_mx_batch(samples[si])
                inp, lab, msk = batch
                B = inp.shape[0]

                # 用 dict 输入闭包, value_and_grad 返回 dict grads
                def _loss_fn(sd):
                    logits = forward_with_state(self.model, inp, sd, B)
                    lp = nn.log_softmax(logits, -1)
                    g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
                    return (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)

                lr = cosine_lr(step, total_steps, cfg)
                opt.learning_rate = lr
                loss, grads = mx.value_and_grad(_loss_fn)(states)
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

            # state std 预警
            if sstd > cfg.max_state_std:
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
            final_state_std=sstd,
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
        """存 checkpoint: state(npz) + meta(json, 含 optimizer 状态)。

        npz: layer_{i} (P0 内部格式, generate 直接可读)
        meta: epoch / step / best / patience_left / lr / optimizer.m / optimizer.v
              (Adam 的 m/v 手动序列化为 npz,恢复时重建)
        """
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fname = name or f"epoch{epoch:03d}.npz"
        npz_path = ckpt_dir / fname

        # state
        state_arrays = {f"layer_{k}": np.array(states[k]) for k in sorted(states)}
        # optimizer state(Adam: m + v, key 是参数名)
        opt_m = {}
        opt_v = {}
        if hasattr(opt, "_state"):
            for k, st in opt._state.items():
                if isinstance(k, tuple) and len(k) == 2:
                    param_key = k[1]
                    if "m" in st:
                        opt_m[f"layer_{param_key}"] = np.array(st["m"])
                    if "v" in st:
                        opt_v[f"layer_{param_key}"] = np.array(st["v"])

        np.savez(npz_path, **state_arrays, **{f"_optm_{k}": v for k, v in opt_m.items()},
                 **{f"_optv_{k}": v for k, v in opt_v.items()})

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
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return npz_path

    def _load_checkpoint(
        self, path: Path, states: Dict[int, "mx.array"], opt: optim.Optimizer
    ) -> Tuple[int, Dict[int, "mx.array"], optim.Optimizer, float, int]:
        """从 checkpoint 恢复: state + optimizer + 进度。"""
        data = np.load(path)
        n_layers = len(states)
        for i in range(n_layers):
            states[i] = mx.array(data[f"layer_{i}"])

        # 恢复 optimizer 状态
        if hasattr(opt, "_state"):
            for k in list(opt._state.keys()):
                if isinstance(k, tuple) and len(k) == 2:
                    param_key = k[1]
                    m_key = f"_optm_layer_{param_key}"
                    v_key = f"_optv_layer_{param_key}"
                    if m_key in data and v_key in data:
                        opt._state[k]["m"] = mx.array(data[m_key])
                        opt._state[k]["v"] = mx.array(data[v_key])

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
    np.savez(path, **arrays)
    return path
