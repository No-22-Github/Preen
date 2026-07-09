"""
RWKV-7 State Tuner 核心。

P0 理论要点 (见 P0-理论指南.md):
  §一: 只优化每层每头的 64×64 state 矩阵 S_0 (初始状态),其余权重全冻。
  §二: 必须走 _wkv7_step_ops 纯 ops 路径 (可微),patch 掉 Metal kernel 分发。
  §三: lr 1.0 (state tuning 特性), ctx 短, 每条样本从 S_0 独立启动, loss mask 只算 Assistant。

实现三步:
  1. patch_rwkv7_for_train: monkeypatch Rwkv7TimeMixing._wkv7 强制走 ops 循环。
  2. StateParams: 可训练 state (每层 (H, D, D) fp32),冻结模型权重。
  3. 训练循环: value_and_grad 只对 state 求导, loss = masked cross-entropy。
"""
import math
import time
from dataclasses import dataclass

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.rwkv7 import Rwkv7TimeMixing, _wkv7_step_ops


def patch_rwkv7_for_train():
    """Monkeypatch: 强制 _wkv7 走 _wkv7_step_ops 循环路径。

    原始 _wkv7 在 GPU 上走 wkv7_kernel (Metal 黑盒,无 VJP,梯度静默断裂)。
    训练必须走 ops 路径 (纯 MLX 原语,每步有 VJP,梯度能穿透 512 步回到 S_0)。

    替换后的逻辑: 无视设备,始终走逐 token 循环。
    """

    def _wkv7_train(self, r, w, k, v, a, b, state):
        B, L, _, _ = r.shape
        if state is None:
            state = mx.zeros(
                (B, self.num_heads, self.head_dim, self.head_dim), dtype=r.dtype
            )
        ys = []
        for t in range(L):
            y, state = _wkv7_step_ops(
                r[:, t], w[:, t], k[:, t], v[:, t], a[:, t], b[:, t], state
            )
            ys.append(y)
        y = mx.stack(ys, axis=1).astype(r.dtype)
        return y, state

    Rwkv7TimeMixing._wkv7 = _wkv7_train
    # print("[patch] Rwkv7TimeMixing._wkv7 → ops 路径 (可微)")


def make_state_params(model, dtype=mx.float32):
    """构造可训练 state 参数: 每层一个 (H, D, D) fp32 张量,零初始化。

    返回 dict[layer_idx, mx.array],作为唯一可训练参数。
    形状不含 batch 维,前向时广播到 (B, H, D, D)。
    """
    states = {}
    H = model.args.hidden_size // model.args.head_dim
    D = model.args.head_dim
    for i, layer in enumerate(model.layers):
        # 关键: 初始为零 (与推理默认 S_0=0 一致),梯度从此处生长
        states[i] = mx.zeros((H, D, D), dtype=dtype)
    return states


def freeze_model(model):
    """冻结模型全部权重: 不参与梯度。"""
    model.freeze()
    # mx.freeze 会让参数 .requires_grad=False,但 state 不在模型里(外部 dict)


def build_state_cache(states, batch_size):
    """把 state dict 转成 model 期望的 cache 结构。

    Rwkv7Layer.__call__(x, v_first, cache): cache 是 ArraysCache(size=3),
    cache[1] = state。每层独立,每条样本独立 (广播 batch)。

    我们不直接用 ArraysCache (它有自动更新逻辑),而是用普通 list,
    每层 cache = [None, state_broadcast, None],其中:
      cache[0] = token_shift state (None 让模型内部零初始化)
      cache[1] = 可训练 state (广播到 batch)
      cache[2] = ffn token_shift state (None)
    """
    caches = []
    for i in sorted(states.keys()):
        s = states[i]  # (H, D, D)
        # 广播到 batch: (B, H, D, D)
        s_batched = mx.broadcast_to(s, (batch_size,) + s.shape)
        cache = [None, s_batched, None]
        caches.append(cache)
    return caches


def forward_with_state(model, input_ids, states, batch_size):
    """前向传播,注入可训练 state,返回 logits。

    input_ids: (B, L) int
    states: dict[layer_idx → (H,D,D)]
    返回 logits (B, L, vocab)
    """
    caches = build_state_cache(states, batch_size)
    logits = model(input_ids, caches)
    return logits


def compute_loss(model, batch, states):
    """masked 交叉熵 loss。

    batch: (input_ids, labels, mask, lengths)
      input_ids: (B, L)
      labels: (B, L) 偏移一位的目标
      mask: (B, L) 1=算 loss (Assistant 段), 0=忽略
    states: 可训练 state dict

    loss = sum(mask * ce(logits, labels)) / sum(mask)
    """
    input_ids, labels, mask, lengths = batch
    B, L = input_ids.shape

    def _loss_fn(*state_list):
        # 把 list 重组回 dict (value_and_grad 需要 flat 参数)
        sdict = {i: state_list[i] for i in range(len(state_list))}
        logits = forward_with_state(model, input_ids, sdict, B)
        # logits: (B, L, vocab)
        log_probs = nn.log_softmax(logits, axis=-1)
        # gather 每个 label 的 log_prob
        # labels: (B, L) → (B, L, 1)
        gathered = mx.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
        # gathered: (B, L)
        per_token = -gathered * mask  # masked
        total = per_token.sum()
        count = mx.maximum(mask.sum(), 1.0)
        return total / count

    return _loss_fn


@dataclass
class TrainConfig:
    lr_peak: float = 1.0        # state tuning 学习率 (P0 §三, 非笔误)
    lr_floor: float = 0.01      # cosine 衰减终点
    warmup: int = 10            # warmup 步数
    ctx_len: int = 512
    bsz: int = 1
    epochs: int = 15
    grad_clip: float = 1.0      # state tuning 梯度可能大,裁剪
    log_every: int = 10


def cosine_lr(step, total_steps, cfg):
    """lr: warmup → peak → cosine → floor。"""
    if step < cfg.warmup:
        return cfg.lr_peak * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, total_steps - cfg.warmup)
    progress = min(1.0, progress)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr_floor + (cfg.lr_peak - cfg.lr_floor) * cosine


def generate(model, tokenizer, prompt, state_npz=None, max_tokens=80):
    """注入训练 state 做贪心生成。

    state_npz 非 None 时,从 npz 加载每层 state (key: layer_{i}, shape (H,D,D))
    注入到 cache[1];None 时用模型默认零 state (基线)。
    贪心解码 (argmax),遇 eos (token 0) 停止。
    """
    if state_npz is not None:
        caches = []
        n_layers = len(model.layers)
        data = np.load(state_npz)
        for i in range(n_layers):
            s = mx.array(data[f"layer_{i}"])
            s_b = mx.broadcast_to(s, (1,) + s.shape)
            caches.append([None, s_b, None])
    else:
        caches = model.make_cache()

    prompt_ids = tokenizer.encode(prompt)
    input_ids = mx.array([prompt_ids])
    generated = []
    for _ in range(max_tokens):
        logits = model(input_ids, caches)
        next_token = int(mx.argmax(logits[0, -1], axis=-1))
        generated.append(next_token)
        if next_token == 0:  # eos
            break
        input_ids = mx.array([[next_token]])
    return tokenizer.decode(generated)
