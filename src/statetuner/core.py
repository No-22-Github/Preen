"""RWKV-7 State Tuner 核心引擎。

理论要点(见 docs/P0-理论指南.md):
  §一: 只优化每层每头的 64×64 初始状态矩阵 S₀,其余权重全冻。
  §二: 必须走 _wkv7_step_ops 纯 ops 路径(可微),patch 掉 Metal kernel 分发。
       原始 _wkv7 在 GPU 上走 wkv7_kernel(Metal 黑盒,无 VJP,梯度静默断裂)。

state 语义(导出 .pth 时的关键依据):
  _wkv7_step_ops 里 state 是 (B,H,D,D),递归为
    state = state * w + v ⊗ k + sab ;   y = state @ r
  即 v 在行、k 在列、r 缩并最后一维(列)。
  这与 BlinkDL CUDA kernel(rwkv7.cu)同向,也是 rwkv pip RWKV_x070 同向(实验A验证)。
  RWKV-Runner 对 x070(version>=7) 加载 .pth 时**不** transpose(rwkv.py:843),
  所以导出器直接存训练方向原样(详见 export.py)。

本模块三件事:
  1. patch_rwkv7_for_train: monkeypatch Rwkv7TimeMixing._wkv7 强制走 ops 循环。
  2. state 构造/注入: make_state_params / build_state_cache / forward_with_state。
  3. 推理: generate(支持 dict / npz 路径 / pth 路径 / None)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.rwkv7 import Rwkv7TimeMixing, _wkv7_step_ops

# state 参数类型: {layer_idx: mx.array(H,D,D)} 或路径(npz/pth)或 None(零 state)
StateInput = Optional[Union[Dict[int, "mx.array"], str, Path]]


def patch_rwkv7_for_train() -> None:
    """Monkeypatch: 强制 _wkv7 走 _wkv7_step_ops 循环路径。

    原始 _wkv7 在 GPU 上走 wkv7_kernel(Metal 黑盒,无 VJP,梯度静默断裂)。
    训练必须走 ops 路径(纯 MLX 原语,每步有 VJP,梯度能穿透 512 步回到 S₀)。

    幂等: 多次调用安全(每次替换为同样逻辑)。要恢复原 kernel 路径需重新 load 模型。
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


def make_state_params(model, dtype=mx.float32) -> Dict[int, "mx.array"]:
    """构造可训练 state 参数: 每层一个 (H, D, D) fp32 张量,零初始化。

    返回 dict[layer_idx → mx.array(H,D,D)],作为唯一可训练参数。
    形状不含 batch 维,前向时广播到 (B, H, D, D)。
    初始为零(与推理默认 S₀=0 一致),梯度从此处生长。
    """
    states: Dict[int, "mx.array"] = {}
    H = model.args.hidden_size // model.args.head_dim
    D = model.args.head_dim
    for i in range(len(model.layers)):
        states[i] = mx.zeros((H, D, D), dtype=dtype)
    return states


def build_state_cache(states: Dict[int, "mx.array"], batch_size: int):
    """把 state dict 转成 model 期望的 cache 结构。

    Rwkv7Layer.__call__(x, v_first, cache): cache 是 ArraysCache(size=3),
    cache[1] = state。我们用普通 list(避开 ArraysCache 的自动更新逻辑),
    每层 cache = [None, state_broadcast, None]:
      cache[0] = token_shift state (None 让模型内部零初始化)
      cache[1] = 可训练 state (广播到 batch)
      cache[2] = ffn token_shift state (None)
    """
    caches = []
    for i in sorted(states.keys()):
        s = states[i]  # (H, D, D)
        s_batched = mx.broadcast_to(s, (batch_size,) + s.shape)
        cache = [None, s_batched, None]
        caches.append(cache)
    return caches


def forward_with_state(model, input_ids, states: Dict[int, "mx.array"], batch_size: int):
    """前向传播,注入可训练 state,返回 logits。

    input_ids: (B, L) int
    states: {layer_idx → (H,D,D)}
    返回 logits (B, L, vocab)
    """
    caches = build_state_cache(states, batch_size)
    logits = model(input_ids, caches)
    return logits


def compute_loss(model, batch, states: Dict[int, "mx.array"]):
    """masked 交叉熵 loss(closure,供 mx.value_and_grad 使用)。

    batch: (input_ids, labels, mask)
      input_ids: (B, L)
      labels:    (B, L) 偏移一位的目标
      mask:      (B, L) 1=算 loss(Assistant 段),0=忽略
    返回一个 callable(*state_list) -> loss,把 list 重组回 dict。
    """
    input_ids, labels, mask = batch
    B, L = input_ids.shape

    def _loss_fn(*state_list):
        sdict = {i: state_list[i] for i in range(len(state_list))}
        logits = forward_with_state(model, input_ids, sdict, B)
        log_probs = nn.log_softmax(logits, axis=-1)
        gathered = mx.take_along_axis(
            log_probs, labels[..., None], axis=-1
        ).squeeze(-1)
        per_token = -gathered * mask
        total = per_token.sum()
        count = mx.maximum(mask.sum(), 1.0)
        return total / count

    return _loss_fn


def state_std(states: Dict[int, "mx.array"]) -> float:
    """各层 state std 的均值,用于训练时监控是否数值爆炸。

    P0 实测:正常推理 state std ~0.01~0.23;lr=1.0 训练会涨到 7~13(爆炸)。
    >1.0 视为异常,train.py 据此发 std_warning。
    """
    stds = [float(np.array(states[i]).std()) for i in sorted(states)]
    return float(np.mean(stds)) if stds else 0.0


def _load_state_dict(state: StateInput) -> Optional[Dict[int, "mx.array"]]:
    """把 StateInput 统一成 {layer: mx.array} 或 None。

    支持:
      None                       → None(模型默认零 state)
      dict[int, mx.array]        → 原样
      npz 路径(P0 内部格式 layer_{i}) → 读 npz
      pth 路径(RWKV-PEFT 格式 blocks.{i}.att.time_state)→ 读 pth 并 transpose(1,2)
        注: pth 里存的是 transpose 后的, 加载要转回来还原成 MLX 方向的 S
    """
    if state is None:
        return None
    if isinstance(state, dict):
        return dict(state)

    path = Path(state)
    if not path.exists():
        raise FileNotFoundError(f"state 文件不存在: {path}")

    if path.suffix == ".npz":
        # P0 内部格式: layer_{i}
        data = np.load(path)
        return {i: mx.array(data[f"layer_{i}"]) for i in range(len(data.files))}

    if path.suffix == ".pth":
        # RWKV-7 (x070) 格式: blocks.{i}.att.time_state, 原样存训练方向(Runner 不转置)
        # 见 export.py docstring: x070 导出不 swapaxes, 直接读回即训练方向
        from .export import load_pth_as_numpy

        raw = load_pth_as_numpy(path)  # {layer: ndarray}, 原样 = 训练方向
        return {i: mx.array(arr) for i, arr in raw.items()}

    raise ValueError(f"不支持的 state 文件格式: {path.suffix} (支持 .npz / .pth)")


def generate(
    model,
    tokenizer,
    prompt: str,
    state: StateInput = None,
    max_tokens: int = 80,
) -> str:
    """注入 state 做贪心生成。

    state:
      None              → 模型默认零 state(基线)
      dict / npz 路径   → 直接注入
      pth 路径          → 读回并 transpose 还原后注入
    贪心解码(argmax),遇 eos(token 0)停止。
    """
    state_dict = _load_state_dict(state)

    if state_dict is not None:
        caches = build_state_cache(state_dict, batch_size=1)
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


def load_model(model_path: Union[str, Path], *, patch: bool = False):
    """加载 mlx-lm RWKV-7 模型 + tokenizer。

    patch=True 时 patch ops 路径(训练用);False 时走默认 kernel(推理用)。
    返回 (model, tokenizer)。trust_remote_code=True(World tokenizer)。
    """
    from mlx_lm import load

    if patch:
        patch_rwkv7_for_train()
    model, tokenizer = load(
        str(model_path), tokenizer_config={"trust_remote_code": True}
    )
    return model, tokenizer
