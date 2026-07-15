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
  3. 兼容推理入口:generate；完整推理 API 见 inference.py。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.rwkv7 import Rwkv7TimeMixing, _wkv7_step_ops

from .fast_wkv7 import make_wkv7_checkpoint

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


def patch_rwkv7_for_train_fast() -> None:
    """Monkeypatch(fast path):_wkv7 走 Metal checkpoint kernel(实验性)。

    与 patch_rwkv7_for_train 等价替换 _wkv7,但底层走 fast_wkv7 的整段 Metal
    kernel(forward+backward 各一次 dispatch)而非 Python ops 循环。
    state 透传给 kernel 作为 h_in(可训练 S₀ 梯度全通,实验验证)。

    何时用:--fast-wkv 开关显式开启。默认仍走 patch_rwkv7_for_train(ops 路径)。
    kernel 约束 T % 32 == 0;逐样本训练管线下样本长度不固定且常非 32 倍数,
    故闭包内就地 pad 到 32 倍数(末尾补零)、算完 slice 回真实 L。因果递归保证
    pad 段对真实 token 的 y 零影响(见下)。kernel 按 (H, L_pad) JIT 缓存。
    """

    def _wkv7_fast(self, r, w, k, v, a, b, state):
        B, L, H, D = *r.shape[:2], self.num_heads, self.head_dim
        if state is None:
            state = mx.zeros((B, H, D, D), dtype=r.dtype)

        # checkpoint kernel 要求 T % 32 == 0;pad 序列末尾到 32 倍数。
        # pad 段 r/k/v/a/b=0、w=1.0:递归 h = 1.0*h + 0 + 0 = h 不变(state 原样
        # 穿过 pad 段),因果性保证 pad 段对真实 token 的 y 零影响。返回的 h_out
        # 含 pad 段递归但训练下游不用(S₀ 每步重新注入),故 w pad 值无副作用。
        L_pad = ((L + 31) // 32) * 32
        if L_pad != L:
            pad = L_pad - L
            r = mx.pad(r, [(0, 0), (0, pad), (0, 0), (0, 0)])
            w = mx.pad(w, [(0, 0), (0, pad), (0, 0), (0, 0)], constant_values=1.0)
            k = mx.pad(k, [(0, 0), (0, pad), (0, 0), (0, 0)])
            v = mx.pad(v, [(0, 0), (0, pad), (0, 0), (0, 0)])
            a = mx.pad(a, [(0, 0), (0, pad), (0, 0), (0, 0)])
            b = mx.pad(b, [(0, 0), (0, pad), (0, 0), (0, 0)])

        wkv7 = make_wkv7_checkpoint(B, L_pad, H, D)
        y, h_out = wkv7(r, w, k, v, a, b, state)
        # slice 回真实 L(pad 段 y 丢弃)。
        y = y[:, :L] if L_pad != L else y
        return y.astype(r.dtype), h_out.astype(state.dtype)

    Rwkv7TimeMixing._wkv7 = _wkv7_fast


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
    """各层 state std 的均值，仅作训练观测；健康阈值尚未标定。"""
    stds = [float(np.array(states[i]).std()) for i in sorted(states)]
    return float(np.mean(stds)) if stds else 0.0


def _load_state_dict(state: StateInput) -> Optional[Dict[int, "mx.array"]]:
    """把 StateInput 统一成 {layer: mx.array} 或 None。

    支持:
      None                       → None(模型默认零 state)
      dict[int, mx.array]        → 原样
      npz 路径(P0 内部格式 layer_{i}) → 读 npz
      pth 路径(RWKV-PEFT 格式 blocks.{i}.att.time_state)→ 按 x070 原样读回
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
        from .export import load_npz_as_numpy

        raw = load_npz_as_numpy(path)
        return {i: mx.array(arr) for i, arr in raw.items()}

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
    """兼容入口：注入 state 做贪心生成并仅返回文本。

    state:
      None              → 模型默认零 state(基线)
      dict / npz 路径   → 直接注入
      pth 路径          → 按 RWKV-7 x070 原样读回后注入
    贪心解码(argmax),遇 eos(token 0)停止。eos 本身不解码进输出
    (旧实现把 eos append 进结果,导致输出末尾出现 <|...end_of_text|> 字面量)。
    """
    from .inference import GenerationConfig, InferenceEngine

    result = InferenceEngine(model, tokenizer).generate(
        prompt,
        state=state,
        # 纯贪心:显式关闭重复惩罚。core.generate 是底层兼容入口(golden 测试
        # 守护编码路径),penalty 是上层 ChatSession/CLI 的采样增强,不在此生效。
        config=GenerationConfig(
            max_tokens=max_tokens, temperature=0.0,
            presence_penalty=0.0, frequency_penalty=0.0,
        ),
    )
    return result.text


def load_model(model_path: Union[str, Path], *, patch: bool = False):
    """加载 mlx-lm RWKV-7 模型 + tokenizer。

    patch=True 时 patch ops 路径(训练用);False 时走默认 kernel(推理用)。
    返回 (model, tokenizer)。trust_remote_code=True(World tokenizer)。

    抑制 transformers 的 "model of type" 警告:HF 把 rwkv7 config 的 model_type
    与加载器基类对不上(我们走 mlx-lm 自己的 RWKV7 模型类,不经 HF AutoModel),
    这条警告对用户纯噪声,压到 ERROR 级。
    """
    from mlx_lm import load

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except ImportError:
        pass  # transformers 不可用时无警告可压

    if patch:
        patch_rwkv7_for_train()
    model, tokenizer = load(
        str(model_path), tokenizer_config={"trust_remote_code": True}
    )
    return model, tokenizer
