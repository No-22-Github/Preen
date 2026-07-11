"""实验用 bf16 版 RWKV-7 训练 patch(选项 D)。

**这是实验代码,不入主干。** 与 src/statetuner/core.py 的 patch_rwkv7_for_train 对照:
两者结构完全相同,唯一区别是 state 进 _wkv7_step_ops 前 cast 成 bf16。

方案(需求单原文):
  S₀ 在 optimizer/参数侧保持 fp32 不变;patch 的前向里,state 进 _wkv7_step_ops
  前 cast 为 bf16,循环全程 bf16;loss 计算精度不变。梯度回传后天然回到 fp32
  master 上更新。

为什么这能让循环全程 bf16(机制论证):
  - 模型权重全部是 bf16(实测:pre_norm/attn 投影都是 mlx.core.bfloat16),
    所以 r/w/k/v/a/b 投影输出本就是 bf16。
  - 唯一的 fp32 源是可训练 S₀(make_state_params(dtype=mx.float32))。
  - MLX 类型提升规则:bf16 + fp32 = fp32。所以 fp32 state 会把整个循环提升成 fp32
    —— 这正是内存排查报告 §7 实测的大头根因。
  - 把 state cast 成 bf16 后,bf16 + bf16 不提升,循环全程 bf16,中间 state 内存减半。

梯度路径:
  state.astype(bf16) 的 VJP 对 fp32 master 是恒等投影(梯度原样流回,不缩放)。
  反向传播天然回到 fp32 更新。这是标准 mixed precision master weights 的机制。
  注:数值上 bf16 前向有舍入误差,这正是 smoke_numeric.py 要量化的风险。
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.rwkv7 import Rwkv7TimeMixing, _wkv7_step_ops


def patch_rwkv7_for_train_bf16() -> None:
    """Monkeypatch: 强制 _wkv7 走 _wkv7_step_ops 循环路径,state 全程 bf16。

    与 core.patch_rwkv7_for_train 的唯一差异:
      state 进循环前 .astype(mx.bfloat16),r/w/k/v/a/b 本就是 bf16(权重 dtype),
      整个循环维持 bf16,中间 state 内存减半。返回的 final state cast 回原 dtype
      (训练循环不使用返回的 final state,仅保持 cache 结构类型一致)。

    幂等。要恢复 kernel 路径需重新 load 模型。
    """

    def _wkv7_train_bf16(self, r, w, k, v, a, b, state):
        B, L, _, _ = r.shape
        if state is None:
            state = mx.zeros(
                (B, self.num_heads, self.head_dim, self.head_dim), dtype=r.dtype
            )
        # 关键改动:state 进 _wkv7_step_ops 前 cast bf16。
        # r/w/k/v/a/b 本就是 bf16(模型权重 dtype),bf16+bf16 不提升,循环全程 bf16。
        state_bf = state.astype(mx.bfloat16)
        ys = []
        for t in range(L):
            y, state_bf = _wkv7_step_ops(
                r[:, t], w[:, t], k[:, t], v[:, t], a[:, t], b[:, t], state_bf
            )
            ys.append(y)
        y = mx.stack(ys, axis=1).astype(r.dtype)
        # state_bf 经 astype 的 VJP(恒等)梯度流回 fp32 state master。
        # 返回时 cast 回原 dtype,保持 cache 结构类型一致。
        return y, state_bf.astype(state.dtype)

    Rwkv7TimeMixing._wkv7 = _wkv7_train_bf16


def load_model_bf16(model_path, **kwargs):
    """加载模型并 patch bf16 路径。与 core.load_model(patch=True) 对照,patch 换成 bf16 版。"""
    patch_rwkv7_for_train_bf16()
    from mlx_lm import load

    model, tokenizer = load(
        str(model_path),
        tokenizer_config={"trust_remote_code": True},
        **kwargs,
    )
    return model, tokenizer
