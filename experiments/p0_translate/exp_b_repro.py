"""
决定性诊断: 用 rwkv pip RWKV_x070 复现台式机实验。
直接注入我们的训练 state(两个方向), forward 中文, 看输出语义。

这绕开 Runner, 直接测 "我们的 state 在官方 x070 实现里是否产生翻译行为"。

state list 结构(RWKV_x070.generate_zero_state, model.py:284-290):
  state[i*3+0] = zeros(n_embd)       # att x_prev
  state[i*3+1] = att_kv (H,N,N)      # ← 我们注入这里
  state[i*3+2] = zeros(n_embd)       # ffn x_prev

三个条件:
  C0) state=None          → 基线(应中文续写/中性)
  C1) state[att_kv] = 训练原方向(= ep04.npz 原样)  → 若出翻译, 则原方向对
  C2) state[att_kv] = 训练转置方向(= .pth 文件内容) → 若出翻译, 则转置方向对

关键判据: 哪个方向产出"对该中文的英文翻译", 哪个就是 Runner 该拿到的。
"""
from __future__ import annotations
import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ["RWKV_V7_ON"] = "1"

import numpy as np
import torch

RWKV_NATIVE = str(REPO / "models" / "rwkv7-g1d-0.4b-20260210-ctx8192")
NPZ = str(REPO / "experiments" / "p0_translate" / "checkpoints_v3" / "ep04.npz")

TEST_SENT = "谢谢你的帮助。"  # 训练集外, 短句便于肉眼判断

PROMPT = f"User: {TEST_SENT}\n\nAssistant:"


def build_state_list(model, att_kv_per_layer):
    """构造 RWKV_x070 期望的 state list, att_kv 注入指定方向。"""
    n_layer = model.n_layer
    n_embd = model.n_embd
    state = [None] * (n_layer * 3)
    for i in range(n_layer):
        state[i * 3 + 0] = torch.zeros(n_embd, dtype=torch.float32)
        state[i * 3 + 1] = torch.tensor(att_kv_per_layer[i], dtype=torch.float32)
        state[i * 3 + 2] = torch.zeros(n_embd, dtype=torch.float32)
    return state


def greedy_decode(model, pipeline, prompt, state, max_tokens=40):
    """贪心解码(温度0), 复现 Runner 行为。"""
    import torch.nn.functional as F
    ids = pipeline.encode(prompt)
    out, state = model.forward(ids, state)
    generated = []
    for _ in range(max_tokens):
        token = int(torch.argmax(out).item())
        if token == 0:
            break
        generated.append(token)
        out, state = model.forward([token], state)
    return pipeline.decode(generated), state


def main():
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE

    print("=" * 70)
    print("决定性诊断: rwkv pip RWKV_x070 复现台式机实验")
    print(f"模型: {RWKV_NATIVE}")
    print(f"测试句: {TEST_SENT}")
    print(f"前缀: {PROMPT!r}")
    print("=" * 70)

    model = RWKV(RWKV_NATIVE, "cpu fp32")
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

    # 加载训练 state, 准备两个方向
    data = np.load(NPZ)
    orig = {i: data[f"layer_{i}"].astype(np.float32) for i in range(len(data.files))}
    transposed = {i: np.ascontiguousarray(orig[i].swapaxes(-2, -1)) for i in orig}

    print("\n--- C0) 基线: state=None ---")
    out0, _ = greedy_decode(model, pipeline, PROMPT, None, max_tokens=40)
    print(f"  输出: {out0!r}")

    print("\n--- C1) 注入训练原方向 (ep04.npz 原样) ---")
    s1 = build_state_list(model, orig)
    out1, _ = greedy_decode(model, pipeline, PROMPT, s1, max_tokens=40)
    print(f"  输出: {out1!r}")

    print("\n--- C2) 注入训练转置方向 (= 当前 .pth 文件内容) ---")
    s2 = build_state_list(model, transposed)
    out2, _ = greedy_decode(model, pipeline, PROMPT, s2, max_tokens=40)
    print(f"  输出: {out2!r}")

    print("\n" + "=" * 70)
    print("判读:")
    print("  期望: 某个方向应输出 'Thank you for your help.' 类翻译")
    print("  C0(基线) 应是中性/中文续写 — 验证基模型健康")
    print("  C1 vs C2 哪个出翻译 = 哪个方向是 Runner 该注入的")


if __name__ == "__main__":
    main()
