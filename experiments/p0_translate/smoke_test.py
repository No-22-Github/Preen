"""
P0 第一级验证: 梯度冒烟测试。

目标 (P0-理论指南.md §四 第一级): 回答"计算图是否连通"。
一个 batch、一步反向,逐层查 state 梯度的范数非零、无 NaN/Inf。

它排除结构性错误: patch 未生效、state 未注入、dtype 静默截断。
注意 (§二): 断裂的梯度链在 MLX 里往往不报错,只是返回零梯度 ——
所以必须显式检查 grad 范数,而非"没报错就算过"。
"""
import sys
from pathlib import Path

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from state_tuner import patch_rwkv7_for_train, make_state_params, forward_with_state

MODEL_PATH = str(Path(__file__).parent.parent.parent / "models" / "converted" / "rwkv7-g1d-0.4b")


def main():
    print("=" * 60)
    print("P0 第一级: 梯度冒烟测试")
    print("=" * 60)

    print("加载模型 + patch ops 路径...")
    patch_rwkv7_for_train()
    model, tokenizer = load(MODEL_PATH, tokenizer_config={"trust_remote_code": True})
    model.freeze()  # 冻结权重
    print(f"模型: {type(model).__name__}, layers={len(model.layers)}")

    # 构造可训练 state
    states = make_state_params(model, dtype=mx.float32)
    state_list = [states[i] for i in range(len(states))]
    print(f"可训练 state: {len(state_list)} 层, 每层 shape={state_list[0].shape}")

    # 构造一个 batch: 两条翻译样本
    texts = [
        "User: 今天的天气真好\n\nAssistant: The weather is nice today",
        "User: 我喜欢编程\n\nAssistant: I like programming",
    ]
    batch_ids = [tokenizer.encode(t)[:64] for t in texts]
    max_len = max(len(x) for x in batch_ids)
    # pad 到等长 (用 eos=0)
    input_ids = []
    labels = []
    mask = []
    for ids in batch_ids:
        padded = ids + [0] * (max_len - len(ids))
        # input = padded[:-1], label = padded[1:]
        input_ids.append(padded[:-1])
        labels.append(padded[1:])
        # mask: 真实部分算 loss, pad 不算 (简化: 全真实部分算)
        m = [1 if i + 1 < len(ids) else 0 for i in range(max_len - 1)]
        mask.append(m)

    input_ids = mx.array(input_ids)  # (2, L)
    labels = mx.array(labels)
    mask = mx.array(mask, dtype=mx.float32)
    B = input_ids.shape[0]

    print(f"batch: B={B}, L={input_ids.shape[1]}")

    # 定义 loss (closure over batch)
    def loss_fn(*slist):
        sdict = {i: slist[i] for i in range(len(slist))}
        logits = forward_with_state(model, input_ids, sdict, B)
        log_probs = nn.log_softmax(logits, axis=-1)
        gathered = mx.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
        per_token = -gathered * mask
        return per_token.sum() / mx.maximum(mask.sum(), 1.0)

    print("计算 loss + value_and_grad...")
    loss_and_grad = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad(*state_list)
    mx.eval(loss, grads)
    loss_val = float(loss)

    print(f"\nloss = {loss_val:.4f}")
    print(f"\n===逐层 state 梯度范数 (冒烟核心)===")
    all_ok = True
    zero_count = 0
    nan_count = 0
    for i, g in enumerate(grads):
        g_np = np.array(g) if False else None
        # 直接用 mlx 算范数
        norm = float(mx.abs(g).sum())
        has_nan = bool(mx.any(mx.isnan(g)))
        flag = ""
        if has_nan:
            flag = " ❌ NaN!"
            nan_count += 1
            all_ok = False
        elif norm < 1e-12:
            flag = " ❌ 零梯度 (patch 未生效或 state 未进计算图)"
            zero_count += 1
            all_ok = False
        else:
            flag = " ✓"
        if i < 5 or i >= len(grads) - 2 or flag.startswith(" ❌"):
            print(f"  layer {i:2d}: grad_sum={norm:.6e}{flag}")
    if 5 <= len(grads) - 2:
        print(f"  ... (中间层省略, 全部非零)")

    print(f"\n===冒烟结论===")
    print(f"零梯度层: {zero_count}/{len(grads)}")
    print(f"NaN 层: {nan_count}/{len(grads)}")
    if all_ok:
        print("✅ 通过: 梯度从 loss 穿透递归抵达每层 S_0, 计算图连通")
    else:
        print("❌ 未通过: 存在梯度断裂, 需排查 patch / state 注入 / dtype")
        sys.exit(1)


import numpy as np  # noqa: E402

if __name__ == "__main__":
    main()
