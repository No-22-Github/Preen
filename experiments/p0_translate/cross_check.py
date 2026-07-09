"""
P0 §五 末尾: ops 路径 vs Metal kernel 路径 自对照。

价值 (P0-理论指南.md §五):
  "同输入下两条路径的输出应在 fp16 容差内一致。这个内部对照零成本,且能
   单独验证'训练用的前向'和'推理用的前向'算的是同一个函数 —— 这对你的
   工具至关重要,因为你训练时走 ops、预览时走 kernel,两者不等价的话训练
   结果就不可信。"

原理: _wkv7_step_ops (纯 MLX ops,逐 token 循环,可微,训练用) 与
      wkv7_kernel (Metal 黑盒,快,推理用) 数值上算同一个递归。
fp16 精度下两者应在 1e-2~1e-3 量级一致。

方法:
  1. 固定 prompt,分别用 kernel 路径和 ops 路径做前向,取 logits。
  2. 比较 logits 数值 diff (细档)。
  3. 两条路径分别贪心解码,比 token 序列 (粗档)。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from mlx_lm.models.rwkv7 import Rwkv7TimeMixing, _wkv7_step_ops

MODEL_PATH = str(Path(__file__).parent.parent.parent / "models" / "converted" / "rwkv7-g1d-0.4b")


def make_kernel_forward():
    """原始 _wkv7: GPU 上走 wkv7_kernel。"""
    # 不 patch,直接用原始 _wkv7 (它会走 kernel)
    return Rwkv7TimeMixing._wkv7


def patch_ops_forward():
    """patch _wkv7 强制走 ops 循环 (训练路径)。"""

    def _wkv7_ops(self, r, w, k, v, a, b, state):
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

    Rwkv7TimeMixing._wkv7 = _wkv7_ops


def restore_kernel_forward():
    """恢复原始 _wkv7 (kernel 路径)。"""
    # 重新加载模块级原始实现 —— 保存原始引用
    Rwkv7TimeMixing._wkv7 = _ORIGINAL_WKV7


# 保存原始引用 (模块加载时)
_ORIGINAL_WKV7 = Rwkv7TimeMixing._wkv7


def forward_logits(model, input_ids):
    """前向,返回 logits (1, L, vocab)。"""
    cache = model.make_cache()
    logits = model(mx.array([input_ids]), cache)
    return logits


def greedy_decode(model, input_ids, max_tokens=40):
    """贪心解码,返回 token list。"""
    cache = model.make_cache()
    ids = mx.array([input_ids])
    logits = model(ids, cache)
    next_tok = int(mx.argmax(logits[0, -1]))
    out = [next_tok]
    for _ in range(max_tokens - 1):
        if next_tok == 0:
            break
        logits = model(mx.array([[next_tok]]), cache)
        next_tok = int(mx.argmax(logits[0, -1]))
        out.append(next_tok)
    return out


def main():
    print("=" * 60)
    print("P0 §五: ops 路径 vs Metal kernel 路径 自对照")
    print("=" * 60)

    prompts = [
        "User: 你好\n\nAssistant:",
        "The weather is nice today and",
        "Artificial intelligence is",
    ]

    print("\n===第一档 (粗档): 贪心解码 token 序列对比===")
    for prompt in prompts:
        # kernel 路径
        restore_kernel_forward()
        model_k, tok_k = load(MODEL_PATH, tokenizer_config={"trust_remote_code": True})
        ids = tok_k.encode(prompt)
        toks_kernel = greedy_decode(model_k, ids, max_tokens=30)

        # ops 路径 (需要重新加载模型以重置)
        patch_ops_forward()
        model_o, tok_o = load(MODEL_PATH, tokenizer_config={"trust_remote_code": True})
        toks_ops = greedy_decode(model_o, ids, max_tokens=30)
        restore_kernel_forward()

        match_len = 0
        for a, b in zip(toks_kernel, toks_ops):
            if a == b:
                match_len += 1
            else:
                break
        total = max(len(toks_kernel), len(toks_ops))
        print(f"\nprompt: {prompt!r}")
        print(f"  kernel: {tok_k.decode(toks_kernel)[:80]!r}")
        print(f"  ops   : {tok_o.decode(toks_ops)[:80]!r}")
        print(f"  token 一致: 前 {match_len}/{total} 个相同 "
              f"{'✓' if match_len >= 10 else '⚠️ 早分叉'}")

    print("\n===第二档 (细档): logits 数值 diff===")
    prompt = "User: 你好\n\nAssistant:"
    # kernel
    restore_kernel_forward()
    m1, _ = load(MODEL_PATH, tokenizer_config={"trust_remote_code": True})
    ids = _.encode(prompt)
    logits_kernel = forward_logits(m1, ids)
    mx.eval(logits_kernel)

    # ops
    patch_ops_forward()
    m2, _ = load(MODEL_PATH, tokenizer_config={"trust_remote_code": True})
    logits_ops = forward_logits(m2, ids)
    mx.eval(logits_ops)
    restore_kernel_forward()

    diff = mx.abs(logits_kernel.astype(mx.float32) - logits_ops.astype(mx.float32))
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    # 最后一个位置的 logits (最影响生成)
    last_diff = float(diff[0, -1].max())

    print(f"prompt: {prompt!r}")
    print(f"  logits max diff:  {max_diff:.6e}")
    print(f"  logits mean diff: {mean_diff:.6e}")
    print(f"  末位 max diff:    {last_diff:.6e}")
    print(f"\n===对照结论===")
    if max_diff < 1e-2:
        print(f"✅ 通过: 两条路径在 fp16 容差内一致 (max diff {max_diff:.2e} < 1e-2)")
        print(f"   训练(ops)与推理(kernel)算同一个函数, 训练结果可信")
    elif max_diff < 1e-1:
        print(f"⚠️ 可接受: {max_diff:.2e} 略大但 < 0.1, fp16 累加差异范围内")
    else:
        print(f"❌ 异常: {max_diff:.2e} 过大, 两条路径数值不等价, 需排查")


if __name__ == "__main__":
    main()
