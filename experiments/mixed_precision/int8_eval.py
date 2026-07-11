"""方案 E 第3项 eval 判据(裁决选项丁)。

判据(裁决确认):
  两组最终 S₀ 均挂 M(全精度 bf16 模型),对 200 条训练集做纯前向 eval loss。
  配对判据 = |eval@M(int8) − eval@M(基线)| / 基线 < 2%。

诊断加测(不进判据):
  int8-S₀ 的 eval loss @M_q 与 @M 并排,刻画失配代价一个标量。

口径:
  - loss mask 与训练一致(从 prefix_len-1 起 mask=1,含 stop_token)
  - 三段加载串行,每段前 mx.clear_cache(),避免 allocator 池残留污染
    (裁决 §7 指出:@M_q 解码的 cache 3.15G 是池残留,不是量化模型需求)
  - 标签二维化:(bf16,fp32) 基线 / (int8,fp32) 方案E
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "mixed_precision"))

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from statetuner.core import load_model, patch_rwkv7_for_train, make_state_params, forward_with_state
from statetuner.data import load_qa_dataset
from statetuner.train import _to_mx_batch


MODEL = ROOT / "models" / "converted" / "rwkv7-g1g-1.5b"
DATA = ROOT / "train_data" / "NekoQA_10k" / "nekoqa_smoke_200.json"
TC = ROOT / "experiments" / "mixed_precision" / "data" / "int8_traincompare"


def _int8_predicate(path, module):
    """量化白名单:Linear + Embedding,排除 LoRA + 不可整除层。"""
    if "lora" in path.lower():
        return False
    if isinstance(module, nn.Linear):
        return module.weight.shape[-1] % 64 == 0
    if isinstance(module, nn.Embedding):
        return module.weight.shape[-1] % 64 == 0
    return False


def load_state_npz(path: Path) -> dict:
    """读 state.npz(P0 内部格式 layer_{i})→ {layer_idx: mx.array}。"""
    z = np.load(path)
    keys = sorted([k for k in z.files if k.startswith("layer_")],
                  key=lambda x: int(x.split("_")[1]))
    return {int(k.split("_")[1]): mx.array(z[k]) for k in keys}


def eval_loss_on(model, tok, samples, states: dict) -> float:
    """200 条逐条前向 masked loss(与训练 _loss_fn 同口径),返回平均。

    mask: 从 prefix_len-1 起(预测第一个 target token 的位置),含 stop_token。
    """
    total_loss = 0.0
    n = 0
    for s in samples:
        inp, lab, msk = _to_mx_batch(s)
        logits = forward_with_state(model, inp, states, 1)
        lp = nn.log_softmax(logits, -1)
        g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
        loss = (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)
        mx.eval(loss)
        total_loss += float(loss)
        n += 1
    return total_loss / n


def main():
    patch_rwkv7_for_train()

    # ── ① 基线 S₀ @M ──
    print("── [1/3] 基线 (bf16,fp32) S₀ @M ──")
    mx.clear_cache()
    mdl, tok = load_model(MODEL, patch=True)
    mdl.freeze()
    samples = load_qa_dataset(str(DATA), tok, max_len=512)
    states_base = load_state_npz(TC / "15b_s42_fp32" / "state.npz")
    eval_base = eval_loss_on(mdl, tok, samples, states_base)
    print(f"  eval@M(基线) = {eval_base:.5f}")
    del mdl, states_base
    mx.clear_cache()

    # ── ② int8 S₀ @M(判据)──
    print("\n── [2/3] int8 S₀ @M (判据) ──")
    mx.clear_cache()
    mdl, tok = load_model(MODEL, patch=True)
    mdl.freeze()
    states_i8 = load_state_npz(TC / "15b_s42_int8" / "state.npz")
    eval_i8_M = eval_loss_on(mdl, tok, samples, states_i8)
    print(f"  eval@M(int8) = {eval_i8_M:.5f}")
    del mdl
    mx.clear_cache()

    # ── ③ int8 S₀ @M_q(诊断,不进判据)──
    print("\n── [3/3] int8 S₀ @M_q (诊断,不进判据) ──")
    mx.clear_cache()
    mdl, tok = load_model(MODEL, patch=True)
    mdl.freeze()
    nn.quantize(mdl, group_size=64, bits=8, class_predicate=_int8_predicate)
    eval_i8_Mq = eval_loss_on(mdl, tok, samples, states_i8)
    print(f"  eval@M_q(int8) = {eval_i8_Mq:.5f}")
    del mdl, states_i8
    mx.clear_cache()

    # ── 判据 ──
    rel_diff = abs(eval_i8_M - eval_base) / eval_base
    verdict = "🟢 绿(<2%)" if rel_diff < 0.02 else "🔴 红(≥2%)"

    print("\n" + "=" * 60)
    print("eval 判据(裁决选项丁)")
    print("=" * 60)
    print(f"  eval@M (bf16,fp32) 基线 = {eval_base:.5f}")
    print(f"  eval@M (int8,fp32)     = {eval_i8_M:.5f}")
    print(f"  |差|/基线              = {rel_diff*100:+.2f}%")
    print(f"  判据 <2%               → {verdict}")
    print()
    print(f"  [诊断,不进判据]")
    print(f"  eval@M_q(int8)         = {eval_i8_Mq:.5f}")
    print(f"  失配代价 @M_q−@M       = {eval_i8_Mq - eval_i8_M:+.5f} "
          f"({(eval_i8_Mq-eval_i8_M)/eval_i8_M*100:+.2f}%)")

    result = {
        "criterion": "eval_option_d",
        "rule": "|eval@M(int8) - eval@M(baseline)| / baseline < 2%",
        "eval_base_M": round(eval_base, 5),
        "eval_int8_M": round(eval_i8_M, 5),
        "rel_diff_pct": round(rel_diff * 100, 3),
        "verdict": "PASS" if rel_diff < 0.02 else "FAIL",
        "diagnostic": {
            "eval_int8_Mq": round(eval_i8_Mq, 5),
            "mismatch_cost_Mq_minus_M": round(eval_i8_Mq - eval_i8_M, 5),
            "mismatch_cost_pct": round((eval_i8_Mq - eval_i8_M) / eval_i8_M * 100, 3),
            "note": "失配代价标量,不进判据",
        },
    }
    out = TC / "eval_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
