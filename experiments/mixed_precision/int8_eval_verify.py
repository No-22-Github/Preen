"""eval 仪器验证单(纯前向,分钟级,不含任何训练)。

四项验收:
  1. 量化证据机器可查:@M_q 路径量化后,全模型参数字节(预期≈1.67G)+ Quantized 模块数
  2. 初始 S₀ 锚点复现:零 S₀ @M vs @M_q,预期差 ≈+9~10%(对齐 epoch0 2.473 vs 2.718)
  3. 补齐 2×2:基线 S₀ @M_q(原缺),四格齐分解"模型内在差"+"S₀ 失配代价"
  4. 救援探针:M_q′(排除 embedding+lm_head),零 S₀ eval@M_q′,对照第2项回答 logits 量化贡献

口径:
  - loss mask 与训练一致(从 prefix_len-1 起,含 stop_token)
  - S₀ 初始 = make_state_params 零初始化(与训练同,seed 无关)
  - 训练后 S₀ 从 state.npz 读回
  - 串行加载,每段 mx.clear_cache() 避免池残留
  - GB(÷1e9),标签二维化
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
from mlx.utils import tree_flatten

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


def _int8_predicate_no_logits(path, module):
    """救援探针:M_q′ 白名单,额外排除 embedding 与 lm_head(logits 路径回 bf16)。"""
    if "lora" in path.lower():
        return False
    # 排除 embedding 和 lm_head
    if "embed" in path.lower() or "head" in path.lower():
        return False
    if isinstance(module, nn.Linear):
        return module.weight.shape[-1] % 64 == 0
    if isinstance(module, nn.Embedding):
        return module.weight.shape[-1] % 64 == 0
    return False


def model_param_bytes(model) -> float:
    """全模型参数字节(GB),用真实 dtype.itemsize。"""
    total = 0
    for _, p in tree_flatten(model.parameters()):
        # uint32 打包的量化 weight:4 个 int8 装一个 uint32,逻辑字节 = size × 4 × 1
        if p.dtype == mx.uint32:
            total += p.size * 4  # 逻辑 int8 字节
        else:
            total += p.size * p.itemsize
    return total / 1e9


def count_quantized_modules(model) -> int:
    """被替换为 Quantized 类的模块数。"""
    n = 0
    for _, mod in model.named_modules():
        if type(mod).__name__.startswith("Quantized"):
            n += 1
    return n


def load_state_npz(path: Path) -> dict:
    z = np.load(path)
    keys = sorted([k for k in z.files if k.startswith("layer_")],
                  key=lambda x: int(x.split("_")[1]))
    return {int(k.split("_")[1]): mx.array(z[k]) for k in keys}


def eval_loss_on(model, samples, states: dict) -> float:
    """200 条逐条前向 masked loss(与训练 _loss_fn 同口径)。"""
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

    result = {"verification": "eval_instrument", "items": {}}

    # 加载数据(需要一个 tok)
    print("加载数据...")
    mdl_tmp, tok = load_model(MODEL, patch=True)
    samples = load_qa_dataset(str(DATA), tok, max_len=512)
    n_samples = len(samples)
    print(f"  {n_samples} 条")
    del mdl_tmp
    mx.clear_cache()

    # 零 S₀(与训练初始一致,seed 无关)
    # 加载一次模型拿 args 造 state
    mdl_tmp, _ = load_model(MODEL, patch=True)
    zero_state = make_state_params(mdl_tmp, dtype=mx.float32)
    del mdl_tmp
    mx.clear_cache()

    # 训练后 S₀
    s0_base = load_state_npz(TC / "15b_s42_fp32" / "state.npz")
    s0_int8 = load_state_npz(TC / "15b_s42_int8" / "state.npz")

    # ════════════════════════════════════════════════════════════════
    # 第1项:量化证据机器可查
    # ════════════════════════════════════════════════════════════════
    print("\n─── 第1项:量化证据 ───")
    mx.clear_cache()
    mdl, _ = load_model(MODEL, patch=True)
    mdl.freeze()
    bytes_before = model_param_bytes(mdl)
    n_modules_before = count_quantized_modules(mdl)
    del mdl
    mx.clear_cache()

    mdl, _ = load_model(MODEL, patch=True)
    mdl.freeze()
    nn.quantize(mdl, group_size=64, bits=8, class_predicate=_int8_predicate)
    bytes_after = model_param_bytes(mdl)
    n_modules_after = count_quantized_modules(mdl)
    del mdl
    mx.clear_cache()

    # 训练侧白名单数量核对(从第1项 spike 报告:146)
    quantized_in_model = n_modules_after - n_modules_before
    print(f"  量化前参数字节: {bytes_before:.4f}G(预期≈3.05)")
    print(f"  量化后参数字节: {bytes_after:.4f}G(预期≈1.67;若≈3.05 即量化未生效)")
    print(f"  Quantized 模块数: {quantized_in_model}(与训练侧白名单 146 核对)")
    item1_ok = (abs(bytes_after - 1.67) < 0.1) and (quantized_in_model == 146)
    print(f"  → {'✅' if item1_ok else '❌'}")

    result["items"]["quant_evidence"] = {
        "bytes_before_gb": round(bytes_before, 4),
        "bytes_after_gb": round(bytes_after, 4),
        "quantized_modules": quantized_in_model,
        "expected_modules": 146,
        "quant_effective": bool(abs(bytes_after - 1.67) < 0.1),
        "pass": bool(item1_ok),
    }

    # ════════════════════════════════════════════════════════════════
    # 第2项:初始 S₀(零)锚点复现:@M vs @M_q
    # ════════════════════════════════════════════════════════════════
    print("\n─── 第2项:零 S₀ 锚点 @M vs @M_q ───")
    mx.clear_cache()
    mdl, _ = load_model(MODEL, patch=True)
    mdl.freeze()
    eval_zero_M = eval_loss_on(mdl, samples, zero_state)
    print(f"  零S₀ @M  = {eval_zero_M:.5f}(对齐 epoch0 基线 2.473)")
    del mdl
    mx.clear_cache()

    mdl, _ = load_model(MODEL, patch=True)
    mdl.freeze()
    nn.quantize(mdl, group_size=64, bits=8, class_predicate=_int8_predicate)
    eval_zero_Mq = eval_loss_on(mdl, samples, zero_state)
    print(f"  零S₀ @M_q= {eval_zero_Mq:.5f}(对齐 epoch0 int8 2.718)")
    del mdl
    mx.clear_cache()

    anchor_diff = (eval_zero_Mq - eval_zero_M) / eval_zero_M * 100
    # epoch0 锚点:2.718 vs 2.473 = +9.91%
    print(f"  差 = {anchor_diff:+.2f}%(预期 ≈+9~10%,epoch0 锚点 +9.91%)")
    item2_ok = 5 < anchor_diff < 15  # 预期区间,宽松
    if not item2_ok:
        print(f"  ⚠️ 差 ≈0 或偏离预期 → 仪器可能有 bug,本单+上份报告 eval 数字作废重跑")
    print(f"  → {'✅' if item2_ok else '❌'}")

    result["items"]["anchor_zero_s0"] = {
        "eval_zero_M": round(eval_zero_M, 5),
        "eval_zero_Mq": round(eval_zero_Mq, 5),
        "diff_pct": round(anchor_diff, 3),
        "epoch0_anchor_expected": 9.91,
        "pass": bool(item2_ok),
    }

    # ════════════════════════════════════════════════════════════════
    # 第3项:补齐 2×2(基线 S₀ @M_q,原缺)
    # ════════════════════════════════════════════════════════════════
    print("\n─── 第3项:补齐 2×2,基线 S₀ @M_q ───")
    mx.clear_cache()
    mdl, _ = load_model(MODEL, patch=True)
    mdl.freeze()
    nn.quantize(mdl, group_size=64, bits=8, class_predicate=_int8_predicate)
    eval_base_Mq = eval_loss_on(mdl, samples, s0_base)
    print(f"  基线S₀ @M_q = {eval_base_Mq:.5f}")
    del mdl
    mx.clear_cache()

    # 四格汇总(前两项 eval 已有,从 eval_result.json 读或重算)
    # 这里用刚算的 + 重算补齐
    print("\n  ═══ 2×2 分解表 ═══")
    print(f"  {'':16} {'@M(全精度)':>12} {'@M_q(量化)':>12} {'失配代价(@M_q−@M)':>18}")
    print(f"  {'零S₀(锚点)':16} {eval_zero_M:>12.5f} {eval_zero_Mq:>12.5f} {eval_zero_Mq-eval_zero_M:>+18.5f}")
    # 训练后 S₀ 的 @M 和 @M_q 从上一份 eval_result.json 读
    prev = json.load(open(TC / "eval_result.json"))
    eval_base_M = prev["eval_base_M"]
    eval_int8_M = prev["eval_int8_M"]
    eval_int8_Mq = prev["diagnostic"]["eval_int8_Mq"]
    print(f"  {'基线S₀(训后)':16} {eval_base_M:>12.5f} {eval_base_Mq:>12.5f} {eval_base_Mq-eval_base_M:>+18.5f}")
    print(f"  {'int8S₀(训后)':16} {eval_int8_M:>12.5f} {eval_int8_Mq:>12.5f} {eval_int8_Mq-eval_int8_M:>+18.5f}")

    result["items"]["matrix_2x2"] = {
        "zero_s0": {"at_M": round(eval_zero_M, 5), "at_Mq": round(eval_zero_Mq, 5)},
        "base_s0": {"at_M": round(eval_base_M, 5), "at_Mq": round(eval_base_Mq, 5)},
        "int8_s0": {"at_M": round(eval_int8_M, 5), "at_Mq": round(eval_int8_Mq, 5)},
    }

    # ════════════════════════════════════════════════════════════════
    # 第4项:救援探针 M_q′(排除 embedding+lm_head),零 S₀
    # ════════════════════════════════════════════════════════════════
    print("\n─── 第4项:救援探针 M_q′(排除 embedding+lm_head)───")
    mx.clear_cache()
    mdl, _ = load_model(MODEL, patch=True)
    mdl.freeze()
    nn.quantize(mdl, group_size=64, bits=8, class_predicate=_int8_predicate_no_logits)
    bytes_mqp = model_param_bytes(mdl)
    n_mqp = count_quantized_modules(mdl)
    eval_zero_Mqp = eval_loss_on(mdl, samples, zero_state)
    print(f"  M_q′ 参数字节: {bytes_mqp:.4f}G(比 M_q 的 {bytes_after:.4f}G 高,embedding+head 回 bf16)")
    print(f"  M_q′ Quantized 数: {n_mqp - 0}(比 M_q 的 {quantized_in_model} 少 2)")
    print(f"  零S₀ @M_q′ = {eval_zero_Mqp:.5f}")
    print(f"  对照 @M_q = {eval_zero_Mq:.5f},差 {eval_zero_Mqp-eval_zero_Mq:+.5f}")
    print(f"  → logits 路径量化贡献了内在差的: {(eval_zero_Mq - eval_zero_Mqp):+.5f}")
    del mdl
    mx.clear_cache()

    # 分解:内在差 = embedding/head 贡献 + 其余 Linear 贡献
    logits_contrib = eval_zero_Mq - eval_zero_Mqp  # M_q 比 M_q′ 多的 = logits 量化带来
    rest_contrib = eval_zero_Mqp - eval_zero_M  # M_q′ 比 M 多的 = 其余 Linear 量化
    total_intrinsic = eval_zero_Mq - eval_zero_M  # 总内在差
    print(f"\n  内在差分解(@M_q − @M = {total_intrinsic:+.5f}):")
    print(f"    logits 路径量化(embedding+head): {logits_contrib:+.5f} ({logits_contrib/total_intrinsic*100:.0f}%)")
    print(f"    其余 Linear 量化:                  {rest_contrib:+.5f} ({rest_contrib/total_intrinsic*100:.0f}%)")

    result["items"]["rescue_probe"] = {
        "eval_zero_Mqp": round(eval_zero_Mqp, 5),
        "Mqp_bytes_gb": round(bytes_mqp, 4),
        "Mqp_modules": n_mqp,
        "logits_quant_contrib": round(logits_contrib, 5),
        "rest_linear_contrib": round(rest_contrib, 5),
        "total_intrinsic": round(total_intrinsic, 5),
    }

    # ════════════════════════════════════════════════════════════════
    out = TC / "eval_verify.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
