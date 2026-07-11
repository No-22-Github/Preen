"""方案 E spike 探针:int8 冻结权重 + fp32 wkv。

**实验代码,不入主干。**

本轮(P2 + P3 + 第1项):
  P2   量化前向十问 A/B:(bf16,fp32) vs (int8,fp32),无 state 裸基座
  P3   梯度余弦:同 batch 下两版 S₀ 梯度的方向余弦(只记录不设阈值)
  第1项 量化范围清单 + 权重内存实测

口径(裁决 1):
  标签二维化 (weight_dtype, wkv_dtype)。基线=(bf16,fp32),方案E=(int8,fp32)。
  不造 fp32 权重基线。内存全 GB(÷1e9)。

规矩:
  - P2 质量异常 → 回退 lm_head 不量化单独记录(§4.1),不许自由调整量化实现
  - P3 只记录不设阈值(预期 ≠ 1,优化对象本就不同)
  - 测什么让数据流过什么:量化侧用量化模型,基线侧用 bf16 模型,不混
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from statetuner.core import (
    load_model, patch_rwkv7_for_train, make_state_params,
    build_state_cache, forward_with_state, generate,
)
from statetuner.templates import NEKO_QA

EXP = ROOT / "experiments" / "mixed_precision"
DATA = EXP / "data"
MODEL_15B = ROOT / "models" / "converted" / "rwkv7-g1g-1.5b"

TEN_QUESTIONS = [
    "早上好呀，宝宝！今天想吃小鱼干吗？",
    "（轻拍肩膀）宝宝今天很乖哦~",
    "今天的天气适合做什么呢？",
    "为什么猫咪会喜欢纸箱？",
    "遇到难过的事情怎么办？",
    "新来的仓管员，长得好像邻居家那只胖橘耶!?",
    "最近有没有偷喝花盆里的雨水?",
    "用猫语说'我爱你'",
    "今晚月亮格外亮晶莹耶不如一起去屋顶数星星怎么样？",
    "周末偷偷带我去公园追蝴蝶嘛~",
]


def quant_predicate(path: str, module: nn.Module, skip_head: bool = False) -> bool:
    """class_predicate:量化白名单(Linear + Embedding),排除 LoRA + 不可整除层。

    skip_head=True 时跳过 lm_head(§4.1 回退排查用)。
    """
    # 跳过 LoRA 低秩投影
    if "lora" in path.lower():
        return False
    if isinstance(module, nn.Linear):
        if skip_head and ("head" in path.lower() or "lm_head" in path.lower()):
            return False
        # 维度必须被 group_size 整除
        if module.weight.shape[-1] % 64 != 0:
            return False
        return True
    if isinstance(module, nn.Embedding):
        return module.weight.shape[-1] % 64 == 0
    return False


def weight_memory_gb(model) -> dict:
    """用真实 dtype.itemsize 算权重内存(GB 口径)。

    MLX int8 量化:weight 存成 uint32(4个int8打包),itemsize=4,
    但逻辑权重数 = size × 4,每权重 1 byte → 逻辑字节 = size × 4 × 1 = size × itemsize。
    scales/biases 是 bf16(itemsize=2)。
    """
    weight_logical = 0
    weight_bytes = 0
    scales_bytes = 0
    biases_bytes = 0
    other_bytes = 0
    other_params = 0
    for k, p in tree_flatten(model.parameters()):
        if "scales" in k:
            scales_bytes += p.size * p.itemsize
        elif "biases" in k:
            biases_bytes += p.size * p.itemsize
        elif k.endswith(".weight") and p.dtype == mx.uint32:
            # uint32 打包 4 个 int8:逻辑权重 = size × 4
            weight_logical += p.size * 4
            weight_bytes += p.size * 4  # 每 int8 = 1 byte
        else:
            other_bytes += p.size * p.itemsize
            other_params += p.size
    total = weight_bytes + scales_bytes + biases_bytes + other_bytes
    return {
        "weight_logical_params": weight_logical,
        "weight_gb": round(weight_bytes / 1e9, 4),
        "scales_gb": round(scales_bytes / 1e9, 4),
        "biases_gb": round(biases_bytes / 1e9, 4),
        "other_gb": round(other_bytes / 1e9, 4),
        "other_params": other_params,
        "total_gb": round(total / 1e9, 4),
    }


def run_decode(mdl, tok, label: str) -> list[dict]:
    """无 state 裸基座十问贪心解码。"""
    results = []
    for q in TEN_QUESTIONS:
        prompt = NEKO_QA.format_prefix(q=q)
        out = generate(mdl, tok, prompt, state=None, max_tokens=120)
        # 断言无 nan/inf(decode 出来的字符串层面无法直接判断,看 token logits)
        results.append({"q": q, "out": out})
        print(f"  [{label}] {q[:20]}: {out[:60]}{'...' if len(out)>60 else ''}")
    return results


def check_finite(outputs: list[dict]) -> bool:
    """字符串层面检查:无 nan/inf 字面量泄漏到 decode。"""
    for r in outputs:
        o = r["out"]
        if "nan" in o.lower() and ("inf" in o.lower() or float("nan") != float("nan")):
            return False
    return True


# ────────────────────────── P2:十问 A/B ──────────────────────────

def run_p2(skip_head: bool = False) -> dict:
    print("=" * 70)
    print("P2: 量化前向十问 A/B")
    print(f"  skip_head={skip_head}")
    print("=" * 70)

    patch_rwkv7_for_train()

    # A:基线 (bf16, fp32) —— 量化前
    print("\n── A: (bf16, fp32) 基线 ──")
    mdl_a, tok = load_model(MODEL_15B, patch=True)
    mdl_a.freeze()
    out_a = run_decode(mdl_a, tok, "A-bf16")

    # 释放 A,避免 Metal 内存池累积(report_c4g 教训:连续 load_model 叠峰)
    del mdl_a
    mx.metal.clear_cache()

    # B:(int8, fp32) —— 量化后
    print("\n── B: (int8, fp32) 量化 ──")
    mdl_b, tok = load_model(MODEL_15B, patch=True)
    mdl_b.freeze()
    nn.quantize(
        mdl_b, group_size=64, bits=8,
        class_predicate=lambda path, mod: quant_predicate(path, mod, skip_head=skip_head),
    )
    out_b = run_decode(mdl_b, tok, "B-int8")

    finite = check_finite(out_b)
    print(f"\n>>> P2 无 nan/inf 字面泄漏: {'✅' if finite else '❌'}")

    return {
        "baseline": {"weight_dtype": "bf16", "wkv_dtype": "fp32", "outputs": out_a},
        "int8": {"weight_dtype": "int8", "wkv_dtype": "fp32", "outputs": out_b, "skip_head": skip_head},
        "no_nan_inf_literal": finite,
    }


# ────────────────────────── P3:梯度余弦 ──────────────────────────

def run_p3() -> dict:
    print("\n" + "=" * 70)
    print("P3: 梯度方向余弦 (bf16,fp32) vs (int8,fp32),只记录不设阈值")
    print("=" * 70)

    patch_rwkv7_for_train()
    input_ids = mx.array([[1, 100, 200, 300, 400, 500]])  # 固定 batch

    # 固定 S₀ 初始(非零,让梯度有信号)
    mx.random.seed(42)
    S0_init = mx.random.normal((32, 64, 64)) * 0.1

    def grad_S0_on_model(quantize: bool) -> mx.array:
        mdl, tok = load_model(MODEL_15B, patch=True)
        mdl.freeze()
        if quantize:
            nn.quantize(
                mdl, group_size=64, bits=8,
                class_predicate=lambda p, m: quant_predicate(p, m, skip_head=False),
            )
        states = make_state_params(mdl, dtype=mx.float32)
        for i in states:
            states[i] = S0_init  # 同一初始

        def loss_fn(s0):
            states[0] = s0
            caches = build_state_cache(states, batch_size=1)
            logits = forward_with_state(mdl, input_ids, states, batch_size=1)
            return mx.sum(logits)

        loss, grad = mx.value_and_grad(loss_fn)(states[0])
        del mdl
        mx.metal.clear_cache()
        return loss, grad

    print("\n── (bf16, fp32) 求 S₀[0] 梯度 ──")
    loss_bf, grad_bf = grad_S0_on_model(quantize=False)
    print(f"  loss={float(loss_bf):.2f} finite={bool(mx.all(mx.isfinite(loss_bf)))}")
    print(f"  grad nonzero={bool(mx.any(grad_bf!=0))} finite={bool(mx.all(mx.isfinite(grad_bf)))}")

    print("\n── (int8, fp32) 求 S₀[0] 梯度 ──")
    loss_i8, grad_i8 = grad_S0_on_model(quantize=True)
    print(f"  loss={float(loss_i8):.2f} finite={bool(mx.all(mx.isfinite(loss_i8)))}")
    print(f"  grad nonzero={bool(mx.any(grad_i8!=0))} finite={bool(mx.all(mx.isfinite(grad_i8)))}")

    a = grad_bf.astype(mx.float32).reshape(-1)
    b = grad_i8.astype(mx.float32).reshape(-1)
    na = mx.sqrt(mx.sum(a * a))
    nb = mx.sqrt(mx.sum(b * b))
    cos = float(mx.sum(a * b) / (na * nb + 1e-12))
    print(f"\n>>> P3 S₀[0] 梯度余弦: {cos:.4f} (只记录,不设阈值)")

    return {
        "cosine_s0_layer0": round(cos, 6),
        "bf16_loss": float(loss_bf),
        "int8_loss": float(loss_i8),
        "bf16_grad_nonzero": bool(mx.any(grad_bf != 0)),
        "bf16_grad_finite": bool(mx.all(mx.isfinite(grad_bf))),
        "int8_grad_nonzero": bool(mx.any(grad_i8 != 0)),
        "int8_grad_finite": bool(mx.all(mx.isfinite(grad_i8))),
        "note": "预期 ≠ 1:优化对象是不同模型(bf16 vs int8 权重),梯度方向刻画数据不是判据",
    }


# ────────────────────────── 第1项:量化范围清单 ──────────────────────────

def run_quant_inventory(skip_head: bool = False) -> dict:
    print("\n" + "=" * 70)
    print("第1项:量化范围清单 + 权重内存实测")
    print("=" * 70)

    patch_rwkv7_for_train()
    mdl, tok = load_model(MODEL_15B, patch=True)
    mdl.freeze()

    # 量化前内存
    mem_before = weight_memory_gb(mdl)

    # 枚举所有 Linear/Embedding,分类(用 named_modules)
    quantized = []
    skipped_lora = []
    skipped_indivisible = []

    for path, mod in mdl.named_modules():
        if isinstance(mod, nn.Linear):
            wshape = tuple(mod.weight.shape)
            is_lora = "lora" in path.lower()
            indivisible = wshape[-1] % 64 != 0
            if is_lora:
                skipped_lora.append({"path": path, "shape": wshape, "params": mod.weight.size})
            elif indivisible:
                skipped_indivisible.append({"path": path, "shape": wshape, "params": mod.weight.size})
            else:
                quantized.append({"path": path, "shape": wshape, "params": mod.weight.size})
        elif isinstance(mod, nn.Embedding):
            wshape = tuple(mod.weight.shape)
            if wshape[-1] % 64 == 0:
                quantized.append({"path": path, "shape": wshape, "params": mod.weight.size})
            else:
                skipped_indivisible.append({"path": path, "shape": wshape, "params": mod.weight.size})

    # 量化
    nn.quantize(
        mdl, group_size=64, bits=8,
        class_predicate=lambda p, m: quant_predicate(p, m, skip_head=skip_head),
    )
    mem_after = weight_memory_gb(mdl)

    saved_gb = round(mem_before["total_gb"] - mem_after["total_gb"], 4)
    saved_pct = round((1 - mem_after["total_gb"] / mem_before["total_gb"]) * 100, 1)

    print(f"\n量化层({len(quantized)} 个):")
    for q in quantized[:8]:
        print(f"  {q['path']}: {q['shape']} ({q['params']/1e6:.1f}M)")
    if len(quantized) > 8:
        print(f"  ... 共 {len(quantized)} 个")
    print(f"\n跳过-LoRA({len(skipped_lora)} 个):")
    for s in skipped_lora[:4]:
        print(f"  {s['path']}: {s['shape']}")
    print(f"\n跳过-不可整除({len(skipped_indivisible)} 个):")
    for s in skipped_indivisible[:4]:
        print(f"  {s['path']}: {s['shape']}")

    print(f"\n内存对照:")
    print(f"  (bf16,fp32): {mem_before['total_gb']} GB")
    print(f"  (int8,fp32): {mem_after['total_gb']} GB")
    print(f"  节省:        {saved_gb} GB ({saved_pct}%)")

    return {
        "quantized_layers": quantized,
        "skipped_lora": skipped_lora,
        "skipped_indivisible": skipped_indivisible,
        "n_quantized": len(quantized),
        "n_skipped": len(skipped_lora) + len(skipped_indivisible),
        "mem_before": mem_before,
        "mem_after": mem_after,
        "saved_gb": saved_gb,
        "saved_pct": saved_pct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-head", action="store_true",
                    help="P2 质量异常时回退:跳过 lm_head 量化(§4.1 排查顺位)")
    ap.add_argument("--only", choices=["p2", "p3", "inv", "all"], default="all")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "int8_spike_result.json"
    # 累积模式:若文件已存在,读回合并(--only 分项跑时不丢前序结果)
    if out.exists() and args.only != "all":
        result = json.loads(out.read_text())
    else:
        result = {"spike": "int8_e", "commit": None}

    if args.only in ("p2", "all"):
        result["p2"] = run_p2(skip_head=args.skip_head)
    if args.only in ("p3", "all"):
        result["p3"] = run_p3()
    if args.only in ("inv", "all"):
        result["inventory"] = run_quant_inventory(skip_head=args.skip_head)

    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
