"""探针②:梯度体检(纯前反向 + 前 200 步重跑采样)。

子项:
  A. 零 S₀ 处,真实 NekoQA 样本,int8 vs 基线 S₀ 梯度:全局范数比、逐层范数比、逐层余弦
  B. int8 组 ep0 末(step 200)checkpoint 处,重复 A
  C. 前 20 步逐步 S₀ 全局梯度范数,两组对比(从同一重跑里采)

口径:
  - 样本:真实 NekoQA 样本(非 P3 的 6 token 合成),取第一条
  - S₀ 初始:零(make_state_params,与训练一致)
  - 梯度对整个 states dict 求(所有 24 层),不只 layer0
  - 逐层范数比 = ‖grad_int8_i‖ / ‖grad_base_i‖
  - 逐层余弦 = cos(grad_base_i, grad_int8_i)
  - 重跑前 200 步拿 ep0 末 checkpoint:B 在 step 200 采样,C 在 step 0~19 逐步采样
  - 串行加载,每段清池

预测登记(炸伤说):范数比显著异常(整体 >2× 或个别层爆表)
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
import mlx.optimizers as optim

from statetuner.core import (
    load_model, patch_rwkv7_for_train, make_state_params, forward_with_state,
)
from statetuner.data import load_qa_dataset
from statetuner.train import TrainConfig, _to_mx_batch, cosine_lr

MODEL = ROOT / "models" / "converted" / "rwkv7-g1g-1.5b"
DATA = ROOT / "train_data" / "NekoQA_10k" / "nekoqa_smoke_200.json"
OUT = ROOT / "experiments" / "mixed_precision" / "data" / "int8_traincompare"


def _int8_predicate(path, module):
    if "lora" in path.lower():
        return False
    if isinstance(module, nn.Linear):
        return module.weight.shape[-1] % 64 == 0
    if isinstance(module, nn.Embedding):
        return module.weight.shape[-1] % 64 == 0
    return False


def grad_norm(v):
    return float(mx.sqrt(mx.sum(v.astype(mx.float32) ** 2)))


def grad_cos(a, b):
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    na = mx.sqrt(mx.sum(a * a))
    nb = mx.sqrt(mx.sum(b * b))
    return float(mx.sum(a * b) / (na * nb + 1e-12))


def compute_grad_on_model(model, inp, lab, msk, states):
    """对整个 states dict 求 grad,返回 {layer: grad_array}。"""
    B = inp.shape[0]

    def loss_fn(sd):
        logits = forward_with_state(model, inp, sd, B)
        lp = nn.log_softmax(logits, -1)
        g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
        return (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)

    loss, grads = mx.value_and_grad(loss_fn)(states)
    mx.eval(grads)
    return float(loss), grads


def per_layer_report(grads_a, grads_b, label_a="base", label_b="int8"):
    """逐层范数比 + 余弦,返回 dict。"""
    layers = sorted(grads_a.keys())
    report = {"layers": [], "global_norm_ratio": None, "global_cos": None}
    # 全局(所有层拼接)
    ga_all = mx.concatenate([grads_a[i].astype(mx.float32).reshape(-1) for i in layers])
    gb_all = mx.concatenate([grads_b[i].astype(mx.float32).reshape(-1) for i in layers])
    na = grad_norm(ga_all)
    nb = grad_norm(gb_all)
    report["global_norm_ratio"] = round(nb / na, 4) if na > 0 else None
    report["global_cos"] = round(grad_cos(ga_all, gb_all), 4)
    report[f"norm_{label_a}"] = round(na, 4)
    report[f"norm_{label_b}"] = round(nb, 4)

    ratios = []
    for i in layers:
        na_i = grad_norm(grads_a[i])
        nb_i = grad_norm(grads_b[i])
        ratio = nb_i / na_i if na_i > 0 else 0
        cos_i = grad_cos(grads_a[i], grads_b[i])
        ratios.append(ratio)
        report["layers"].append({
            "layer": i, "norm_ratio": round(ratio, 4),
            "cos": round(cos_i, 4),
            f"norm_{label_a}": round(na_i, 5),
            f"norm_{label_b}": round(nb_i, 5),
        })
    report["max_layer_ratio"] = round(max(ratios), 4)
    report["min_layer_ratio"] = round(min(ratios), 4)
    report["layers_with_ratio_gt2"] = sum(1 for r in ratios if r > 2.0)
    return report


def load_model_int8():
    mdl, tok = load_model(MODEL, patch=True)
    mdl.freeze()
    nn.quantize(mdl, group_size=64, bits=8, class_predicate=_int8_predicate)
    return mdl, tok


def load_model_base():
    mdl, tok = load_model(MODEL, patch=True)
    mdl.freeze()
    return mdl, tok


def main():
    patch_rwkv7_for_train()
    result = {"probe": "grad_check", "p3_original_conditions": {
        "state": "mx.random.normal((32,64,64))*0.1 seed=42 (非零随机,非零S₀非训后)",
        "sample": "mx.array([[1,100,200,300,400,500]]) (6 token 合成,非真实 NekoQA)",
        "layer": "仅 layer 0",
        "note": "P3 的 0.9462 不能外推到真实训练条件(零S₀+真实样本+全层)",
    }}

    # ════════════════════════════════════════════════════════════════
    # 子项 A:零 S₀,真实样本,逐层梯度
    # ════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("②-A: 零 S₀,真实 NekoQA 样本,int8 vs 基线 逐层梯度")
    print("=" * 70)

    # 用第一条真实样本(与训练同口径)
    mdl_base, tok = load_model_base()
    samples = load_qa_dataset(str(DATA), tok, max_len=512)
    sample = samples[0]
    inp0, lab0, msk0 = _to_mx_batch(sample)
    print(f"  样本: len={sample.length}, input shape={inp0.shape}")
    del mdl_base
    mx.clear_cache()

    # 基线梯度
    mdl_base, _ = load_model_base()
    zero_state_b = make_state_params(mdl_base, dtype=mx.float32)
    loss_b, grad_b = compute_grad_on_model(mdl_base, inp0, lab0, msk0, zero_state_b)
    print(f"  基线: loss={loss_b:.5f}")
    del mdl_base
    mx.clear_cache()

    # int8 梯度
    mdl_i8, _ = load_model_int8()
    zero_state_i = make_state_params(mdl_i8, dtype=mx.float32)
    loss_i, grad_i = compute_grad_on_model(mdl_i8, inp0, lab0, msk0, zero_state_i)
    print(f"  int8: loss={loss_i:.5f}")
    del mdl_i8
    mx.clear_cache()

    report_a = per_layer_report(grad_b, grad_i, "base", "int8")
    print(f"\n  全局范数比(int8/基线) = {report_a['global_norm_ratio']}")
    print(f"  全局余弦 = {report_a['global_cos']}")
    print(f"  逐层范数比范围: {report_a['min_layer_ratio']} ~ {report_a['max_layer_ratio']}")
    print(f"  范数比>2 的层数: {report_a['layers_with_ratio_gt2']}")
    # 打印最极端的 5 层
    extreme = sorted(report_a["layers"], key=lambda x: max(x["norm_ratio"], 1/x["norm_ratio"]), reverse=True)[:5]
    print(f"  最极端 5 层:")
    for l in extreme:
        print(f"    layer{l['layer']:2d}: ratio={l['norm_ratio']:.3f} cos={l['cos']:.4f} "
              f"(norm base={l['norm_base']:.5f} int8={l['norm_int8']:.5f})")
    result["A_zero_s0"] = report_a

    # ════════════════════════════════════════════════════════════════
    # 子项 C:前 20 步逐步全局梯度范数(两组,从重跑采)
    # ════════════════════════════════════════════════════════════════
    # 重跑 200 步:在 step 0~19 逐步采全局范数(C),在 step 200 采逐层(B)
    print("\n" + "=" * 70)
    print("②-B+C: 重跑前 200 步(逐层 step200 + 逐步范数 step0~19)")
    print("=" * 70)

    cfg = TrainConfig(lr=0.01, lr_floor=1e-4, warmup=10, ctx_len=512,
                      epochs=1, grad_clip=1.0, early_stop=False, seed=42)
    total = cfg.total_steps(len(samples))

    def run_prefix(precision_label, quantize):
        """跑前 200 步,返回 (step0~19 全局范数列表, step200 states, step200 逐层梯度)。"""
        mdl, _ = (load_model_int8() if quantize else load_model_base())
        states = make_state_params(mdl, dtype=mx.float32)
        opt = optim.Adam(learning_rate=0.01, betas=[0.9, 0.99], eps=1e-8)
        import random
        order = list(range(len(samples)))
        random.Random(42).shuffle(order)

        step_norms = []  # step0~19 的全局梯度范数(clip 前)
        for step in range(200):
            batch = _to_mx_batch(samples[order[step % len(order)]])
            inp, lab, msk = batch
            B = inp.shape[0]

            def _loss(sd, inp=inp, lab=lab, msk=msk, B=B):
                logits = forward_with_state(mdl, inp, sd, B)
                lp = nn.log_softmax(logits, -1)
                g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
                return (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)

            opt.learning_rate = cosine_lr(step, total, cfg)
            loss, grads = mx.value_and_grad(_loss)(states)
            # clip 前的全局范数
            gn = grad_norm(mx.concatenate([grads[i].astype(mx.float32).reshape(-1)
                                           for i in sorted(grads)]))
            if step < 20:
                step_norms.append({"step": step, "grad_norm": round(gn, 5),
                                   "loss": round(float(loss), 5)})
            grads = {k: mx.clip(g, -cfg.grad_clip, cfg.grad_clip) for k, g in grads.items()}
            states = opt.apply_gradients(grads, states)
            mx.eval(states, loss)
            if step % 50 == 0:
                print(f"    [{precision_label}] step{step:3d} loss={float(loss):.4f} grad_norm={gn:.4f}")

        # step200 逐层梯度(用第一条样本,与 A 同口径)
        loss_200, grad_200 = compute_grad_on_model(mdl, inp0, lab0, msk0, states)
        print(f"    [{precision_label}] step200 loss={loss_200:.5f} (eval on sample0)")
        del mdl
        mx.clear_cache()
        return step_norms, states, grad_200, loss_200

    print("\n  -- 基线 重跑前200步 --")
    norms_b, states_b_200, grad_b_200, loss_b_200 = run_prefix("base", quantize=False)
    print("\n  -- int8 重跑前200步 --")
    norms_i, states_i_200, grad_i_200, loss_i_200 = run_prefix("int8", quantize=True)

    # C:逐步范数对比
    print("\n  --- ②-C 前20步逐步全局梯度范数 ---")
    print(f"  {'step':>4} {'基线_norm':>10} {'int8_norm':>10} {'比值':>8}")
    for nb, ni in zip(norms_b, norms_i):
        ratio = ni["grad_norm"] / nb["grad_norm"] if nb["grad_norm"] > 0 else 0
        print(f"  {nb['step']:>4} {nb['grad_norm']:>10.5f} {ni['grad_norm']:>10.5f} {ratio:>8.3f}")
    result["C_step_norms"] = {"base": norms_b, "int8": norms_i}

    # B:ep0 末(step200)逐层梯度
    print("\n  --- ②-B ep0末(step200)逐层梯度 ---")
    report_b = per_layer_report(grad_b_200, grad_i_200, "base", "int8")
    print(f"  全局范数比(int8/基线) = {report_b['global_norm_ratio']}")
    print(f"  全局余弦 = {report_b['global_cos']}")
    print(f"  逐层范数比范围: {report_b['min_layer_ratio']} ~ {report_b['max_layer_ratio']}")
    print(f"  范数比>2 的层数: {report_b['layers_with_ratio_gt2']}")
    extreme_b = sorted(report_b["layers"], key=lambda x: max(x["norm_ratio"], 1/x["norm_ratio"]), reverse=True)[:5]
    print(f"  最极端 5 层:")
    for l in extreme_b:
        print(f"    layer{l['layer']:2d}: ratio={l['norm_ratio']:.3f} cos={l['cos']:.4f}")
    result["B_ep0_end"] = report_b

    # 写出
    out = OUT / "grad_probe.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
