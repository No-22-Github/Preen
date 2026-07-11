"""dump 各层 w 衰减系数的**真实分布**(fp32 路径,非 bf16 量化产物)。

⚠️ 镜像坑教训(本脚本的来由):
  模型权重是 bf16,w_lora 投影输出经 exp(-0.6065*sigmoid(...)) 后 .astype(bf16) cast 回 bf16。
  直接 hook _wkv7 拿到的 w 是 bf16——bf16 在 1.0 附近 ulp=2^-8=0.0039,把 0.9997 舍成 1.0,
  制造"p95=1.0"的虚假最坏情况。要测真实分布必须走 fp32 路径。

方法:patch ops 路径(可微,接受 fp32)→ 模型 set_dtype(fp32)→ hook 拿 fp32 的 w。
  (kernel 路径 patch=False 不行,wkv7 kernel 要 bf16 输入会报错)

报口径(用户指定):
  1. 1-w(遗忘率)分布:w 贴近 1 时只有看遗忘率本身才有分辨率
  2. 落进 bf16 舍入盲区(w>~0.9990,会被 bf16 舍成 1.0)的通道占比,分层列出
  bf16 在 [0.5,1.0] 区间 ulp=2^-8=0.00390625,可表示值:1.0, 0.99609, 0.99219...
  所以 w>0.99805 会被舍成 1.0 或 0.99609(盲区)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from statetuner.core import load_model, patch_rwkv7_for_train
from statetuner.data import load_qa_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="train_data/NekoQA_10k/nekoqa_smoke_200.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=5)
    args = ap.parse_args()

    # ★ fp32 路径:patch=True(ops,接受 fp32)+ set_dtype(fp32)
    patch_rwkv7_for_train()
    mdl, tok = load_model(args.model, patch=True)
    mdl.freeze()
    mdl.set_dtype(mx.float32)  # 关键:权重 cast fp32,w 投影输出才是 fp32

    samples = load_qa_dataset(args.data, tok, max_len=512)
    samples.sort(key=lambda s: -s.length)
    s = samples[0]
    inp = mx.array([s.input_ids])
    print(f"样本 token 数: {s.length}", file=sys.stderr, flush=True)

    ATM = type(mdl.layers[0]["attn"])
    orig = ATM._wkv7
    all_w = {}

    def hook(self, r, w, k, v, a, b, state):
        li = getattr(self, "layer_idx", None)
        if li is not None and li not in all_w:
            all_w[li] = np.array(w.astype(mx.float32))  # fp32 路径,w 本就是 fp32
        return orig(self, r, w, k, v, a, b, state)

    ATM._wkv7 = hook
    try:
        _ = mdl(inp, mdl.make_cache())
    finally:
        ATM._wkv7 = orig

    # 验证:第一个 hook 的 w dtype 确实是 fp32(确认不是 bf16 量化产物)
    li0 = sorted(all_w)[0]
    assert all_w[li0].dtype == np.float32, f"w 不是 fp32!是 {all_w[li0].dtype},镜像坑未修"
    print(f"✓ 确认 fp32 路径(w 真实分布,非 bf16 量化产物)", file=sys.stderr)

    per_layer = {}
    for li in sorted(all_w):
        wf = all_w[li].flatten()
        forget = 1.0 - wf  # 遗忘率
        per_layer[li] = {
            "w": {
                "p1": round(float(np.percentile(wf, 1)), 6),
                "p50": round(float(np.percentile(wf, 50)), 6),
                "p95": round(float(np.percentile(wf, 95)), 6),
                "p99": round(float(np.percentile(wf, 99)), 6),
                "max": round(float(wf.max()), 6),
            },
            "forget_rate_1_minus_w": {
                # 遗忘率分位数(scientific),w 贴近1 时这里才有分辨率
                "min": float(f"{forget.min():.2e}"),
                "p5": float(f"{np.percentile(forget, 5):.2e}"),
                "p50": float(f"{np.percentile(forget, 50):.2e}"),
                "p95": float(f"{np.percentile(forget, 95):.2e}"),
            },
            "bf16_blind_frac": {
                # 落进 bf16 舍入盲区的通道占比(w 会被 bf16 舍成 1.0 或 0.99609)
                "w_gt_0p996": round(float(np.mean(wf > 0.996)), 4),
                "w_gt_0p998": round(float(np.mean(wf > 0.998)), 4),
                "w_gt_0p999": round(float(np.mean(wf > 0.999)), 4),
            },
        }
        if li % 6 == 0 or li == len(all_w) - 1:
            print(f"  layer{li}: w_p95={per_layer[li]['w']['p95']} "
                  f"1-w_p5={per_layer[li]['forget_rate_1_minus_w']['p5']:.2e} "
                  f"bf16盲区(>0.999)={per_layer[li]['bf16_blind_frac']['w_gt_0p999']*100:.1f}%",
                  file=sys.stderr, flush=True)

    # 全层汇总
    all_wf = np.concatenate([all_w[li].flatten() for li in sorted(all_w)])
    out = {
        "model": args.model, "sample_len": s.length, "n_layers": len(all_w),
        "method": "fp32 path (ops patch + set_dtype fp32), NOT bf16 quantized",
        "w_param": "exp(-0.606531 * sigmoid(w_lora(xw)))",
        "summary": {
            "w_p95_all": round(float(np.percentile(all_wf, 95)), 6),
            "w_p99_all": round(float(np.percentile(all_wf, 99)), 6),
            "forget_p5_all": float(f"{np.percentile(1-all_wf, 5):.2e}"),
            "bf16_blind_gt_0p999_all": round(float(np.mean(all_wf > 0.999)), 4),
        },
        "per_layer": per_layer,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n写入 {args.out}", file=sys.stderr)
    print(json.dumps({"summary": out["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
