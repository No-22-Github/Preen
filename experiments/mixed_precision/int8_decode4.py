"""四方对照本机推理:四组 state 挂同一全精度模型 M,跑固定十问。

口径(与 Runner 真机一致):
  - 四组 state 都挂全精度 bf16 模型 M(patch=False kernel 路径)
  - 这样纯粹比 state 质量,不含量化失配
  - int8 训出的 state 挂 M = 模拟 Runner 真机(第5项口径)

客观指标:循环检测、自发终止、长度
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import mlx.core as mx
from statetuner.core import load_model, generate
from statetuner.templates import NEKO_QA

MODEL = ROOT / "models" / "converted" / "rwkv7-g1g-1.5b"
TC = ROOT / "experiments" / "mixed_precision" / "data" / "int8_traincompare"

GROUPS = [
    ("baseline_2ep", TC / "15b_s42_fp32" / "state.npz"),
    ("baseline_4ep", TC / "15b_s42_fp32_resume_fp32" / "state.npz"),
    ("int8_2ep", TC / "15b_s42_int8" / "state.npz"),
    ("int8_4ep", TC / "15b_s42_int8_resume" / "state.npz"),
]

QUESTIONS = [
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


def is_circular(text, window=12, min_repeat=3):
    if len(text) < window * min_repeat:
        return False
    tail = text[-(window * min_repeat):]
    seg = tail[:window]
    return all(tail[i*window:(i+1)*window] == seg for i in range(min_repeat))


def main():
    # 加载一次模型,四组共用(都是挂 M)
    print("loading model M (全精度, patch=False kernel 路径)...")
    mdl, tok = load_model(MODEL, patch=False)

    all_results = {}
    for label, state_path in GROUPS:
        print(f"\n{'='*70}")
        print(f"  {label}  ({state_path.name})")
        print(f"{'='*70}")
        results = []
        for qi, q in enumerate(QUESTIONS):
            prompt = NEKO_QA.format_prefix(q=q)
            out = generate(mdl, tok, prompt, state=str(state_path), max_tokens=120)
            circ = is_circular(out)
            early = len(out) < 240
            results.append({
                "q": q, "out": out,
                "circular": circ, "early_stop": early, "len": len(out),
            })
            tag = "🔴circ" if circ else ("⚡stop" if early and len(out) < 30 else "")
            print(f"  [{qi+1:2d}] {q[:22]}")
            print(f"       {out[:80]}{'...' if len(out)>80 else ''}  ({len(out)}字 {tag})")

        n_circ = sum(1 for r in results if r["circular"])
        n_stop = sum(1 for r in results if r["early_stop"])
        avg_len = sum(r["len"] for r in results) / len(results)
        print(f"\n  汇总: 循环={n_circ}/10  终止={n_stop}/10  均长={avg_len:.0f}字")
        all_results[label] = {
            "results": results, "n_circular": n_circ,
            "n_early_stop": n_stop, "avg_len": round(avg_len, 1),
        }

    # 四方对照表
    print(f"\n{'='*70}")
    print("四方对照汇总(都挂全精度 M)")
    print(f"{'='*70}")
    print(f"{'组':>14} {'循环':>6} {'终止':>6} {'均长':>6}")
    for label, _ in GROUPS:
        r = all_results[label]
        print(f"{label:>14} {r['n_circular']:>3}/10 {r['n_early_stop']:>3}/10 {r['avg_len']:>5.0f}字")

    out = TC / "decode4_compare.json"
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
