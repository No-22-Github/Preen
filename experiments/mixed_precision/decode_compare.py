"""任务2:固定十问贪心解码 A/B + state 张量距离。

输入:两个 state.npz(fp32 版 + bf16 版)。
输出:
  1. 十问贪心解码并排(fp32 state vs bf16 state),逐问并排输出
  2. 客观差异标注(是否循环、是否自发终止)—— 风格/口癖留裁决
  3. 两版 state 逐层 std 对比
  4. 补丁判据:两 state 张量的余弦相似度 + 相对 L2 距离(逐层 + 整体)

推理走 kernel 路径(load_model patch=False),与 cli.py preview 一致。
两个 state 推理路径完全相同,只 state 来源不同,保证 A/B 公平。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_ten_questions(path: str) -> list[str]:
    """读 ten_questions.txt,返回纯问句列表(跳过 REF 行和注释空行)。"""
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("REF"):
                continue
            qs.append(line)
    return qs


def load_state_npz(path: str) -> dict[int, np.ndarray]:
    """加载 npz → {layer: ndarray(fp32)}。"""
    data = np.load(path)
    return {i: np.array(data[f"layer_{i}"]).astype(np.float32) for i in range(len(data.files))}


def state_metrics(s_fp: np.ndarray, s_bf: np.ndarray) -> dict:
    """单个 state 张量(展平)的距离:余弦相似度 + 相对 L2。"""
    a = s_fp.flatten()
    b = s_bf.flatten()
    # 余弦相似度
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    # 相对 L2:||a-b|| / ||a||
    rel_l2 = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
    return {"cosine": round(cos, 5), "rel_l2": round(rel_l2, 5)}


def is_circular(text: str, window: int = 12, min_repeat: int = 3) -> bool:
    """简单循环检测:末尾 window 字符重复 min_repeat 次。"""
    if len(text) < window * min_repeat:
        return False
    tail = text[-(window * min_repeat):]
    seg = tail[:window]
    return all(tail[i * window:(i + 1) * window] == seg for i in range(min_repeat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fp32-state", required=True, help="fp32 版 state.npz")
    ap.add_argument("--bf16-state", required=True, help="bf16 版 state.npz")
    ap.add_argument("--questions", default="experiments/mixed_precision/ten_questions.txt")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    args = ap.parse_args()

    qs = parse_ten_questions(args.questions)
    s_fp = load_state_npz(args.fp32_state)
    s_bf = load_state_npz(args.bf16_state)
    assert set(s_fp.keys()) == set(s_bf.keys()), "两版 state 层数不一致"

    # ── state 张量距离(逐层 + 整体) ──
    layers = sorted(s_fp.keys())
    per_layer = {}
    for i in layers:
        m = state_metrics(s_fp[i], s_bf[i])
        per_layer[i] = m
    # 整体(所有层拼接)
    all_fp = np.concatenate([s_fp[i].flatten() for i in layers])
    all_bf = np.concatenate([s_bf[i].flatten() for i in layers])
    overall = state_metrics(all_fp, all_bf)

    # ── 逐层 std ──
    std_fp = {i: round(float(s_fp[i].std()), 5) for i in layers}
    std_bf = {i: round(float(s_bf[i].std()), 5) for i in layers}
    std_diff = {i: round(std_bf[i] - std_fp[i], 5) for i in layers}

    # ── 十问贪心解码 ──
    from statetuner.core import generate, load_model
    from statetuner.templates import NEKO_QA

    print(f"# 加载模型 {args.model} (kernel 路径, 推理用)", file=sys.stderr, flush=True)
    mdl, tok = load_model(args.model, patch=False)

    decode_results = []
    print("\n" + "=" * 70)
    print("# 固定十问贪心解码 A/B(fp32 state vs bf16 state)")
    print("=" * 70)
    for idx, q in enumerate(qs):
        prompt = NEKO_QA.format_prefix(q=q)
        out_fp = generate(mdl, tok, prompt, state=s_fp, max_tokens=args.max_tokens)
        out_bf = generate(mdl, tok, prompt, state=s_bf, max_tokens=args.max_tokens)

        # 客观差异标注
        fp_circ = is_circular(out_fp)
        bf_circ = is_circular(out_bf)
        # 自发终止:max_tokens 内停(没跑到上限)
        fp_stopped = len(out_fp) < args.max_tokens * 2  # 粗略:decode 后字符数 < max_tokens*2 视为早停
        bf_stopped = len(out_bf) < args.max_tokens * 2
        notes = []
        if fp_circ or bf_circ:
            notes.append(f"循环(fp32={fp_circ}, bf16={bf_circ})")
        if fp_stopped != bf_stopped:
            notes.append(f"终止行为不同(fp32早停={fp_stopped}, bf16早停={bf_stopped})")
        note_str = " | ".join(notes) if notes else "无客观异常"

        print(f"\n[{idx + 1}] {q}")
        print(f"  fp32: {out_fp[:200]}")
        print(f"  bf16: {out_bf[:200]}")
        print(f"  客观: {note_str}")

        decode_results.append({
            "q": q, "fp32_out": out_fp, "bf16_out": out_bf,
            "fp32_circular": fp_circ, "bf16_circular": bf_circ,
            "fp32_early_stop": fp_stopped, "bf16_early_stop": bf_stopped,
            "note": note_str,
        })

    # 汇总
    n_circ_fp = sum(1 for r in decode_results if r["fp32_circular"])
    n_circ_bf = sum(1 for r in decode_results if r["bf16_circular"])
    summary = {
        "state_distance": {
            "overall_cosine": overall["cosine"],
            "overall_rel_l2": overall["rel_l2"],
            "per_layer_cosine_min": min(m["cosine"] for m in per_layer.values()),
            "per_layer_cosine_max": max(m["cosine"] for m in per_layer.values()),
            "per_layer_rel_l2_max": max(m["rel_l2"] for m in per_layer.values()),
        },
        "std": {
            "mean_fp32": round(sum(std_fp.values()) / len(std_fp), 5),
            "mean_bf16": round(sum(std_bf.values()) / len(std_bf), 5),
            "per_layer": {"fp32": std_fp, "bf16": std_bf, "diff": std_diff},
        },
        "decode": {
            "n_circular_fp32": n_circ_fp,
            "n_circular_bf16": n_circ_bf,
            "n_questions": len(decode_results),
        },
        "per_layer_distance": per_layer,
        "decode_results": decode_results,
    }

    print("\n" + "=" * 70)
    print("# state 距离汇总")
    print(f"  整体余弦相似度: {overall['cosine']:.5f}")
    print(f"  整体相对 L2:    {overall['rel_l2']:.5f}  ({overall['rel_l2']*100:.2f}%)")
    print(f"  逐层余弦范围:   [{summary['state_distance']['per_layer_cosine_min']:.5f}, "
          f"{summary['state_distance']['per_layer_cosine_max']:.5f}]")
    print(f"  逐层相对L2最大: {summary['state_distance']['per_layer_rel_l2_max']:.5f}")
    print(f"  std 均值: fp32={summary['std']['mean_fp32']:.5f}  bf16={summary['std']['mean_bf16']:.5f}")
    print(f"  循环检测: fp32={n_circ_fp}/{len(decode_results)}  bf16={n_circ_bf}/{len(decode_results)}")

    out = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"\n# 写入 {args.out}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
