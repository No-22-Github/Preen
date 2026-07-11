"""红线标定用的分桶数据集构造器。

从 NekoQA-10K 按 encode 后 token 数分桶抽样,产出 L≈450/500/550 各 30~50 条的数据集。
口径与 mem_probe_v2.build_fixed_length_samples 一致(真实编码 + max_len 截断保证上界),
但 target_len 提到 450~550 区间(超出 nekoqa200 的 max273,需要从 10K 全集筛长样本)。

注意:NekoQA 多数样本较短,长样本稀有。本脚本诚实报告每个桶实际凑到多少条
(不够就如实报告,不强行重复——红线标定需要真实长样本分布)。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from statetuner.data import encode_template_sample
from statetuner.templates import NEKO_QA


def build_bucket(tok, target_len, source_items, lo_ratio=0.92, hi_ratio=1.08, max_n=50):
    """从候选里筛 token 数落在 [target*lo, target*hi] 的,最多 max_n 条。

    用 max_len=target_len*1.1 截断保证上界(与 mem_probe_v2 一致)。
    返回 (samples, n_unique_source) —— n_unique 是去重前的唯一源样本数,
    用于诚实报告该桶是否够数。
    """
    lo, hi = target_len * lo_ratio, target_len * hi_ratio
    cap = int(target_len * 1.1)
    picked = []
    for it in source_items:
        q = (it.get("instruction") or "").strip()
        a = (it.get("output") or "").strip()
        if not a:
            continue
        s = encode_template_sample(tok, NEKO_QA, max_len=4096, q=q, a=a)
        if lo <= s.length <= hi:
            # 截断到 cap 保证上界
            s2 = encode_template_sample(tok, NEKO_QA, max_len=cap, q=s.cn, a=s.en)
            picked.append(s2)
            if len(picked) >= max_n:
                break
    return picked, len(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="train_data/NekoQA_10k/NekoQA-10K.json",
                    help="NekoQA-10K 源数据")
    ap.add_argument("--model", required=True, help="用来加载 tokenizer 的模型")
    ap.add_argument("--targets", default="450,500,550",
                    help="目标 token 数,逗号分隔")
    ap.add_argument("--per-bucket", type=int, default=40, help="每桶最多多少条")
    ap.add_argument("--out-dir", default="experiments/mixed_precision/data/redline_buckets")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载 tokenizer
    from statetuner.core import load_model
    _, tok = load_model(args.model, patch=False)

    # 加载源数据并打乱
    with open(args.source, encoding="utf-8") as f:
        items = json.load(f)
    rng = random.Random(args.seed)
    rng.shuffle(items)

    # 先看全集长度分布,诚实报告长样本稀缺度
    all_lens = []
    for it in items[:3000]:
        q = (it.get("instruction") or "").strip()
        a = (it.get("output") or "").strip()
        if not a:
            continue
        s = encode_template_sample(tok, NEKO_QA, max_len=4096, q=q, a=a)
        all_lens.append(s.length)
    all_lens.sort()
    n = len(all_lens)
    print(f"源数据长度分布(前{n}条):min={all_lens[0]} p50={all_lens[n//2]} "
          f"p90={all_lens[int(n*0.9)]} p99={all_lens[int(n*0.99)]} max={all_lens[-1]}")
    print(f"  >=400 的:{sum(1 for l in all_lens if l>=400)} 条")
    print(f"  >=450 的:{sum(1 for l in all_lens if l>=450)} 条")
    print(f"  >=500 的:{sum(1 for l in all_lens if l>=500)} 条")
    print(f"  >=550 的:{sum(1 for l in all_lens if l>=550)} 条")
    print()

    targets = [int(x) for x in args.targets.split(",")]
    manifest = {"source": args.source, "model": args.model, "seed": args.seed,
                "buckets": {}}

    for t in targets:
        samples, n = build_bucket(tok, t, items, max_n=args.per_bucket)
        bucket_name = f"L{t}"
        # 存成 NekoQA 兼容格式(instruction/output),供 load_qa_dataset 读
        bucket_items = [{"instruction": s.cn, "output": s.en} for s in samples]
        lens = sorted(s.length for s in samples)
        out_path = out_dir / f"{bucket_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bucket_items, f, ensure_ascii=False, indent=2)

        stats = {"target": t, "n": n, "min": lens[0] if lens else 0,
                 "max": lens[-1] if lens else 0, "mean": round(sum(lens)/n, 1) if lens else 0,
                 "path": str(out_path)}
        manifest["buckets"][bucket_name] = stats
        flag = "✓" if n >= 30 else ("⚠️" if n > 0 else "✗")
        print(f"{flag} {bucket_name}: {n} 条(目标 {args.per_bucket}) "
              f"len min={stats['min']} mean={stats['mean']} max={stats['max']} → {out_path}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nmanifest → {manifest_path}")

    # 诚实报告:有没有桶凑不够
    short = [k for k, v in manifest["buckets"].items() if v["n"] < 30]
    if short:
        print(f"\n⚠️ 以下桶不足 30 条:{short}。NekoQA 长样本稀缺,这是数据本身的限制,不强行凑。")


if __name__ == "__main__":
    main()
