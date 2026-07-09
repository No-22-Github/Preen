"""
P0 最终验收确认: epoch4 (lr=0.01) + 完整条件性对照。

这是复核问题清单全部对齐后的最终验收。
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
import numpy as np
from mlx_lm import load
from data_v2 import extract_cn_en
from state_tuner import generate

MODEL = str(Path(__file__).parent.parent.parent / "models" / "converted" / "rwkv7-g1d-0.4b")
CKPT = str(Path(__file__).parent / "checkpoints_v3" / "ep04.npz")
TEST = str(Path(__file__).parent.parent.parent / "train_data" / "translate" / "test_10.jsonl")


def main():
    print("=" * 60)
    print("P0 最终验收 (epoch4, lr=0.01, std≈0.17)")
    print("=" * 60)
    model, tok = load(MODEL, tokenizer_config={"trust_remote_code": True})

    # 中文 held-out (前缀 {中文}\n)
    print("\n===中文 held-out (前缀 {中文}\\n)===")
    pairs = []
    with open(TEST, encoding="utf-8") as f:
        for line in f:
            cn, en = extract_cn_en(json.loads(line.strip())["text"])
            pairs.append((cn, en))

    for i, (cn, ref) in enumerate(pairs):
        out = generate(model, tok, f"{cn}\n", state_npz=CKPT, max_tokens=55)
        oc = out.split("\n")[0].strip() if "\n" in out else out.strip()
        print(f"[{i+1}] {cn}")
        print(f"    REF: {ref}")
        print(f"    OUT: {oc[:95]!r}")

    # 英文条件性 (5条)
    print("\n===条件性: 英文输入 (期望续写/照抄, 不翻译)===")
    eng = ["The weather is nice today.", "I like programming and reading.",
           "Artificial intelligence is changing the world.",
           "Please bring me a cup of coffee.", "The meeting starts at three."]
    for e in eng:
        out = generate(model, tok, f"{e}\n", state_npz=CKPT, max_tokens=40)
        oc = out.split("\n")[0].strip() if "\n" in out else out.strip()
        print(f"EN: {e}")
        print(f"OUT: {oc[:80]!r}")

    # 乱码/数字条件性 (3条)
    print("\n===条件性: 乱码/数字 (期望不吐英文翻译句子)===")
    junk = ["1234567890", "asdfghjkl zxcvbnm", "！！！@#￥%……"]
    for j in junk:
        out = generate(model, tok, f"{j}\n", state_npz=CKPT, max_tokens=40)
        oc = out.split("\n")[0].strip() if "\n" in out else out.strip()
        print(f"IN: {j}")
        print(f"OUT: {oc[:80]!r}")


if __name__ == "__main__":
    main()
