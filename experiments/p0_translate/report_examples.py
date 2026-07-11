"""
为实验报告生成 10 个完整推理示例。
- 5 条来自训练集 (data_100.jsonl): 模型见过, 体现记忆/复现能力
- 5 条来自测试集 (test_10.jsonl): 模型未见过, 体现泛化能力
用最终 state (epoch4, lr=0.01), 前缀 {中文}\n (与训练对齐)。
输出完整不截断, 供报告引用。
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# 仓库根,用于 import statetuner.templates
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlx.core as mx
import numpy as np
from mlx_lm import load
from data_v2 import extract_cn_en
from state_tuner import generate
from statetuner.templates import P0_BARE

MODEL = str(Path(__file__).parent.parent.parent / "models" / "converted" / "rwkv7-g1d-0.4b")
STATE = str(Path(__file__).parent / "final_state_v3.npz")
TRAIN = str(Path(__file__).parent.parent.parent / "train_data" / "translate" / "data_100.jsonl")
TEST = str(Path(__file__).parent.parent.parent / "train_data" / "translate" / "test_10.jsonl")


def load_pairs(path, limit=None):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cn, en = extract_cn_en(json.loads(line)["text"])
            pairs.append((cn, en))
    return pairs[:limit] if limit else pairs


def main():
    model, tok = load(MODEL, tokenizer_config={"trust_remote_code": True})

    train_pairs = load_pairs(TRAIN, limit=5)
    # 测试集挑 5 条 (跳过前2条让样本分散, 选语义多样的)
    test_all = load_pairs(TEST)
    test_pairs = [test_all[i] for i in [0, 2, 4, 6, 8]]  # 隔行取, 覆盖不同主题

    print("########## 训练集内 (模型见过) ##########")
    for i, (cn, ref) in enumerate(train_pairs, 1):
        out = generate(model, tok, P0_BARE.format_prefix(cn=cn), state_npz=STATE, max_tokens=70)
        out_clean = out.split("\n")[0].strip() if "\n" in out else out.strip()
        print(f"\n[T{i}] CN: {cn}")
        print(f"     REF: {ref}")
        print(f"     OUT: {out_clean}")

    print("\n\n########## 测试集 (模型未见过) ##########")
    for i, (cn, ref) in enumerate(test_pairs, 1):
        out = generate(model, tok, P0_BARE.format_prefix(cn=cn), state_npz=STATE, max_tokens=70)
        out_clean = out.split("\n")[0].strip() if "\n" in out else out.strip()
        print(f"\n[S{i}] CN: {cn}")
        print(f"     REF: {ref}")
        print(f"     OUT: {out_clean}")


if __name__ == "__main__":
    main()
