"""数据管线: jsonl → World tokenizer → 编码 + loss mask。

P0 复核结论(实验报告 §2.2):训练与评估的分布必须对齐,否则 state 只学到
模板偏置。本模块统一用「裸格式」:{中文}\\n{英文}。

格式解析(extract_cn_en)兼容两种输入:
  ① User/Assistant 模板: "User: {中}\\n\\nAssistant: {英}"
  ② 裸格式: "{中}\\n{英}"

loss mask 只算英文段(中文是条件,不是学习目标)。
边界用 encode(cn) 的长度近似——World tokenizer 基本是 char 级,稳定。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

PathLike = Union[str, Path]


@dataclass
class Sample:
    """单条训练样本。

    input_ids / labels: 偏移一位的 token 序列(下一个 token 预测)
    mask[i]=1 → labels[i](= full[i+1])落在英文段,算 loss
    cn / en:  原文中英文,debug/eval 用
    cn_len:   中文段 token 长度(mask 边界)
    """

    input_ids: List[int]
    labels: List[int]
    mask: List[int]
    cn: str
    en: str
    cn_len: int

    @property
    def length(self) -> int:
        return len(self.input_ids)


def load_jsonl(path: PathLike) -> List[dict]:
    """读取 jsonl,每行一个 {"text": ...} 或 {"cn":..., "en":...}。"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_cn_en(item: dict) -> Tuple[str, str]:
    """从一条记录提取 (中文, 英文)。

    兼容三种子格式:
      ① {"text": "User: {中}\\n\\nAssistant: {英}"}
      ② {"text": "{中}\\n{英}"}(已是裸格式)
      ③ {"cn": "{中}", "en": "{英}"}
    """
    if "cn" in item and "en" in item:
        return item["cn"].strip(), item["en"].strip()

    text = item.get("text", "")
    if "Assistant:" in text and "User:" in text:
        # 标准格式
        user_part = text.split("Assistant:")[0]
        cn = user_part.replace("User:", "").strip().rstrip("\n")
        en = text.split("Assistant:", 1)[1].strip()
        return cn, en
    if "\n" in text:
        # 已是 {中}\n{英} 裸格式
        cn, en = text.split("\n", 1)
        return cn.strip(), en.strip()
    return text, ""


def encode_sample(
    cn: str, en: str, tokenizer, max_len: int = 128
) -> Sample:
    """裸格式编码: "{中文}\\n{英文}" → input/label/mask。

    loss mask 只算英文段:mask[i]=1 当 labels[i](=full[i+1])落在英文段。
    边界用 cn_len = len(encode(cn)):中文占 [0, cn_len),从 full[cn_len] 开始
    的预测算 loss(\n 作为分隔符归入条件区,不算 loss)。
    """
    bare_text = f"{cn}\n{en}"
    full_ids = tokenizer.encode(bare_text)
    cn_len = len(tokenizer.encode(cn))

    input_ids = full_ids[:-1]
    labels = full_ids[1:]
    mask = [1 if (i + 1) >= cn_len else 0 for i in range(len(input_ids))]

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
        mask = mask[:max_len]

    return Sample(input_ids, labels, mask, cn, en, cn_len)


def load_dataset(
    path: PathLike, tokenizer, max_len: int = 128
) -> List[Sample]:
    """加载 jsonl → 编码为 Sample 列表。"""
    items = load_jsonl(path)
    samples = []
    for item in items:
        cn, en = extract_cn_en(item)
        if not en:
            continue  # 跳过无英文目标的记录
        samples.append(encode_sample(cn, en, tokenizer, max_len))
    return samples


def train_test_split(
    samples: List[Sample],
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """划分训练/held-out(为 early stop 服务)。

    若调用方已提供独立 test 文件,应直接用 load_dataset 加载而非用此函数。
    返回 (train, held_out)。
    """
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_test = max(1, int(len(samples) * test_ratio))
    test_idx = set(idx[:n_test])
    train = [samples[i] for i in range(len(samples)) if i not in test_idx]
    test = [samples[i] for i in sorted(test_idx)]
    return train, test


def verify_boundary(samples: List[Sample], tokenizer, n: int = 3) -> None:
    """打印前 n 条样本的 mask 边界,人工确认切对了(CLI preview 用)。"""
    for i, s in enumerate(samples[:n]):
        print(f"--- 样本 {i} ---")
        print(f"  中文: {s.cn}")
        print(f"  英文: {s.en}")
        print(f"  cn_len(边界): {s.cn_len}, 总长: {s.length}")
        before = tokenizer.decode(s.input_ids[: s.cn_len])
        at_boundary = (
            tokenizer.decode([s.input_ids[s.cn_len]])
            if s.cn_len < len(s.input_ids)
            else "?"
        )
        after = tokenizer.decode(s.input_ids[s.cn_len : s.cn_len + 5])
        print(f"  边界前(中文): {before!r}")
        print(f"  边界处 token[{s.cn_len}]: {at_boundary!r}")
        print(f"  边界后(英文): {after!r}")
        transitions = [
            (j, s.mask[j])
            for j in range(len(s.mask))
            if j == 0 or s.mask[j] != s.mask[j - 1]
        ]
        print(f"  mask 0→1 转换: {transitions}")
        print()
