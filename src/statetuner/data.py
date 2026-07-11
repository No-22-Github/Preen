"""数据管线: QA 数据集 → World tokenizer → 编码 + loss mask。

核心是 encode_template_sample: 对 prefix/target 拆分独立编码(不联合 encode),
再拼 + stop_token,让模型学会"停"。禁止在此手写 "\\n"-拼接的格式字面量(验收 d),
格式一律从 templates.TaskTemplate 派生。

loss mask 只算 target 段(prefix 是条件,不是学习目标)。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

from .templates import NEKO_QA

PathLike = Union[str, Path]


@dataclass
class Sample:
    """单条训练样本。

    full_ids:    prefix_ids + target_ids + [stop_token] (原始拼接,含终止符)
                 验收 b/c 断言拿这个。input_ids = full_ids[:-1]。
    input_ids / labels: 偏移一位的 token 序列(下一个 token 预测)
    mask[i]=1 → labels[i](= full[i+1])落在 target 段或终止符,算 loss
    cn / en:    原文中英文,debug/eval 用
    prefix_len: prefix 段 token 长度(= len(encode(prefix)),含 \\n)。
                mask 边界:label 落在 [prefix_len, len(full)) 区间才算 loss。
    """

    full_ids: List[int]
    input_ids: List[int]
    labels: List[int]
    mask: List[int]
    cn: str
    en: str
    prefix_len: int

    @property
    def length(self) -> int:
        return len(self.input_ids)


def load_jsonl(path: PathLike) -> List[dict]:
    """读取 jsonl,每行一个 json 对象。"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def encode_template_sample(
    tokenizer, template, *, max_len: int = 128, **fields
) -> Sample:
    """通用模板编码: prefix/target 各自独立 encode,再拼 + stop_token。

    禁止整段联合 encode(旧实现 `encode(f"{cn}\\n{en}")`):它让 prefix 与 target
    共享分词上下文,虽然 World tokenizer 基本不受影响,但一旦换 tokenizer 就会
    mask 错位。拆分编码保证 train/inference 同构——推理 prompt 永远是
    encode(prefix),训练 prefix 段就是 encode(prefix),逐 token 相等(验收 c)。

    新增终止符:full = prefix_ids + target_ids + [stop_token]。
    模型在 target 末尾必须学会预测 stop_token(eos=0),否则推理永不自停(旧实现
    无终止符,是模型学不会"停"的根因)。core.generate 已有 `next_token==0: break`
    分支消费它,无需改 core。

    loss mask:从预测第一个 target token 的位置(prefix_len-1)起 mask=1,
    含末位 stop_token。即 label 落在 [prefix_len, len(full)) 区间才算 loss。
    (mask[i]=1 当 (i+1) >= prefix_len。)

    tokenizer: 需有 .encode(str)->list[int] / .decode(ids)->str。
    template:  TaskTemplate 实例(prefix/target/stop_token 同源)。
    **fields:  模板占位符的值(如 cn=..., en=... 或 q=..., a=...)。
    """
    prefix_text = template.format_prefix(**fields)
    target_text = template.format_target(**fields)

    prefix_ids = tokenizer.encode(prefix_text)
    target_ids = tokenizer.encode(target_text)
    full_ids = prefix_ids + target_ids + [template.stop_token]

    prefix_len = len(prefix_ids)
    input_ids = full_ids[:-1]
    labels = full_ids[1:]
    mask = [1 if (i + 1) >= prefix_len else 0 for i in range(len(input_ids))]

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
        mask = mask[:max_len]

    # cn/en 仅用于 debug/eval 展示;通用模板无此概念时取 fields 里的近似值
    cn = fields.get("cn", fields.get("q", ""))
    en = fields.get("en", fields.get("a", ""))
    return Sample(full_ids, input_ids, labels, mask, cn, en, prefix_len)


def load_qa_dataset(
    path: PathLike,
    tokenizer,
    *,
    template=NEKO_QA,
    max_len: int = 512,
    question_key: str = "instruction",
    answer_key: str = "output",
) -> List[Sample]:
    """加载 QA 格式数据集 → 编码为 Sample 列表。

    用于角色扮演/问答任务。数据格式兼容两种:
      - .jsonl:每行一个 {question_key, answer_key}
      - .json:一个数组 [{...}, {...}](如 NekoQA-10K.json)

    每条 → encode_template_sample(template, q=..., a=..., max_len)。
    默认模板 NEKO_QA(prefix="User: {q}\\n\\nAssistant:", target=" {a}")。
    跳过 answer 为空的条目;超长样本按 max_len 截断(不丢弃)。

    question_key / answer_key 默认对齐 NekoQA 的 instruction/output 字段;
    其他 QA 数据集可显式传 question_key="..." / answer_key="..."。
    """
    path = Path(path)
    items = []
    if path.suffix == ".json":
        # 整个文件是一个 JSON 数组
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        items = loaded if isinstance(loaded, list) else [loaded]
    else:
        # jsonl(每行一个 json 对象)
        items = load_jsonl(path)

    samples = []
    for item in items:
        q = (item.get(question_key) or "").strip()
        a = (item.get(answer_key) or "").strip()
        if not a:
            continue  # 跳过无回答的条目
        s = encode_template_sample(tokenizer, template, max_len=max_len, q=q, a=a)
        samples.append(s)
    return samples


def train_test_split(
    samples: List[Sample],
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """划分训练/held-out(为 early stop 服务)。

    若调用方已提供独立 test 文件,应直接用 load_qa_dataset 加载而非用此函数。
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
        print(f"  prefix_len(边界): {s.prefix_len}, 总长: {s.length}")
        before = tokenizer.decode(s.input_ids[: s.prefix_len])
        at_boundary = (
            tokenizer.decode([s.input_ids[s.prefix_len]])
            if s.prefix_len < len(s.input_ids)
            else "?"
        )
        after = tokenizer.decode(s.input_ids[s.prefix_len : s.prefix_len + 5])
        print(f"  边界前(prefix): {before!r}")
        print(f"  边界处 token[{s.prefix_len}]: {at_boundary!r}")
        print(f"  边界后(target): {after!r}")
        transitions = [
            (j, s.mask[j])
            for j in range(len(s.mask))
            if j == 0 or s.mask[j] != s.mask[j - 1]
        ]
        print(f"  mask 0→1 转换: {transitions}")
        print()
