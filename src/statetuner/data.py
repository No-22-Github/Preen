"""数据管线: QA 数据集 → World tokenizer → 编码 + loss mask。

核心是 encode_template_sample: 对 prefix/target 拆分独立编码(不联合 encode),
再拼 + stop_token,让模型学会"停"。禁止在此手写 "\\n"-拼接的格式字面量(验收 d),
格式一律从 templates.TaskTemplate 派生。

loss mask 只算 target 段(prefix 是条件,不是学习目标)。

内部标准 jsonl 字段名(Spec §1.3,导入器 §4 的产物契约):
  - qa 模板:    {"prompt": ..., "response": ...}
  - instruction: {"instruction": ..., "input": ..., "response": ...}
  load_qa_dataset 仍按 instruction/output 字段读现有数据集(数据文件不改名)。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

from .templates import INSTRUCTION, QA

PathLike = Union[str, Path]


@dataclass
class Sample:
    """单条训练样本。

    full_ids:    prefix_ids + target_ids + [stop_token] (原始拼接,含终止符)
                 验收 b/c 断言拿这个。input_ids = full_ids[:-1]。
    input_ids / labels: 偏移一位的 token 序列(下一个 token 预测)
    mask[i]=1 → labels[i](= full[i+1])落在 target 段或终止符,算 loss
    prompt_text / target_text:  原文 prompt 与 target,debug/eval 用
                 (历史字段 cn/en 已重命名,Spec §1.3 内部命名清债)
    prefix_len: prefix 段 token 长度(= len(encode(prefix)),含 \\n)。
                mask 边界:label 落在 [prefix_len, len(full)) 区间才算 loss。
    """

    full_ids: List[int]
    input_ids: List[int]
    labels: List[int]
    mask: List[int]
    prompt_text: str
    target_text: str
    prefix_len: int
    truncated: bool = False  # 是否因 max_len 截头(供 drop_truncated 过滤)

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

    超长样本:按 max_len **截头部**(丢 prefix 早期 token)以保尾部 stop_token,
    否则会砍掉 "让模型学会停" 的终止符。旧实现切尾部是 bug(S3)。

    tokenizer: 需有 .encode(str)->list[int] / .decode(ids)->str。
    template:  TaskTemplate 实例(prefix/target/stop_token 同源)。
    **fields:  模板占位符的值(如 q=..., a=... 或 instruction=..., input=..., a=...)。
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

    truncated = len(input_ids) > max_len
    if truncated:
        # S3:超长样本按 max_len 截断,且必须保尾部(含 stop_token)。
        # 旧实现切尾部 → 超长样本第一个被砍的就是 stop_token,正好毁掉
        # "让模型学会停" 这个核心设计。改为切头部:丢掉 prefix 早期 token,
        # 保留 target 末尾 + stop_token。同时 full_ids 一致截断(Sample 内部
        # 不再自相矛盾)。仍可能丢 target 前段,但 stop 一定在。
        overflow = len(input_ids) - max_len
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        mask = mask[overflow:]
        full_ids = full_ids[overflow:]

    # prompt_text/target_text 仅用于 debug/eval 展示;通用模板无 cn/en 概念,
    # 取 fields 里的近似值(兼容 q/a 与 instruction/input/a 占位符)。
    prompt_text = fields.get("prompt_text", fields.get("q", fields.get("instruction", "")))
    target_text_dbg = fields.get("target_text", fields.get("a", ""))
    return Sample(
        full_ids, input_ids, labels, mask, prompt_text, target_text_dbg, prefix_len,
        truncated=truncated,
    )


def load_qa_dataset(
    path: PathLike,
    tokenizer,
    *,
    template=QA,
    max_len: int = 512,
    question_key: str = "instruction",
    answer_key: str = "output",
    drop_truncated: bool = False,
) -> List[Sample]:
    """加载 QA 格式数据集 → 编码为 Sample 列表。

    用于角色扮演/问答任务。数据格式兼容两种:
      - .jsonl:每行一个 {question_key, answer_key}
      - .json:一个数组 [{...}, {...}](如 NekoQA-10K.json)

    每条 → encode_template_sample(template, q=..., a=..., max_len)。
    默认模板 QA(prefix="User: {q}\\n\\nAssistant:", target=" {a}")。
    跳过 answer 为空的条目;超长样本按 max_len 截断(不丢弃)。

    question_key / answer_key 默认对齐 NekoQA 数据集的 instruction/output 字段;
    其他 QA 数据集可显式传 question_key="..." / answer_key="..."。

    内部标准 jsonl(Spec §1.3,导入器产物)字段名为 prompt/response,
    那条路径由 §4 导入器实现时新增专用 loader;本函数仍服务现有数据文件。
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
        if drop_truncated and s.truncated:
            continue  # 用户选择丢弃截断样本(宁少训完整条,不训截头条)
        samples.append(s)
    return samples


def load_standard_jsonl(
    path: PathLike,
    tokenizer,
    *,
    template: str = "qa",
    max_len: int = 512,
    drop_truncated: bool = False,
) -> List[Sample]:
    """读取内部标准 jsonl(importer.py 产物)→ 编码为 Sample 列表。

    字段契约(Spec §1.3,与 importer.py write_import 产物一致):
      - qa 模板:        {"prompt": ..., "response": ...}
      - instruction 模板: {"instruction": ..., "input": ..., "response": ...}

    与 load_qa_dataset 的区别:后者按 instruction/output 键读现有 NekoQA 数据文件
    (数据文件不改名);本函数读 importer 产出的标准字段名。

    template 决定字段名 + 用哪个 TaskTemplate 编码(qa→QA, instruction→INSTRUCTION)。
    跳过 response 为空的条目;超长按 max_len 截断(不丢弃)。
    """
    if template == "qa":
        tmpl = QA
    elif template == "instruction":
        tmpl = INSTRUCTION
    else:
        raise ValueError(f"Standard JSONL supports only qa / instruction templates; received {template!r}")

    items = load_jsonl(Path(path))
    samples = []
    for item in items:
        if template == "qa":
            q = (item.get("prompt") or "").strip()
            a = (item.get("response") or "").strip()
            if not a:
                continue
            s = encode_template_sample(tokenizer, tmpl, max_len=max_len, q=q, a=a)
        else:  # instruction
            instruction = (item.get("instruction") or "").strip()
            inp = item.get("input") or ""
            a = (item.get("response") or "").strip()
            if not a:
                continue
            s = encode_template_sample(
                tokenizer, tmpl, max_len=max_len,
                instruction=instruction, input=inp, a=a,
            )
        if drop_truncated and s.truncated:
            continue  # 用户选择丢弃截断样本
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
