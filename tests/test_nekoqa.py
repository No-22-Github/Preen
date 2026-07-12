"""NekoQA 数据管线快测(不依赖模型,秒级)。

验证 load_qa_dataset 的编码结构:终止符、mask 边界、prefix/target 同构。
真实模型上的行为(训练收敛、风格注入)在慢测/人工评估里验。
"""
import json
from pathlib import Path

import pytest

from statetuner.data import load_qa_dataset, train_test_split
from statetuner.templates import QA as NEKO_QA  # tests 局部别名


class _DummyTokenizer:
    """字符级 dummy tokenizer(测编码结构,不加载模型)。

    encode: char → ord(char);decode: ids → ''.join(chr(i))。
    """

    @staticmethod
    def encode(text):
        return [ord(c) for c in text]

    @staticmethod
    def decode(ids):
        return "".join(chr(i) for i in ids)


def _write_json_array(tmp_path, items, name="data.json"):
    p = tmp_path / name
    p.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return p


# ── load_qa_dataset 基础结构 ───────────────────────────────

def test_load_qa_dataset_json_array(tmp_path):
    """加载 .json 数组格式(NekoQA-10K.json 的格式)。"""
    items = [
        {"instruction": "你好", "output": "喵~主人好！"},
        {"instruction": "再见", "output": "呜...主人不要走"},
    ]
    p = _write_json_array(tmp_path, items)
    samples = load_qa_dataset(p, _DummyTokenizer(), max_len=512)

    assert len(samples) == 2
    for s in samples:
        # 终止符 + mask(验收 b 同款断言)
        assert s.full_ids[-1] == NEKO_QA.stop_token
        assert s.mask[-1] == 1
        # prefix 段不含 target
        assert s.full_ids[: s.prefix_len] == _DummyTokenizer().encode(
            NEKO_QA.format_prefix(q=s.prompt_text)
        )


def test_load_qa_dataset_skips_empty_answer(tmp_path):
    """answer 为空的条目应被跳过。"""
    items = [
        {"instruction": "有回答", "output": "喵"},
        {"instruction": "无回答", "output": ""},
        {"instruction": "null回答", "output": None},
    ]
    p = _write_json_array(tmp_path, items)
    samples = load_qa_dataset(p, _DummyTokenizer())
    assert len(samples) == 1
    assert samples[0].prompt_text == "有回答"


def test_load_qa_dataset_nekoqa_prefix_structure(tmp_path):
    """NEKO_QA 默认:prefix 末尾应是 'Assistant:',target 首字符是空格。"""
    items = [{"instruction": "你好", "output": "喵~"}]
    p = _write_json_array(tmp_path, items)
    samples = load_qa_dataset(p, _DummyTokenizer())
    s = samples[0]

    # prefix 解码应含 "User: 你好\n\nAssistant:"
    prefix_text = _DummyTokenizer().decode(s.full_ids[: s.prefix_len])
    assert prefix_text == "User: 你好\n\nAssistant:"
    # target 段(去掉末位 stop)首字符应是前导空格
    target_text = _DummyTokenizer().decode(s.full_ids[s.prefix_len:-1])
    assert target_text == " 喵~"


def test_load_qa_dataset_custom_keys(tmp_path):
    """非 NekoQA 字段名(question/answer)能用 question_key/answer_key 指定。"""
    items = [{"question": "Q1", "answer": "A1"}]
    p = _write_json_array(tmp_path, items)
    samples = load_qa_dataset(
        p, _DummyTokenizer(), question_key="question", answer_key="answer"
    )
    assert len(samples) == 1
    assert samples[0].prompt_text == "Q1"


def test_load_qa_dataset_jsonl_format(tmp_path):
    """jsonl 格式(每行一个 json)也能加载。"""
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"instruction": "你好", "output": "喵"}\n'
        '{"instruction": "再见", "output": "呜"}\n',
        encoding="utf-8",
    )
    samples = load_qa_dataset(p, _DummyTokenizer())
    assert len(samples) == 2
    assert [s.prompt_text for s in samples] == ["你好", "再见"]


# ── prefix/target 同构(验收 c 同款精神,用真实模板)──────────

def test_nekoqa_prefix_target_isomorphism(tmp_path):
    """encode(prefix)+encode(target) == encode_template_sample 的 prefix/target 段。

    NEKO_QA 在 dummy tokenizer 上验结构(真实 tokenizer 在 smoke 训练里验)。
    """
    items = [{"instruction": "你好", "output": "喵~主人"}]
    p = _write_json_array(tmp_path, items)
    samples = load_qa_dataset(p, _DummyTokenizer())
    s = samples[0]

    tok = _DummyTokenizer()
    prefix_ids = tok.encode(NEKO_QA.format_prefix(q="你好"))
    target_ids = tok.encode(NEKO_QA.format_target(a="喵~主人"))
    assert s.full_ids[: s.prefix_len] == prefix_ids
    assert s.full_ids[s.prefix_len:-1] == target_ids


# ── 与 train_test_split 的兼容性 ──────────────────────────

def test_nekoqa_train_test_split(tmp_path):
    """load_qa_dataset 产出的 Sample 列表能正常 train_test_split。"""
    items = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(20)]
    p = _write_json_array(tmp_path, items)
    samples = load_qa_dataset(p, _DummyTokenizer())

    tr1, te1 = train_test_split(samples, test_ratio=0.2, seed=42)
    tr2, te2 = train_test_split(samples, test_ratio=0.2, seed=42)
    assert [s.prompt_text for s in tr1] == [s.prompt_text for s in tr2]  # 可复现
    assert len(te1) == 4 and len(tr1) == 16
