"""importer 快测(纯数据,无模型依赖,~1min)。

覆盖 Spec §4.5 验收 a/b/c/e:
  a. 四类格式探测全部正确命中
  b. DPO 格式(chosen/rejected)走到 unknown 手动映射
  c. ShareGPT first/all 行数与人工计数一致
  e. import.json hash 可复现
(验收 d 端到端 train 冒烟在 test_importer_e2e.py --slow。)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from statetuner.importer import (
    CONFIDENCE_FLOOR,
    convert,
    detect_schema,
    import_dataset,
    preview_records,
    read_records,
    write_import,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "import"


# ── 读取 ────────────────────────────────────────────────────

def test_read_records_jsonl():
    items = read_records(FIXTURES / "bare_qa.jsonl")
    assert len(items) == 4
    assert all(isinstance(i, dict) for i in items)


def test_read_records_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("prompt,response\n你好,喵\n再见,拜\n", encoding="utf-8")
    items = read_records(csv_path)
    assert len(items) == 2
    assert items[0]["prompt"] == "你好"


def test_read_records_parquet_rejected_with_hint(tmp_path):
    pq = tmp_path / "data.parquet"
    pq.write_bytes(b"PAR1fake")
    with pytest.raises(ValueError, match="parquet"):
        read_records(pq)


def test_read_records_json_object_wrapped(tmp_path):
    """对象包 list:{"data": [...]} 包装。"""
    p = tmp_path / "wrapped.json"
    p.write_text(json.dumps({"data": [{"q": "x", "a": "y"}]}, ensure_ascii=False))
    items = read_records(p)
    assert len(items) == 1
    assert items[0]["q"] == "x"


# ── 探测(§4.5.a)────────────────────────────────────────────

def test_detect_messages():
    items = read_records(FIXTURES / "messages_chatml.jsonl")
    d = detect_schema(items)
    assert d.schema == "messages"
    assert d.confidence == 1.0
    assert "messages[].role==user" in d.prompt_keys


def test_detect_sharegpt():
    items = read_records(FIXTURES / "sharegpt_multiturn.jsonl")
    d = detect_schema(items)
    assert d.schema == "sharegpt"
    assert d.confidence == 1.0


def test_detect_alpaca():
    items = read_records(FIXTURES / "alpaca_sample.jsonl")
    d = detect_schema(items)
    assert d.schema == "alpaca"
    assert d.confidence == 1.0


def test_detect_bare_qa():
    items = read_records(FIXTURES / "bare_qa.jsonl")
    d = detect_schema(items)
    assert d.schema == "bare_qa"
    assert d.confidence == 1.0
    # 每行键名不同,但都收齐
    assert "question" in d.prompt_keys or "prompt" in d.prompt_keys
    assert "answer" in d.response_keys or "completion" in d.response_keys


# ── DPO 负样本(§4.5.b)──────────────────────────────────────

def test_detect_dpo_falls_to_unknown():
    """DPO 格式(chosen/rejected,无 instruction/output)不应误判。"""
    items = read_records(FIXTURES / "dpo_sample.jsonl")
    d = detect_schema(items)
    # DPO 有 prompt 但无 response 侧标准键(chosen/rejected 不在 RESPONSE_ALIASES),
    # 且不是 messages/sharegpt/alpaca → unknown
    assert d.schema == "unknown"


def test_convert_unknown_raises():
    items = read_records(FIXTURES / "dpo_sample.jsonl")
    d = detect_schema(items)
    with pytest.raises(ValueError, match="unknown"):
        convert(items, d)


# ── ShareGPT 多轮策略(§4.5.c)──────────────────────────────

def test_convert_sharegpt_first_policy():
    """first 策略:每条原始记录只取首对 user/assistant → 3 条样本。"""
    items = read_records(FIXTURES / "sharegpt_multiturn.jsonl")
    d = detect_schema(items)
    result = convert(items, d, turn_policy="first")
    assert len(result.records) == 3
    assert result.template == "qa"
    # 第一条原始有多轮,first 只取首对
    assert result.records[0]["prompt"] == "你好"
    assert result.records[0]["response"] == "你好!有什么可以帮你的吗?"
    # system 消息被丢弃计数(第一条有 1 个 system)
    assert result.dropped_system == 1


def test_convert_sharegpt_all_policy():
    """all 策略:每个相邻 user/assistant 对独立成样本。

    第 1 条原始:2 对(你好/你好!, 今天天气/我无法)
    第 2 条原始:1 对(1+1/等于2)
    第 3 条原始:2 对(教我写/从print, 然后呢/接着学)
    合计 5 条样本。
    """
    items = read_records(FIXTURES / "sharegpt_multiturn.jsonl")
    d = detect_schema(items)
    result = convert(items, d, turn_policy="all")
    assert len(result.records) == 5
    assert result.template == "qa"
    # all 策略扁平化:第 0/1 条来自原始第 1 条的两对,
    # 第 2 条来自原始第 2 条,第 3/4 条来自原始第 3 条。
    assert result.records[1]["prompt"] == "今天天气怎么样?"
    assert result.records[2]["prompt"] == "1+1等于几?"
    assert result.records[3]["prompt"] == "教我写 Python"


# ── Alpaca 转换 + 降级提示 ──────────────────────────────────

def test_convert_alpaca_instruction_template():
    items = read_records(FIXTURES / "alpaca_sample.jsonl")
    d = detect_schema(items)
    result = convert(items, d)
    assert result.template == "instruction"
    assert all("instruction" in r and "response" in r for r in result.records)
    # 有非空 input 的条目存在 → degradation_hint=False
    assert result.qa_degradation_hint is False


def test_convert_alpaca_all_empty_input_suggests_qa():
    """全部 input 为空 → qa_degradation_hint=True(§4.3 Alpaca 行)。"""
    items = [
        {"instruction": "Q1", "input": "", "output": "A1"},
        {"instruction": "Q2", "input": "", "output": "A2"},
    ]
    d = detect_schema(items)
    assert d.schema == "alpaca"
    result = convert(items, d)
    assert result.qa_degradation_hint is True


# ── 产物 + sidecar(§4.5.e hash 可复现)─────────────────────

def test_write_import_produces_jsonl_and_sidecar(tmp_path):
    items = read_records(FIXTURES / "alpaca_sample.jsonl")
    d = detect_schema(items)
    result = convert(items, d)
    out = tmp_path / "out.jsonl"
    artifact = write_import(FIXTURES / "alpaca_sample.jsonl", result, out)
    assert artifact.jsonl_path == out
    assert artifact.sidecar_path.exists()
    assert artifact.record_count == len(result.records)
    # jsonl 每行可解析
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(result.records)
    assert all(json.loads(line) for line in lines)


def test_import_json_hash_reproducible(tmp_path):
    """§4.5.e:同文件两次导入 sidecar hash 相等。"""
    src = FIXTURES / "alpaca_sample.jsonl"
    art1, _ = import_dataset(src, tmp_path / "a.jsonl")
    art2, _ = import_dataset(src, tmp_path / "b.jsonl")
    assert art1.sha256 == art2.sha256
    # sidecar 内容(除时间戳外)应一致——这里无时间戳字段,直接比对
    s1 = json.loads(art1.sidecar_path.read_text(encoding="utf-8"))
    s2 = json.loads(art2.sidecar_path.read_text(encoding="utf-8"))
    assert s1["source"]["sha256"] == s2["source"]["sha256"]
    assert s1["result"]["record_count"] == s2["result"]["record_count"]


def test_import_dataset_end_to_end_pipeline(tmp_path):
    """探测 → 转换 → 落盘 一站式入口。"""
    src = FIXTURES / "bare_qa.jsonl"
    artifact, result = import_dataset(src, tmp_path / "bare.jsonl")
    assert artifact.record_count == 4
    assert result.template == "qa"
    assert result.records[0]["prompt"] == "什么是 RWKV?"


# ── 预览(§4.4)─────────────────────────────────────────────

class FakeTokenizer:
    """简易 tokenizer:按空格 + 单字符近似分词(只测边界逻辑,不测真实编码)。"""

    def encode(self, text: str) -> list[int]:
        # 近似:每个非空字符一个 token(含标点),空格也算分隔
        return [ord(c) for c in text if c != " "]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def test_preview_records_qa_boundary():
    records = [{"prompt": "你好", "response": "喵~"}]
    rendered = preview_records(records, template="qa", tokenizer=FakeTokenizer())
    assert len(rendered) == 1
    r = rendered[0]
    # QA 模板 prefix = "User: 你好\n\nAssistant:",target = " 喵~"
    assert "User: 你好" in r.full_text
    assert "Assistant:" in r.full_text
    assert "喵~" in r.full_text
    # prefix_len = encode("User: 你好\n\nAssistant:") 的 token 数
    assert r.prefix_len > 0
    # prefix 段不应包含 target 文本
    prefix_text = r.full_text[: r.full_text.find("Assistant:") + len("Assistant:")]
    assert "喵" not in prefix_text


def test_preview_records_instruction_empty_input():
    """instruction 空 input 降级后无残留空行(§1.4.d)。"""
    records = [{"instruction": "计算", "input": "", "response": "结果"}]
    rendered = preview_records(records, template="instruction", tokenizer=FakeTokenizer())
    assert len(rendered) == 1
    r = rendered[0]
    assert "Input:" not in r.full_text  # 空 input 自动降级
    assert "\n\n\n" not in r.full_text   # 无三连空行
    assert r.prefix_len > 0
