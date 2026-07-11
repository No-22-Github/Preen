import json

import pytest

from statetuner.inspection import inspect_data, load_qa_pairs


class CharTokenizer:
    @staticmethod
    def encode(text):
        return [ord(char) for char in text]


def test_inspect_data_counts_invalid_and_truncated(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            [
                {"instruction": "你好", "output": "喵"},
                {"instruction": "", "output": "喵"},
                {"instruction": "问题", "output": ""},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = inspect_data(path, CharTokenizer(), ctx_len=10)
    assert result.total == 3
    assert result.valid == 1
    assert result.skipped_empty_question == 1
    assert result.skipped_empty_answer == 1
    assert result.truncated == 1
    assert result.target_fully_truncated == 1


def test_inspect_data_reports_jsonl_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"instruction":"q","output":"a"}\n{bad}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        inspect_data(path, CharTokenizer())


def test_inspect_data_rejects_wrong_field_type(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('[{"instruction": 123, "output": "a"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="instruction 必须是字符串"):
        inspect_data(path, CharTokenizer())


def test_load_qa_pairs_allows_missing_reference(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text('[{"instruction":"q"}]', encoding="utf-8")
    assert load_qa_pairs(path) == [("q", "")]


def test_load_qa_pairs_rejects_empty_question(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text('[{"instruction":"","output":"a"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="非空字符串"):
        load_qa_pairs(path)
