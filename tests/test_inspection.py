import json

import numpy as np
import pytest

from statetuner.inspection import inspect_data, load_qa_pairs, validate_state_for_model


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


# ── validate_state_for_model（不加载真实模型，tmp npz + fake model）──


class _FakeModelArgs:
    """模拟 mlx-lm 模型 args：hidden_size=1024, head_dim=64 → 16 头。"""

    hidden_size = 1024
    head_dim = 64


class _FakeModel:
    """模拟 mlx-lm 模型：2 层，每层期望 (16, 64, 64)。"""

    args = _FakeModelArgs()

    def __init__(self, n_layers=2):
        self.layers = [object() for _ in range(n_layers)]


def _save_state_npz(tmp_path, arrays: dict, name="state.npz"):
    """构造 tmp npz，arrays = {layer_idx: ndarray}。"""
    path = tmp_path / name
    np.savez(path, **{f"layer_{i}": arr for i, arr in arrays.items()})
    return path


def test_validate_state_accepts_matching_state(tmp_path):
    """层数与 shape 匹配 → 通过，返回加载的 dict。"""
    path = _save_state_npz(
        tmp_path,
        {0: np.zeros((16, 64, 64), dtype=np.float32),
         1: np.ones((16, 64, 64), dtype=np.float32)},
    )
    result = validate_state_for_model(path, _FakeModel(n_layers=2))
    assert set(result.keys()) == {0, 1}


def test_validate_state_rejects_layer_count_mismatch(tmp_path):
    """state 层数 ≠ 模型层数 → ValueError（错误信息含两边的数字）。"""
    path = _save_state_npz(
        tmp_path, {0: np.zeros((16, 64, 64), dtype=np.float32)}
    )
    with pytest.raises(ValueError, match="层数 1 与模型层数 2 不匹配"):
        validate_state_for_model(path, _FakeModel(n_layers=2))


def test_validate_state_rejects_shape_mismatch(tmp_path):
    """state shape ≠ 模型期望 → ValueError（错误信息含期望 shape）。"""
    path = _save_state_npz(
        tmp_path,
        {0: np.zeros((8, 64, 64), dtype=np.float32),   # 头数错(8 vs 16)
         1: np.zeros((16, 64, 64), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="shape 与模型不匹配"):
        validate_state_for_model(path, _FakeModel(n_layers=2))


def test_validate_state_rejects_non_rwkv7_format(tmp_path):
    """shape 非 (H,64,64) → rwkv7_compatible=False → ValueError。"""
    path = _save_state_npz(
        tmp_path,
        {0: np.zeros((16, 32, 64), dtype=np.float32),   # (H,32,64) 非 (H,64,64)
         1: np.zeros((16, 32, 64), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="RWKV-7"):
        validate_state_for_model(path, _FakeModel(n_layers=2))
