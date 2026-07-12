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


def test_p95_near_limit_threshold():
    """M4:p95 接近 16G bf16 红线(均值 591/max 644)时 p95_near_limit=True。

    红线来自 AGENTS.md;阈值 580(略低于均值 591,留提前量)。
    DataInspection 直接构造(不依赖 inspect_data 的数据准备)。
    """
    from statetuner.inspection import DataInspection

    common = dict(
        path="x", total=10, valid=10, skipped_empty_question=0,
        skipped_empty_answer=0, truncated=0, target_fully_truncated=0,
        min_tokens=100, mean_tokens=400, max_tokens=700, ctx_len=512,
    )
    # p95 远低于红线
    safe = DataInspection(p95_tokens=300, **common)
    assert safe.p95_near_limit is False
    # p95 接近/超过红线
    near = DataInspection(p95_tokens=585, **common)
    assert near.p95_near_limit is True
    over = DataInspection(p95_tokens=650, **common)
    assert over.p95_near_limit is True
    # 边界:正好阈值
    at = DataInspection(p95_tokens=580, **common)
    assert at.p95_near_limit is True


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
