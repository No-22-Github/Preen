"""P2 独立推理引擎快测（mock 模型，不加载权重）。"""
import numpy as np
import pytest

import mlx.core as mx

from statetuner.inference import GenerationConfig, InferenceEngine, render_prompt
from statetuner.templates import NEKO_QA


class DummyTokenizer:
    @staticmethod
    def encode(text):
        return [ord(char) for char in text]

    @staticmethod
    def decode(ids):
        return "".join(chr(i) for i in ids)


class ContextualBoundaryTokenizer(DummyTokenizer):
    """模拟角色边界独立编码与生成上下文中的 tokenization 不同。"""

    @staticmethod
    def encode(text):
        if text == NEKO_QA.inference_stop_sequences[0]:
            return [127]
        return [ord(char) for char in text]


class SequenceModel:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.calls = 0

    def make_cache(self):
        return []

    def __call__(self, input_ids, caches):
        token = self.tokens[min(self.calls, len(self.tokens) - 1)]
        self.calls += 1
        logits = np.zeros((1, input_ids.shape[1], 128), dtype=np.float32)
        logits[0, -1, token] = 10
        return mx.array(logits)


def test_greedy_result_has_stop_reason_and_tokens():
    engine = InferenceEngine(SequenceModel([72, 73, 0]), DummyTokenizer())
    result = engine.generate("x", config=GenerationConfig(max_tokens=10))
    assert result.text == "HI"
    assert result.token_ids == [72, 73]
    assert result.stop_reason == "eos"
    assert result.token_count == 2
    assert result.used_state is False


def test_max_tokens_stop_reason():
    engine = InferenceEngine(SequenceModel([65]), DummyTokenizer())
    result = engine.generate("x", config=GenerationConfig(max_tokens=3))
    assert result.text == "AAA"
    assert result.stop_reason == "max_tokens"


def test_sampling_path_is_seeded_and_callable():
    cfg = GenerationConfig(max_tokens=3, temperature=0.8, top_p=0.9, seed=7)
    first = InferenceEngine(SequenceModel([65]), DummyTokenizer()).generate("x", config=cfg)
    second = InferenceEngine(SequenceModel([65]), DummyTokenizer()).generate("x", config=cfg)
    assert first.token_ids == second.token_ids == [65, 65, 65]


def test_template_stop_sequence_is_removed_from_output():
    boundary = [ord(char) for char in NEKO_QA.inference_stop_sequences[0]]
    model = SequenceModel([72, 73, *boundary, 65])
    engine = InferenceEngine(model, DummyTokenizer())
    result = engine.generate(
        "x",
        config=GenerationConfig(
            max_tokens=30,
            stop_sequences=NEKO_QA.inference_stop_sequences,
        ),
    )
    assert result.text == "HI"
    assert result.stop_reason == "stop_sequence"
    assert result.token_ids == [72, 73]


def test_streaming_eos_flushes_text():
    chunks = []
    result = InferenceEngine(
        SequenceModel([72, 73, 0]), DummyTokenizer()
    ).generate("x", config=GenerationConfig(max_tokens=10), on_text=chunks.append)
    assert "".join(chunks) == result.text == "HI"
    assert result.stop_reason == "eos"


def test_streaming_buffers_and_hides_stop_sequence():
    boundary = [ord(char) for char in NEKO_QA.inference_stop_sequences[0]]
    chunks = []
    result = InferenceEngine(
        SequenceModel([72, 73, *boundary, 65]), DummyTokenizer()
    ).generate(
        "x",
        config=GenerationConfig(
            max_tokens=30,
            stop_sequences=NEKO_QA.inference_stop_sequences,
        ),
        on_text=chunks.append,
    )
    streamed = "".join(chunks)
    assert streamed == result.text == "HI"
    assert "User:" not in streamed
    assert result.stop_reason == "stop_sequence"


def test_text_stop_detection_ignores_contextual_tokenization_difference():
    boundary = [ord(char) for char in NEKO_QA.inference_stop_sequences[0]]
    chunks = []
    result = InferenceEngine(
        SequenceModel([72, 73, *boundary, 65]), ContextualBoundaryTokenizer()
    ).generate(
        "x",
        config=GenerationConfig(
            max_tokens=30,
            stop_sequences=NEKO_QA.inference_stop_sequences,
        ),
        on_text=chunks.append,
    )
    assert result.text == "HI"
    assert "".join(chunks) == "HI"
    assert result.stop_reason == "stop_sequence"


def test_ab_requires_state():
    engine = InferenceEngine(SequenceModel([0]), DummyTokenizer())
    with pytest.raises(ValueError, match="必须提供 state"):
        engine.compare("x", state=None)


def test_generation_config_validation():
    with pytest.raises(ValueError, match="max_tokens"):
        GenerationConfig(max_tokens=0).validate()
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(temperature=-1).validate()
    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(top_p=0).validate()


def test_render_prompt_uses_nekoqa_single_source():
    assert render_prompt("你好", "nekoqa") == NEKO_QA.format_prefix(q="你好")
    assert render_prompt("raw", "raw") == "raw"
