"""P2 独立推理引擎快测（mock 模型，不加载权重）。"""
import numpy as np
import pytest

import mlx.core as mx

from statetuner.inference import (
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
    render_prompt,
)
from statetuner.templates import QA as NEKO_QA  # tests 局部别名：验收 f 扫 src/，不扫 tests/
from statetuner.templates import INSTRUCTION


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
    """验收 a: render_prompt(p, 'qa') 输出与旧 NEKO_QA.format_prefix 逐 token 相等。"""
    assert render_prompt("你好", "qa") == NEKO_QA.format_prefix(q="你好")
    assert render_prompt("raw", "raw") == "raw"


def test_render_prompt_qa_reasoning_fast_equals_legacy_g1g():
    """验收 b: qa + reasoning + think=fast 等价旧 G1G 模板(逐 token 相等)。

    旧 G1G.format_prefix(q="你好") = "<|bos|>User: 你好\\n\\nAssistant: <think>\\n</think>"
    新 render_prompt(p, "qa", reasoning=True, think="fast") 必须复现同一序列。
    """
    rendered = render_prompt("你好", "qa", reasoning=True, think="fast")
    expected = (
        "<|rwkv_tokenizer_end_of_text|>"
        "User: 你好\n\nAssistant: <think>\n</think>"
    )
    assert rendered == expected
    # 开头 bos(World tokenizer eos 字面量,encode 后即 token 0)
    assert rendered.startswith("<|rwkv_tokenizer_end_of_text|>")
    # 结尾空 think(告诉模型跳过思考直接答)
    assert rendered.endswith("Assistant: <think>\n</think>")


def test_render_prompt_think_modes_official_alignment():
    """验收 c: 三档 think 渲染对照官方文档字面量。

    Spec §1.1 映射表:
      off  → Assistant: 后追加 ""           (直答)
      fast → Assistant: 后追加 " <think>\n</think>"  (空 think 标签)
      on   → Assistant: 后追加 " <think"    (模型续写思考段)
    """
    p = "你好"
    # off: 不带 think 标签(reasoning=True 但 think=off,加 bos 不加 think 尾)
    off = render_prompt(p, "qa", reasoning=True, think="off")
    assert off == "<|rwkv_tokenizer_end_of_text|>User: 你好\n\nAssistant:"
    # fast: 空 think 标签
    fast = render_prompt(p, "qa", reasoning=True, think="fast")
    assert fast.endswith("Assistant: <think>\n</think>")
    # on: 尾部截断的 <think
    on = render_prompt(p, "qa", reasoning=True, think="on")
    assert on.endswith("Assistant: <think")


def test_render_prompt_think_requires_reasoning():
    """reasoning=False 时 think != off 应报参数错误(防误用)。"""
    with pytest.raises(ValueError, match="reasoning=False"):
        render_prompt("你好", "qa", reasoning=False, think="fast")


def test_render_prompt_instruction_empty_input_degrades():
    """验收 d: instruction 模板空 input 降级无残留空行。"""
    # 空 input: 应降级为 "Instruction: ...\\n\\nResponse:"(无 Input 段)
    degraded = render_prompt("做某事", "instruction", instruction_input="")
    assert degraded == "Instruction: 做某事\n\nResponse:"
    # 核心断言:不出现三连空行
    assert "\n\n\n" not in degraded
    # 非空 input: 保留完整格式
    full = render_prompt("做某事", "instruction", instruction_input="某上下文")
    assert full == "Instruction: 做某事\n\nInput: 某上下文\n\nResponse:"


def test_instruction_template_drop_input_when_empty_helper():
    """TaskTemplate.format_prefix 的 instruction-aware 降级在模板层可单独触发。"""
    assert INSTRUCTION.format_prefix(instruction="X", input="") == "Instruction: X\n\nResponse:"
    assert INSTRUCTION.format_prefix(instruction="X", input="Y") == "Instruction: X\n\nInput: Y\n\nResponse:"


def test_summary_line_format_is_stable():
    """summary_line() 是 cli.py preview / chat.py 共用的摘要行单一事实源。

    锁定输出格式（对齐 llama.cpp prompt/gen 分段），改动此格式需同步 golden/快照。
    """
    result = GenerationResult(
        text="hi",
        token_ids=[72, 73],
        stop_reason="eos",
        elapsed=1.25,
        used_state=True,
        config=GenerationConfig(),
        prompt_tokens=10,
        prompt_time=0.5,
        generation_time=0.75,
    )
    assert result.summary_line() == (
        "[stop=eos, tokens=2, 1.25s | "
        "Prompt: 20.0 t/s | Generation: 2.7 t/s]"
    )
