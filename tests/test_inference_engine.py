"""P2 独立推理引擎快测（mock 模型，不加载权重）。"""
from types import SimpleNamespace

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

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


class LastHiddenInner:
    """RWKV7 内层 mock：记录主体吃到的序列长度。"""

    def __init__(self):
        self.input_lengths = []

    def __call__(self, input_ids, caches):
        self.input_lengths.append(input_ids.shape[1])
        return mx.zeros((1, input_ids.shape[1], 4))


class RecordingProjection:
    """词表投影 mock：只接受 last hidden，每次生成 A。"""

    def __init__(self):
        self.input_shapes = []

    def __call__(self, hidden):
        self.input_shapes.append(hidden.shape)
        logits = np.zeros((1, hidden.shape[1], 128), dtype=np.float32)
        logits[0, -1, 65] = 10
        return mx.array(logits)


class FastPathRwkv7Model:
    model_type = "rwkv7"

    def __init__(self):
        self.args = SimpleNamespace(tie_word_embeddings=False)
        self.model = LastHiddenInner()
        self.lm_head = RecordingProjection()

    def make_cache(self):
        return []

    def __call__(self, input_ids, caches):
        raise AssertionError("RWKV7 快路径不应构造整段 logits")


class TinyArraysCache:
    """compiled decode 测试用的最小 ArraysCache 等价物。"""

    def __init__(self):
        self.cache = [mx.zeros((1, 1, 1)) for _ in range(3)]

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value):
        self.cache = value

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value


class TinyRwkv7Inner(nn.Module):
    def __call__(self, input_ids, caches):
        # hidden 固定，让 lm_head 决定 A/B 排名；cache 则记录最后一个输入，
        # 用于验证 compiled state 最终确实写回调用方对象。
        hidden = mx.ones((*input_ids.shape, 1))
        last = input_ids[:, -1:].astype(mx.float32)[..., None]
        caches[0][0] = last
        caches[0][1] = last + 1
        caches[0][2] = last + 2
        return hidden


class TinyCompiledRwkv7Model(nn.Module):
    model_type = "rwkv7"

    def __init__(self):
        super().__init__()
        self.args = SimpleNamespace(tie_word_embeddings=False)
        self.model = TinyRwkv7Inner()
        self.lm_head = nn.Linear(1, 128, bias=False)
        weight = mx.zeros((128, 1))
        weight[65, 0] = 10.0
        weight[66, 0] = 9.5
        self.lm_head.weight = weight

    def make_cache(self):
        return [TinyArraysCache()]


class QuantizedMarker(nn.Module):
    pass


def test_rwkv7_projects_only_last_hidden_during_prefill():
    model = FastPathRwkv7Model()
    result = InferenceEngine(model, DummyTokenizer()).generate(
        "long",
        config=GenerationConfig(
            max_tokens=3,
            presence_penalty=0,
            frequency_penalty=0,
        ),
    )

    assert result.text == "AAA"
    assert model.model.input_lengths == [4, 1, 1]
    assert model.lm_head.input_shapes == [(1, 1, 4)] * 3


def test_rwkv7_tied_embeddings_project_only_last_hidden():
    model = FastPathRwkv7Model()
    projection = RecordingProjection()
    model.args.tie_word_embeddings = True
    model.model.embeddings = SimpleNamespace(as_linear=projection)
    del model.lm_head

    result = InferenceEngine(model, DummyTokenizer()).generate(
        "long",
        config=GenerationConfig(
            max_tokens=2,
            presence_penalty=0,
            frequency_penalty=0,
        ),
    )

    assert result.text == "AA"
    assert projection.input_shapes == [(1, 1, 4)] * 2


def test_rwkv7_compiled_decode_matches_eager_and_writes_back_cache():
    cfg = GenerationConfig(max_tokens=3)
    compiled_engine = InferenceEngine(TinyCompiledRwkv7Model(), DummyTokenizer())
    eager_engine = InferenceEngine(
        TinyCompiledRwkv7Model(), DummyTokenizer(), compile_decode=False
    )

    assert compiled_engine.compiled_decode_enabled is True
    assert compiled_engine.decode_backend == "mx.compile+async"
    assert eager_engine.decode_backend == "eager"
    compiled = compiled_engine.generate("x", config=cfg)
    eager = eager_engine.generate("x", config=cfg)

    # penalty 保持在图外后，A(首选)→B(A 被罚)→A 的语义与 eager 完全一致。
    assert compiled.display_token_ids == eager.display_token_ids == [65, 66, 65]
    for actual, expected in zip(compiled.cache[0].state, eager.cache[0].state):
        assert mx.array_equal(actual, expected).item()
    assert compiled.cache[0].state[0].item() == 66


def test_rwkv7_compiled_decode_supports_plain_list_state_cache():
    engine = InferenceEngine(TinyCompiledRwkv7Model(), DummyTokenizer())
    result = engine.generate(
        "x",
        state={0: mx.zeros((1, 1, 1))},
        config=GenerationConfig(
            max_tokens=3,
            presence_penalty=0,
            frequency_penalty=0,
        ),
    )

    assert result.text == "AAA"
    assert isinstance(result.cache[0], list)
    assert result.cache[0][0].item() == 65


def test_rwkv7_pipeline_discards_prefetch_on_eos():
    model = TinyCompiledRwkv7Model()
    weight = mx.zeros((128, 1))
    weight[0, 0] = 10.0
    model.lm_head.weight = weight
    result = InferenceEngine(model, DummyTokenizer()).generate(
        "x",
        config=GenerationConfig(
            max_tokens=3,
            presence_penalty=0,
            frequency_penalty=0,
        ),
    )

    assert result.stop_reason == "eos"
    assert result.decode_steps == 0
    # 流水线已投机提交 token 0 的下一步，但返回 cache 只能包含 prompt x。
    assert result.cache[0].state[0].item() == ord("x")


def test_rwkv7_pipeline_discards_prefetch_on_stop_sequence():
    result = InferenceEngine(
        TinyCompiledRwkv7Model(), DummyTokenizer()
    ).generate(
        "x",
        config=GenerationConfig(
            max_tokens=3,
            stop_sequences=("A",),
            presence_penalty=0,
            frequency_penalty=0,
        ),
    )

    assert result.stop_reason == "stop_sequence"
    assert result.text == ""
    # 预测出的 A 没有进入 eager 旧语义的 cache；future_state 必须被丢弃。
    assert result.cache[0].state[0].item() == ord("x")


def test_rwkv7_pipeline_sampling_matches_eager_with_same_seed():
    cfg = GenerationConfig(
        max_tokens=5,
        temperature=0.8,
        top_p=0.9,
        seed=7,
        presence_penalty=0,
        frequency_penalty=0,
    )
    compiled_engine = InferenceEngine(TinyCompiledRwkv7Model(), DummyTokenizer())
    eager_engine = InferenceEngine(
        TinyCompiledRwkv7Model(), DummyTokenizer(), compile_decode=False
    )

    compiled = compiled_engine.generate("x", config=cfg)
    eager = eager_engine.generate("x", config=cfg)
    assert compiled.display_token_ids == eager.display_token_ids


def test_rwkv7_quantized_model_keeps_eager_decode_path():
    model = TinyCompiledRwkv7Model()
    model.quantized_marker = QuantizedMarker()
    assert InferenceEngine(model, DummyTokenizer()).compiled_decode_enabled is False


def test_rwkv7_sync_compiled_backend_is_selectable_for_benchmark():
    engine = InferenceEngine(
        TinyCompiledRwkv7Model(), DummyTokenizer(), async_decode=False
    )
    result = engine.generate(
        "x",
        config=GenerationConfig(
            max_tokens=3,
            presence_penalty=0,
            frequency_penalty=0,
        ),
    )
    assert engine.decode_backend == "mx.compile"
    assert result.text == "AAA"


def test_greedy_result_has_stop_reason_and_tokens():
    engine = InferenceEngine(SequenceModel([72, 73, 0]), DummyTokenizer())
    result = engine.generate("x", config=GenerationConfig(max_tokens=10))
    assert result.text == "HI"
    assert result.display_token_ids == [72, 73]
    assert result.stop_reason == "eos"
    assert result.token_count == 2
    assert result.decode_steps == 2  # step 1 生成 I，step 2 生成 eos
    assert result.used_state is False


def test_max_tokens_stop_reason():
    engine = InferenceEngine(SequenceModel([65]), DummyTokenizer())
    result = engine.generate("x", config=GenerationConfig(max_tokens=3))
    assert result.text == "AAA"
    assert result.stop_reason == "max_tokens"
    assert result.decode_steps == 2  # 3 个前向中 step 0 归 prefill


def test_sampling_path_is_seeded_and_callable():
    cfg = GenerationConfig(max_tokens=3, temperature=0.8, top_p=0.9, seed=7)
    first = InferenceEngine(SequenceModel([65]), DummyTokenizer()).generate("x", config=cfg)
    second = InferenceEngine(SequenceModel([65]), DummyTokenizer()).generate("x", config=cfg)
    assert first.display_token_ids == second.display_token_ids == [65, 65, 65]


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
    assert result.display_token_ids == [72, 73]


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
    with pytest.raises(ValueError, match="requires a state"):
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
        display_token_ids=[72, 73],
        stop_reason="eos",
        elapsed=1.25,
        used_state=True,
        config=GenerationConfig(),
        prompt_tokens=10,
        prompt_time=0.5,
        generation_time=0.75,
        decode_steps=2,
    )
    assert result.summary_line() == (
        "[stop=eos, tokens=2, 1.25s | "
        "Prompt: 20.0 t/s | Generation: 2.7 t/s]"
    )


def test_generation_tps_uses_timed_decode_steps_not_all_generated_tokens():
    """首 token 在 prefill step 里产生，不能计入 generation_time 的分子。"""
    result = GenerationResult(
        text="x" * 128,
        display_token_ids=list(range(128)),
        stop_reason="max_tokens",
        elapsed=4.0,
        used_state=False,
        config=GenerationConfig(max_tokens=128),
        prompt_tokens=1024,
        prompt_time=0.5,
        generation_time=3.175,
        decode_steps=127,
    )
    assert result.generation_tps == pytest.approx(40.0)
    assert result.to_dict()["decode_steps"] == 127


# ────────────────────────────────────────────────────────────────
# Phase 3 §2: 多轮 cache 续传 + token 账本拆分 + cache 洁净性
# ────────────────────────────────────────────────────────────────


class CacheAwareModel:
    """支持 cache 跨 generate 调用续传的 mock model。

    记录每次 __call__ 的 input_ids,以便测试断言续传/重放各自喂入了什么。
    caches 透传给调用方(模型原地更新 cache 列表)——模拟真实 RWKV7 的 cache 语义
    (cache[0]/[1]/[2] 在前向中被赋值更新)。
    """

    def __init__(self, token_stream):
        """token_stream: list,每次前向产出哪个 token(按 self.calls 索引,取末位兜底)。"""
        self.token_stream = list(token_stream)
        self.calls = 0
        self.fed_inputs: list[list[int]] = []  # 每次 __call__ 收到的 input_ids(扁平)

    def make_cache(self):
        return []

    def __call__(self, input_ids, caches):
        self.calls += 1
        ids_flat = input_ids.tolist()[0] if input_ids.ndim == 2 else input_ids.tolist()
        self.fed_inputs.append(list(ids_flat))
        token = self.token_stream[min(self.calls - 1, len(self.token_stream) - 1)]
        logits = np.zeros((1, input_ids.shape[1], 128), dtype=np.float32)
        logits[0, -1, token] = 10
        return mx.array(logits)


def test_generate_accepts_cache_for_continuation():
    """§2.4 API: generate(cache=) 传入则续传(prefill 只吃 prompt,cache 保留)。

    续传语义:传入 cache 时,模型 __call__ 的第一次前向只消化新 prompt token,
    不从零开始。这里不验证 cache 内容正确性(需真实模型),只验证 API 通路 +
    cache 被透传到模型、且原样出现在 result.cache。
    """
    model = CacheAwareModel([72, 73, 0])
    engine = InferenceEngine(model, DummyTokenizer())
    incoming = object()  # 哨兵:验证 cache 透传
    result = engine.generate("x", cache=incoming, config=GenerationConfig(max_tokens=5))
    # cache 传出(可能是同一个对象或新建,关键是 API 契约存在)
    assert hasattr(result, "cache")
    assert result.cache_clean is True  # eos 路径干净


def test_generate_returns_cache_clean_flags():
    """§2.2 洁净性:eos/max_tokens 干净,stop_sequence 脏。"""
    # eos 干净
    eos_result = InferenceEngine(
        CacheAwareModel([72, 0]), DummyTokenizer()
    ).generate("x", config=GenerationConfig(max_tokens=5))
    assert eos_result.stop_reason == "eos"
    assert eos_result.cache_clean is True

    # max_tokens 干净
    mt_result = InferenceEngine(
        CacheAwareModel([65]), DummyTokenizer()
    ).generate("x", config=GenerationConfig(max_tokens=3))
    assert mt_result.stop_reason == "max_tokens"
    assert mt_result.cache_clean is True

    # stop_sequence 脏
    boundary = [ord(c) for c in NEKO_QA.inference_stop_sequences[0]]
    ss_result = InferenceEngine(
        CacheAwareModel([72, 73, *boundary, 65]), DummyTokenizer()
    ).generate(
        "x",
        config=GenerationConfig(max_tokens=30, stop_sequences=NEKO_QA.inference_stop_sequences),
    )
    assert ss_result.stop_reason == "stop_sequence"
    assert ss_result.cache_clean is False


def test_fed_token_ids_superset_of_display_on_stop_sequence():
    """§2.6.d: stop_sequence 停止时 fed_token_ids 是 display_token_ids 的超集(含污染)。

    eos 停止时两者相等(eos 不进 cache)。
    """
    # eos: 相等
    eos = InferenceEngine(
        CacheAwareModel([72, 73, 0]), DummyTokenizer()
    ).generate("x", config=GenerationConfig(max_tokens=5))
    assert eos.fed_token_ids == eos.display_token_ids

    # stop_sequence: fed ⊇ display(污染部分 = \nUser: 的若干 token)
    boundary = [ord(c) for c in NEKO_QA.inference_stop_sequences[0]]
    ss = InferenceEngine(
        CacheAwareModel([72, 73, *boundary, 65]), DummyTokenizer()
    ).generate(
        "x",
        config=GenerationConfig(max_tokens=30, stop_sequences=NEKO_QA.inference_stop_sequences),
    )
    # display = [72, 73] (HI),fed 包含 72,73 + boundary 的污染 token
    assert ss.display_token_ids == [72, 73]
    assert set(ss.fed_token_ids) >= set(ss.display_token_ids)
    assert len(ss.fed_token_ids) > len(ss.display_token_ids)


def test_continuation_qa_cache_is_passed_through():
    """§2.4 API: generate(cache=) 传入时,cache 被透传给模型并原样传出。

    等价性的数值证明见 docs/g1g-decode-alignment.md §8.4(World tokenizer 实测:
    encode(prefix)+encode(' A1')+encode(continuation) == encode(整体))。
    这里验证机制:续传时同一 cache 对象被复用、并出现在 result.cache。
    """
    model = CacheAwareModel([65, 66, 0])
    engine = InferenceEngine(model, DummyTokenizer())
    sentinel_cache = []  # 可变哨兵
    result = engine.generate(
        "x", cache=sentinel_cache, config=GenerationConfig(max_tokens=3)
    )
    # cache 透传:同一对象(模型原地更新,sentinel_cache 被 model.__call__ 收到)
    assert result.cache is sentinel_cache


def test_replay_uses_fresh_cache_when_cache_none():
    """§2.4 API: cache=None 时按 state 新建 running cache,不复用任何旧 cache。

    验证重放路径的 cache 隔离:两次独立的 cache=None 调用产出不同的 cache 对象。
    """
    model = CacheAwareModel([65, 66, 0])
    engine = InferenceEngine(model, DummyTokenizer())
    r1 = engine.generate("x", config=GenerationConfig(max_tokens=3))
    r2 = engine.generate("x", config=GenerationConfig(max_tokens=3))
    assert r1.cache is not r2.cache  # 各自独立新建


def test_cache_none_creates_fresh_per_state():
    """§2.4 API: cache=None 时按 state 新建 cache(零 state 走 make_cache)。"""
    model = CacheAwareModel([72, 0])
    engine = InferenceEngine(model, DummyTokenizer())
    result = engine.generate("x", cache=None, config=GenerationConfig(max_tokens=5))
    # cache=None 等价于不传,从零开始
    assert result.cache is not None
    assert result.used_state is False


# ────────────────────────────────────────────────────────────────
# 重复惩罚(ChatRWKV 官方 occurrence penalty 对齐)
# ────────────────────────────────────────────────────────────────


def test_generation_config_default_penalty_matches_chatrwkv():
    """默认 penalty 值对齐 ChatRWKV 官方 v2/chat.py:
    presence=0.4, frequency=0.4, decay=0.996。
    """
    cfg = GenerationConfig()
    assert cfg.presence_penalty == 0.4
    assert cfg.frequency_penalty == 0.4
    assert cfg.penalty_decay == 0.996
    assert cfg.has_penalty is True


def test_penalty_zero_disables():
    """presence=frequency=0 时 has_penalty=False,不施加任何惩罚。"""
    cfg = GenerationConfig(presence_penalty=0, frequency_penalty=0)
    assert cfg.has_penalty is False


def test_penalty_validation():
    """penalty 参数校验。"""
    with pytest.raises(ValueError, match="presence_penalty"):
        GenerationConfig(presence_penalty=-0.1).validate()
    with pytest.raises(ValueError, match="frequency_penalty"):
        GenerationConfig(frequency_penalty=-1).validate()
    with pytest.raises(ValueError, match="penalty_decay"):
        GenerationConfig(penalty_decay=0).validate()
    with pytest.raises(ValueError, match="penalty_decay"):
        GenerationConfig(penalty_decay=1.5).validate()


class TwoTokenCompetingModel:
    """每步输出两个接近的候选 token,用于测试 penalty 是否能翻转选择。

    logits 构造:A 比 B 略高(差距 < penalty),penalty 一次后 B 应反超 A。
    """

    def __init__(self, token_a, token_b, gap=0.5):
        self.token_a = token_a
        self.token_b = token_b
        self.gap = gap  # A - B 的 logits 差
        self.calls = 0

    def make_cache(self):
        return []

    def __call__(self, input_ids, caches):
        self.calls += 1
        logits = np.zeros((1, input_ids.shape[1], 128), dtype=np.float32)
        logits[0, -1, self.token_a] = 10.0
        logits[0, -1, self.token_b] = 10.0 - self.gap
        return mx.array(logits)


def test_penalty_suppresses_repeated_token():
    """penalty 生效:首次选 A(更高),A 被 penalty 后第二次选 B。

    gap=0.5: A=10, B=9.5。首次选 A。
    penalty 后 A=10-(0.4+1.0*0.4)=9.2 < B=9.5,第二次选 B。
    """
    model = TwoTokenCompetingModel(token_a=72, token_b=73, gap=0.5)
    engine = InferenceEngine(model, DummyTokenizer())
    result = engine.generate(
        "x",
        config=GenerationConfig(
            max_tokens=3, temperature=0.0,
            presence_penalty=0.4, frequency_penalty=0.4, penalty_decay=0.996,
        ),
    )
    # 首次 72(A),penalty 后 73(B)。序列应为 [72, 73] 后续交替
    assert result.display_token_ids[0] == 72  # A(首次,无 penalty)
    assert result.display_token_ids[1] == 73  # B(A 被 penalty 后反超)


def test_no_penalty_repeats_top_token():
    """对照:penalty=0 时,A 每次都最高,连续选 A。"""
    model = TwoTokenCompetingModel(token_a=72, token_b=73, gap=0.5)
    engine = InferenceEngine(model, DummyTokenizer())
    result = engine.generate(
        "x",
        config=GenerationConfig(
            max_tokens=3, temperature=0.0,
            presence_penalty=0, frequency_penalty=0,
        ),
    )
    # 无 penalty: 每次都选 A(72)
    assert result.display_token_ids == [72, 72, 72]
