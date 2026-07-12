from pathlib import Path

from statetuner.chat import ChatSession
from statetuner.inference import ABResult, GenerationConfig, GenerationResult
from statetuner.templates import QA as NEKO_QA  # tests 局部别名


class FakeEngine:
    """记录调用以支持多轮续传/重放断言。

    generate 的 cache 参数被记录(哨兵 `_sentinel` 表示未传),
    用于断言 ChatSession 在续传 vs 重放场景下的 cache 传入行为。
    """

    _sentinel = object()

    def __init__(self):
        self.calls: list[dict] = []  # 每次 generate 的参数快照

    def load_state(self, state):
        return {0: str(state)}

    def generate(self, prompt, *, state=None, config=None, cache=_sentinel, on_text=None):
        self.calls.append({"prompt": prompt, "state": state, "cache": cache})
        text = "tuned" if state is not None else "baseline"
        if on_text is not None:
            on_text(text)
        return GenerationResult(
            text=text,
            display_token_ids=[1],
            fed_token_ids=[1],
            stop_reason="eos",
            cache_clean=True,
            cache=object(),  # 哨兵 cache 对象,供 ChatSession 续传持有
            elapsed=0.1,
            used_state=state is not None,
            config=config,
        )

    def compare(self, prompt, *, state, config):
        return ABResult(
            prompt=prompt,
            with_state=self.generate(prompt, state=state, config=config),
            baseline=self.generate(prompt, state=None, config=config),
        )


def test_chat_dynamic_state_switch():
    session = ChatSession(FakeEngine(), state_loader=lambda path: {0: str(path)})
    assert session.handle("你好").payload["text"] == "baseline"

    loaded = session.handle('/state "states/neko state.npz"')
    assert "已加载" in loaded.lines[0]
    assert session.state_label == "states/neko state.npz"
    assert session.handle("你好").payload["text"] == "tuned"

    session.handle("/state off")
    assert session.state is None
    assert session.handle("你好").payload["text"] == "baseline"


def test_chat_ab_toggle_and_output():
    session = ChatSession(FakeEngine(), state={0: "state"}, state_label="s.npz")
    assert session.handle("/ab on").lines == ["A/B: on"]
    reply = session.handle("测试")
    assert reply.payload["with_state"]["text"] == "tuned"
    assert reply.payload["baseline"]["text"] == "baseline"
    assert "=== 有 state ===" in reply.lines


def test_chat_runtime_sampling_configuration():
    session = ChatSession(FakeEngine())
    session.handle("/temperature 0.6")
    session.handle("/top-p 0.7")
    session.handle("/max-tokens 200")
    session.handle("/seed 7")
    assert session.config == GenerationConfig(
        max_tokens=200,
        temperature=0.6,
        top_p=0.7,
        seed=7,
        stop_sequences=NEKO_QA.inference_stop_sequences,
    )


def test_chat_runtime_penalty_configuration():
    """/presence /frequency /penalty-decay 运行中调重复惩罚(ChatRWKV 官方语义)。"""
    session = ChatSession(FakeEngine())
    session.handle("/presence 0.5")
    session.handle("/frequency 0.3")
    session.handle("/penalty-decay 0.99")
    # 只断言重复惩罚三参被正确设置(其他默认值见 GenerationConfig)。
    assert session.config.presence_penalty == 0.5
    assert session.config.frequency_penalty == 0.3
    assert session.config.penalty_decay == 0.99
    assert session.config.stop_sequences == NEKO_QA.inference_stop_sequences


def test_penalty_decay_rejects_out_of_range():
    """/penalty-decay 范围 (0, 1](strict_min + maximum=1)。"""
    session = ChatSession(FakeEngine())
    assert "超出范围" in session.handle("/penalty-decay 0").lines[0]
    assert "超出范围" in session.handle("/penalty-decay 1.5").lines[0]


def test_chat_failed_state_load_preserves_current_state():
    def fail(path: Path):
        raise ValueError("bad state")

    original = {0: "old"}
    session = ChatSession(
        FakeEngine(), state=original, state_label="old.npz", state_loader=fail
    )
    reply = session.handle("/state bad.npz")
    assert "加载失败" in reply.lines[0]
    assert session.state is original
    assert session.state_label == "old.npz"


def test_chat_quit_and_help():
    session = ChatSession(FakeEngine())
    assert session.handle("/quit").exit is True
    assert any("/state PATH" in line for line in session.handle("/help").lines)


def test_chat_streams_without_returning_duplicate_text():
    session = ChatSession(FakeEngine(), state={0: "state"})
    chunks = []
    reply = session.handle("你好", on_text=chunks.append)
    assert "".join(chunks) == "tuned"
    assert reply.lines == [
        "[stop=eos, tokens=1, 0.10s | Prompt: 0.0 t/s | Generation: 0.0 t/s]"
    ]


# ────────────────────────────────────────────────────────────────
# Phase 3 §2: ChatSession 多轮 — 续传/重放/rewind/clear/state
# ────────────────────────────────────────────────────────────────


def test_qa_multiturn_uses_continuation_after_clean_turn():
    """§2.6.a: 纯 qa(eos 干净),第二轮续传——engine 收到上一轮的 cache。"""
    engine = FakeEngine()
    session = ChatSession(engine, template="qa", state={0: "s"})
    session.handle("Q1")
    # 第一轮 cache=None(新建)
    assert engine.calls[0]["cache"] is FakeEngine._sentinel or engine.calls[0]["cache"] is None
    session.handle("Q2")
    # 第二轮:续传,应传入第一轮产出的 cache
    assert engine.calls[1]["cache"] is not None
    assert engine.calls[1]["cache"] is not FakeEngine._sentinel


def test_reasoning_multiturn_always_replays():
    """§2.6.a(修订): reasoning(任意 think 档)每轮走重放——cache=None。

    g1g 多轮实测(docs §8.4)证明 reasoning 续传偏离训练分布,
    裁决:所有 reasoning 组合全量走重放,continuation_safe = (qa and not reasoning)。
    """
    for think in ("off", "fast", "on"):
        engine = FakeEngine()
        session = ChatSession(engine, template="qa", reasoning=True, think=think, state={0: "s"})
        session.handle("Q1")
        session.handle("Q2")
        session.handle("Q3")
        # 每一轮都应是 cache=None(重放,不续传)
        for call in engine.calls:
            assert call["cache"] is None or call["cache"] is FakeEngine._sentinel, (
                f"reasoning think={think} 必须走重放(cache=None),第{engine.calls.index(call)}轮违反"
            )


def test_dirty_cache_triggers_replay_next_turn():
    """§2.6.b: stop_sequence 停止后 cache 脏,下一轮自动重放(cache=None)。"""
    engine = FakeEngine()
    session = ChatSession(engine, template="qa", state={0: "s"})

    # 构造一个 stop_sequence 停止的轮次(让 result.cache_clean=False)
    original_generate = engine.generate

    def dirty_generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        # 模拟 stop_sequence 路径:把返回的 result 标脏
        from dataclasses import replace
        return replace(result, cache_clean=False, stop_reason="stop_sequence")

    engine.generate = dirty_generate
    session.handle("Q1")  # 这一轮标脏
    engine.generate = original_generate  # 恢复

    session.handle("Q2")  # 这一轮应重放
    # 第二轮 cache 应是 None(重放,因为上一轮脏)
    assert engine.calls[1]["cache"] is None or engine.calls[1]["cache"] is FakeEngine._sentinel


def test_rewind_truncates_history_and_replays():
    """§2.6.c: /rewind [n] 截断最后 n 轮,触发重放。

    history 按 [user, assistant, ...] 分别记录,每轮 = 2 个 turn。
    """
    engine = FakeEngine()
    session = ChatSession(engine, template="qa", state={0: "s"})
    session.handle("Q1")
    session.handle("Q2")
    session.handle("Q3")
    # 3 轮 = 6 个 turn(user/assistant 各 3)
    assert len(session.history) == 6

    reply = session.handle("/rewind")  # 默认撤销 1 轮
    assert len(session.history) == 4  # 撤 1 轮 = 删 2 turn → 4
    # 下一轮:history 被改 → 重放
    engine.calls.clear()
    session.handle("Q4")
    # 重放:cache=None
    assert engine.calls[0]["cache"] is None or engine.calls[0]["cache"] is FakeEngine._sentinel

    # /rewind 2 撤销多轮:还剩 2 轮(4 turn + 这次 Q4 的 2 turn = 6,撤 2 轮 = 删 4 turn → 2)
    reply2 = session.handle("/rewind 2")
    assert len(session.history) == 2


def test_rewind_beyond_history_clamps():
    """/rewind n 超过 history 长度时 clamp 到 0(不报错)。"""
    session = ChatSession(FakeEngine(), template="qa", state={0: "s"})
    session.handle("Q1")  # 1 轮 = 2 turn
    reply = session.handle("/rewind 5")  # 超过轮数
    assert len(session.history) == 0
    assert "已撤销" in reply.lines[0] or "无历史" in reply.lines[0]


def test_clear_resets_history_and_cache():
    """§2.4: /clear 获得真实语义——清空 history、丢弃 cache、回到 S₀。"""
    engine = FakeEngine()
    session = ChatSession(engine, template="qa", state={0: "s"})
    session.handle("Q1")
    session.handle("Q2")
    assert len(session.history) == 4  # 2 轮 = 4 turn
    assert session.cache is not None

    reply = session.handle("/clear")
    assert len(session.history) == 0
    assert session.cache is None
    assert session.cache_clean is True

    # 下一轮从零开始(cache=None)
    engine.calls.clear()
    session.handle("Q3")
    assert engine.calls[0]["cache"] is None or engine.calls[0]["cache"] is FakeEngine._sentinel


def test_state_switch_midway_resets_session():
    """§2.4: /state PATH 在多轮中途切换 state → 清空会话(换 S₀ = 换人设)。"""
    engine = FakeEngine()
    session = ChatSession(engine, template="qa", state={0: "old"}, state_label="old.npz")
    session.handle("Q1")
    session.handle("Q2")
    assert len(session.history) == 4

    reply = session.handle('/state "new.npz"')
    # 切换 state 应清空会话
    assert len(session.history) == 0
    assert session.cache is None
    assert "重置" in reply.lines[0] or "已加载" in reply.lines[0]
    assert session.state_label == "new.npz"


def test_history_records_turns():
    """history 持有 Turn 列表(role + display 文本),按 user/assistant 分别记录。"""
    engine = FakeEngine()
    session = ChatSession(engine, template="qa", state={0: "s"})
    session.handle("Q1")
    session.handle("Q2")
    # 每轮 = [user, assistant] 两个 turn,2 轮 = 4 turn
    assert len(session.history) == 4
    roles = [t.role for t in session.history]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_continuation_prefix_used_on_second_turn_qa():
    """§2.4: qa 多轮第二轮 prompt = 只有 continuation 胶水(不含上一轮回答)。

    上一轮回答的 token 已在生成循环里逐个喂入、固化进 cache。续传 prompt
    只需 "\\n\\nUser: {q}\\n\\nAssistant:",再 prefill 进现有 cache。
    若把上一轮回答也拼进 prompt,模型会"看到两遍回答",上下文错乱。
    """
    engine = FakeEngine()
    session = ChatSession(engine, template="qa", state={0: "s"})
    session.handle("Q1")
    session.handle("Q2")
    # 第二轮的 prompt(传给 engine.generate 的)
    turn2_prompt = engine.calls[1]["prompt"]
    # 关键: prompt 只是胶水,不含上一轮回答(FakeEngine 返回 'tuned')
    assert "tuned" not in turn2_prompt
    assert turn2_prompt == "\n\nUser: Q2\n\nAssistant:"


# ────────────────────────────────────────────────────────────────
# think=on 显示清洗:_display_text 剥 think 段(history 重放安全)
# ────────────────────────────────────────────────────────────────


def _make_bare_session(**kwargs):
    """造一个不触发 engine 的 ChatSession(只测 _display_text 展示清洗)。

    _display_text 只依赖 self.reasoning / self.think / self.template,
    无需真实 engine 或完整 __init__。
    """
    s = ChatSession.__new__(ChatSession)
    s.template = kwargs.get("template", "qa")
    s.reasoning = kwargs.get("reasoning", False)
    s.think = kwargs.get("think", "off")
    return s


def test_display_text_think_on_strips_thinking_section():
    """think=on 时 _display_text 只保留 </think> 之后的 answer。

    品类铁律(docs §8.4):reasoning 模型重放时历史 assistant 必须是裸 answer,
    think 标签只在当前生成轮。这里验证进 history 的文本已剥 think。
    """
    s = _make_bare_session(reasoning=True, think="on")
    assert s._display_text("思考内容</think>正式回答") == "正式回答"


def test_display_text_think_on_strips_leading_newlines_after_close():
    """think=on 的 answer 段也去掉 </think> 后的前导换行(reasoning 方言一致性)。"""
    s = _make_bare_session(reasoning=True, think="on")
    assert s._display_text("思考</think>\n\n正式回答") == "正式回答"


def test_display_text_think_on_no_close_tag_returns_empty_answer():
    """think=on 但无 </think>(max_tokens 截断等)→ 空 answer(思考未完成,无有效回答)。

    兜底语义:半截思考不该当历史回答重放(品类铁律 + 重放安全)。
    展示层(cli.py)会单独把 raw 文本的思考段 dim 显示 + 标注截断。
    """
    s = _make_bare_session(reasoning=True, think="on")
    assert s._display_text("未闭合的半截思考") == ""


def test_display_text_think_fast_preserves_lstrip_newline():
    """回归保护:think=fast 仍走 lstrip('\\n')(空 think 标签后的自然换行)。

    确保加 think=on 分支没改坏 fast 档的既有清洗逻辑。
    """
    s = _make_bare_session(reasoning=True, think="fast")
    assert s._display_text("\n\n带前导换行的回答") == "带前导换行的回答"


def test_display_text_think_off_passthrough():
    """回归保护:非 reasoning + think=off 原样返回(不受新分支影响)。"""
    s = _make_bare_session(reasoning=False, think="off")
    assert s._display_text("原样文本") == "原样文本"


def test_display_text_think_on_end_to_end_in_history():
    """端到端:think=on 会话,assistant turn 进 history 的是纯 answer。

    用真实 ChatSession + FakeEngine,把 FakeEngine 返回值改成含 </think> 的文本,
    验证 session.history 里的 assistant turn.text 不含 think 段。
    """
    engine = FakeEngine()
    # 让 FakeEngine.generate 返回含 think 段的文本
    original_generate = engine.generate

    def think_generate(prompt, *, state=None, config=None, cache=FakeEngine._sentinel, on_text=None):
        result = original_generate(
            prompt, state=state, config=config, cache=cache, on_text=on_text
        )
        from dataclasses import replace as _replace
        # 覆盖 text 为 think=on 形态的输出
        return _replace(result, text="我的思考过程</think>可见的回答")

    engine.generate = think_generate
    session = ChatSession(engine, template="qa", reasoning=True, think="on", state={0: "s"})
    session.handle("Q1")
    # assistant turn(第 2 个)的 text 应只含 answer,不含 think 段
    assistant_turn = session.history[1]
    assert assistant_turn.role == "assistant"
    assert assistant_turn.text == "可见的回答"
    assert "我的思考过程" not in assistant_turn.text
    assert "</think>" not in assistant_turn.text
