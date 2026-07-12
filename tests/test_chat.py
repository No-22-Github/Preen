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
