from pathlib import Path

from statetuner.chat import ChatSession
from statetuner.inference import ABResult, GenerationConfig, GenerationResult
from statetuner.templates import QA as NEKO_QA  # tests 局部别名


class FakeEngine:
    def load_state(self, state):
        return {0: str(state)}

    def generate(self, prompt, *, state, config, on_text=None):
        text = "tuned" if state is not None else "baseline"
        if on_text is not None:
            on_text(text)
        return GenerationResult(
            text=text,
            token_ids=[1],
            stop_reason="eos",
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
