"""thinking.py UI 中立层拆分逻辑测试(T1 单一事实源)。

验证 split_thinking / classify_phase 的行为,以及与 console 包装的一致性
(chat._display_text 旧 rfind vs console.find 不一致的 bug 已统一)。
"""
from __future__ import annotations

from statetuner.thinking import classify_phase, split_thinking


# ── split_thinking ──────────────────────────────────────────


def test_split_normal_case():
    thinking, answer = split_thinking("思考过程</think>正式回答")
    assert thinking == "思考过程"
    assert answer == "正式回答"


def test_split_strips_residual_gt_from_tag_completion():
    """think=on prompt 以 <think 结尾,模型续写补全成 <think>,raw 常以 >\\n 开头。

    这个 > 不是思考内容,清掉。只删开头 > + 紧跟换行的固定形态。
    """
    thinking, answer = split_thinking(">\n思考过程</think>回答")
    assert thinking == "思考过程"
    assert answer == "回答"


def test_split_no_close_returns_empty_answer():
    """未闭合(max_tokens 截断)→ thinking=已生成内容,answer 空。"""
    thinking, answer = split_thinking("未闭合的半截思考")
    assert thinking == "未闭合的半截思考"
    assert answer == ""


def test_split_empty_think_tag():
    """</think> 紧开头(空思考)→ thinking 空,answer=剩余。"""
    thinking, answer = split_thinking("</think>直接回答")
    assert thinking == ""
    assert answer == "直接回答"


def test_split_strips_thinking_whitespace():
    """thinking 段 strip(前后空白)。"""
    thinking, answer = split_thinking("  思考  </think>回答")
    assert thinking == "思考"
    assert answer == "回答"


def test_split_uses_find_not_rfind():
    """T1 关键:统一用 find(取第一个 </think>),不是 rfind。

    reasoning 模型 think 段内部不会合法出现闭合标签(那是边界),
    第一个就是真正的边界。旧 chat._display_text 用 rfind 会因 answer 里
    偶然包含 </think> 字面(如讨论标签的问答)而误判。
    """
    raw = "思考</think>回答里提到了 </think> 这个标签"
    thinking, answer = split_thinking(raw)
    # find:第一个 </think> 是边界 → answer 含第二个字面 </think>
    assert thinking == "思考"
    assert "回答里提到了" in answer
    assert "</think>" in answer  # answer 里保留第二个(字面)


# ── classify_phase(T1: text_chunk.phase 字段)──────────────


def test_classify_phase_think_before_close():
    """未越过 </think> → 'think'。"""
    assert classify_phase("还没闭合的思考") == "think"
    assert classify_phase("思考过程") == "think"


def test_classify_phase_answer_after_close():
    """已含 </think> → 'answer'。"""
    assert classify_phase("思考</think>回答") == "answer"
    assert classify_phase("思考</think>回答继续") == "answer"


def test_classify_phase_empty_is_think():
    """空累积 → think(默认,流式起始)。"""
    assert classify_phase("") == "think"


# ── console 包装一致性(T1:console.split_thinking 是 thin wrapper)────


def test_console_wrapper_matches_thinking_module():
    """console.split_thinking 应与 thinking.split_thinking 行为一致。"""
    from statetuner.console import split_thinking as console_split

    cases = [
        "思考</think>回答",
        "未闭合",
        "</think>直接回答",
        ">\n思考</think>回答",
        "",
    ]
    for raw in cases:
        assert console_split(raw) == split_thinking(raw), (
            f"console 与 thinking 模块不一致 on {raw!r}"
        )
