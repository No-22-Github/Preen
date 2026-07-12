"""展示层(console.py)测试 — 不依赖真实模型。

验证:渲染对象不抛异常、表格列宽/方括号转义、配置分组结构正确。
"""
import io

import pytest
from rich.console import Console

from statetuner import console
from statetuner.chat import ChatSession
from statetuner.inference import GenerationConfig


def _render_to_plain(renderable, width: int = 100) -> str:
    """把任意 rich renderable 渲染成纯文本(非 TTY,无 ANSI)。"""
    buf = io.StringIO()
    c = Console(file=buf, force_terminal=False, width=width)
    c.print(renderable)
    return buf.getvalue()


def _make_session(**kwargs) -> ChatSession:
    """造一个不触发 engine 的 ChatSession(只测展示辅助方法)。"""
    s = ChatSession.__new__(ChatSession)
    s.config = GenerationConfig()
    s.template = kwargs.get("template", "qa")
    s.reasoning = kwargs.get("reasoning", False)
    s.think = kwargs.get("think", "off")
    s.state = kwargs.get("state", None)
    s.state_label = kwargs.get("state_label", None)
    s.ab = kwargs.get("ab", False)
    return s


# ── make_console ──────────────────────────────────────────────


def test_make_console_returns_console():
    c = console.make_console()
    assert isinstance(c, Console)


def test_render_to_plain_has_no_ansi_escape():
    """非 TTY 渲染不含 ANSI 转义码(测试/管道友好)。"""
    out = _render_to_plain(console.render_markdown("# 标题"))
    assert "\x1b[" not in out
    assert "标题" in out


# ── config_groups / render_config_table ───────────────────────


def test_config_groups_three_sections_with_expected_fields():
    s = _make_session()
    groups = s.config_groups()
    group_names = [name for name, _ in groups]
    assert group_names == ["模板", "采样", "重复惩罚"]
    # 重复惩罚组应标注 ChatRWKV 官方默认值
    penalty_fields = dict(groups[2][1])
    assert "ChatRWKV 默认 0.4" in penalty_fields["presence"]
    assert "ChatRWKV 默认 0.996" in penalty_fields["decay"]


def test_config_groups_brief_strips_default_annotations():
    """brief=True 去掉默认值注解,用于启动横幅紧凑显示。"""
    s = _make_session()
    groups = s.config_groups(brief=True)
    penalty_fields = dict(groups[2][1])
    assert "ChatRWKV" not in penalty_fields["presence"]
    assert penalty_fields["presence"] == "0.4"


def test_config_compact_renders_two_lines():
    """紧凑配置(启动横幅)渲染成两行,含采样+重复惩罚字段。"""
    s = _make_session()
    out = _render_to_plain(console.render_config_compact(s.config_groups(brief=True)))
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == 2  # 模板+采样一行,state+重复惩罚一行
    joined = " ".join(lines)
    for field in ("template", "temp", "top_p", "presence", "frequency"):
        assert field in joined


def test_config_table_renders_all_field_names():
    s = _make_session()
    out = _render_to_plain(console.render_config_table(s.config_groups()))
    for field in ("template", "reasoning", "temperature", "presence", "frequency"):
        assert field in out


def test_config_groups_reflect_reasoning_state():
    s = _make_session(reasoning=True, think="fast")
    groups = dict((name, dict(fields)) for name, fields in s.config_groups())
    assert groups["模板"]["reasoning"] == "on (think=fast)"


# ── help_lines / render_help_table ────────────────────────────


def test_help_lines_contains_penaly_decay_command():
    """重复惩罚档位说明应包含 /penalty-decay(新补全的命令)。"""
    lines = ChatSession.help_lines()
    assert any(line.startswith("/penalty-decay") for line in lines)


def test_help_table_preserves_bracket_args():
    """/ab [on|off] 的方括号不能被 rich markup 吃掉(回归保护)。"""
    out = _render_to_plain(console.render_help_table(ChatSession.help_lines()))
    assert "/ab [on|off]" in out
    assert "/state PATH" in out
    assert "/penalty-decay N" in out


def test_help_table_returns_none_for_no_command_lines():
    """无任何 / 开头行的 help_lines → render_help_table 返回 None(原样透传)。"""
    assert console.render_help_table(["纯说明文字", "无命令"]) is None


def test_help_table_returns_none_for_empty():
    assert console.render_help_table([]) is None


# ── render_assistant_panel / dim_summary / user_prompt_label ──


def test_assistant_panel_has_rwkv_title():
    out = _render_to_plain(console.render_assistant_panel("回复内容"))
    assert "RWKV" in out
    assert "回复内容" in out


def test_assistant_panel_renders_markdown_code_fence():
    md = '代码:\n\n```python\nprint("hi")\n```\n'
    out = _render_to_plain(console.render_assistant_panel(md))
    assert "print" in out  # 代码块内容应出现


def test_dim_summary_is_text_object():
    t = console.dim_summary("[stop=eos, tokens=5]")
    assert hasattr(t, "plain")
    assert t.plain == "[stop=eos, tokens=5]"


def test_user_prompt_label_is_you():
    t = console.user_prompt_label()
    assert t.plain == "You> "


# ── _render_reply_lines (cli 渲染层) ──────────────────────────


def test_render_reply_lines_handles_all_line_kinds():
    """斜杠命令回包的三类行都不抛异常,正文走 markdown。

    回归:曾因 ui 未定义(只在 chat() 局部 import)导致 NameError,
    在 /max-tokens 等斜杠命令后崩溃。
    """
    from statetuner.cli import _render_reply_lines

    ui_console = console.make_console()
    buf = io.StringIO()
    ui_console.file = buf  # 捕获输出

    lines = [
        "=== 有 state ===",
        "这是助手回复的**正文**。",
    ]
    _render_reply_lines(ui_console, lines)
    out = buf.getvalue()

    assert "正文" in out  # markdown 正文被渲染
    assert "有 state" in out  # === 分隔行原样打印
    assert "\x1b[" not in out  # 非 TTY 无 ANSI


# ── split_thinking / render_thinking_panel (think=on 可视化) ──


def test_split_thinking_normal_case():
    """有 </think> → 拆两段,thinking 去首尾空白。"""
    thinking, answer = console.split_thinking("思考过程</think>正式回答")
    assert thinking == "思考过程"
    assert answer == "正式回答"


def test_split_thinking_strips_residual_gt_from_tag_completion():
    """think=on 的 prompt 以 ``<think`` 结尾,模型补全成 ``<think>``,raw 输出
    常以 ``>\\n`` 开头(标签闭合残余)。这个 > 不是思考内容,要清掉。

    只删开头 > + 紧跟换行的固定形态,不误伤思考正文里合法的 >。
    """
    thinking, answer = console.split_thinking(">\n好的，用户说早上好</think>早安")
    assert thinking == "好的，用户说早上好"
    assert answer == "早安"
    # 正文里的 > 不误删
    thinking2, _ = console.split_thinking("比较 a > b 的大小</think>ans")
    assert "> b" in thinking2


def test_split_thinking_strips_whitespace_around_thinking():
    """thinking 段首尾的自然换行/空白都被 strip(</think> 前的噪音)。"""
    thinking, answer = console.split_thinking("  思考  </think>ans")
    assert thinking == "思考"
    assert answer == "ans"


def test_split_thinking_no_close_tag_treats_all_as_thinking():
    """无 </think>(max_tokens 截断等)→ 已生成内容当 thinking,answer=""(兜底)。

    语义:未闭合 = 思考没写完,模型还在 think 段里。展示层据此 dim 显示思考 +
    标注截断;history 层据此存空 answer(半截思考不该当历史回答重放)。
    """
    thinking, answer = console.split_thinking("未写完的思考内容")
    assert thinking == "未写完的思考内容"
    assert answer == ""


def test_split_thinking_empty_thinking_section():
    """</think> 紧开头(空思考)→ thinking="",answer=剩余。"""
    thinking, answer = console.split_thinking("</think>直接回答")
    assert thinking == ""
    assert answer == "直接回答"


def test_split_thinking_preserves_answer_leading_newlines():
    """answer 原样(含 </think> 后的自然换行),由调用方按需清洗。"""
    thinking, answer = console.split_thinking("思考\n\n</think>\n\n回答")
    assert thinking == "思考"
    assert answer == "\n\n回答"


def test_render_thinking_panel_has_title_and_content():
    """思考过程面板渲染不抛、标题含「Thinking」、内容出现。"""
    out = _render_to_plain(console.render_thinking_panel("我在想..."))
    assert "Thinking" in out  # 标题
    assert "我在想" in out  # 内容


def test_render_thinking_panel_is_not_markdown():
    """思考段走纯 Text(非 markdown):思考里的 #/* 不被当标题/加粗解析。"""
    out = _render_to_plain(console.render_thinking_panel("# 这不是标题 *也不是加粗*"))
    # 非 markdown 渲染:# 和 * 原样出现在输出里(markdown 会渲染成标题/加粗文本)
    assert "#" in out
    assert "*" in out
