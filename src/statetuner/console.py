"""对话展示层 helper — 集中所有终端渲染逻辑。

这是 chat 命令的"皮肤层":ChatSession(chat.py)保持终端无关的纯逻辑,
所有 rich 渲染(panel / markdown / dim 摘要 / 配置表格)收口到这里,
cli.py 只负责调用。未来 SwiftUI 接管时,本文件整层废弃,ChatSession 不动。

风格定位("舒适平衡"):
  - 标签(You/RWKV)淡色,正文单色 → 长输出不花眼。
  - 助手回复用 markdown 渲染(代码块/列表/加粗),但只在整段就绪后一次性渲染,
    流式中途走纯文本增量(避免每 token 重渲染整篇 markdown 的闪烁)。
  - 技术摘要行(stop/tokens/tps)用 dim 灰,不抢内容。
"""
from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# 舒适平衡配色:标签用低饱和色,不刺眼。
USER_LABEL_COLOR = "cyan"
ASSISTANT_PANEL_BORDER = "magenta"


def make_console(*, stderr: bool = False) -> Console:
    """返回渲染用 Console。

    自动适配环境:
      - TTY:正常彩色。
      - 非 TTY(管道/重定向/测试):rich 默认不发 ANSI,输出纯文本。
      - NO_COLOR 环境变量:rich 原生尊重(https://no-color.org)。
    不 force_terminal,让 rich 自己判,保证重定向/CI 干净。
    """
    return Console(stderr=stderr)


def render_markdown(text: str) -> Markdown:
    """把助手回复渲染成 Markdown 对象。

    在整段文本就绪后调用(非流式路径,或流式结束后的一次性渲染)。
    code_lexer 不写死,让 rich 按 ``` 围栏自动判。
    """
    return Markdown(text)


def render_assistant_panel(text: str) -> Panel:
    """助手回复 markdown 面板(淡紫边框,标题 RWKV)。"""
    return Panel(
        render_markdown(text),
        border_style=ASSISTANT_PANEL_BORDER,
        title="[magenta]RWKV[/magenta]",
        title_align="left",
        padding=(0, 1),
    )


THINK_CLOSE = "</think>"


def split_thinking(raw: str) -> tuple[str, str]:
    """把 think=on 的原始输出拆成 (thinking, answer)。

    think=on 时 prompt 以 `` <think`` 结尾(开标签在 prompt 里),模型续写出
    ``思考内容</think>回答``。**开标签不在模型输出里**,所以这里只认 ``</think>``:
      - 找到 ``</think>`` → 前段(去首尾空白)= thinking,后段 = answer
      - 找不到(max_tokens 截断未闭合等)→ 已生成内容当 thinking(它确实是
        未写完的思考),answer=""。展示层据此 dim 显示思考 + 标注被截断;
        history 层据此存空 answer(半截思考不该当历史回答重放)。
      - ``</think>`` 紧开头(空思考)→ thinking="",answer=剩余

    返回的 thinking 已 strip(前后的自然换行/空白);answer 原样,由调用方按
    需要再做模板相关清洗(chat._display_text 的 lstrip('\\n'))。

    仅对 think=on 有意义;off/fast 调用方不会进来。
    """
    idx = raw.find(THINK_CLOSE)
    if idx < 0:
        # 未闭合:已生成内容是未写完的思考,answer 为空。
        return _clean_thinking(raw), ""
    return _clean_thinking(raw[:idx]), raw[idx + len(THINK_CLOSE):]


def _clean_thinking(text: str) -> str:
    """清洗 think 段文本:strip + 去掉开头的残余 ``>``。

    think=on 的 prompt 以 ``<think`` 结尾,模型续写时把它补全成 ``<think>``,
    所以 raw 输出常以 ``>\\n`` 开头(标签闭合的机械副产品)。这个 ``>`` 不是
    思考内容,显示出来突兀,清掉。只删开头 ``>`` + 紧跟换行的固定形态,
    不影响思考正文里合法的 ``>``。
    """
    cleaned = text.strip()
    if cleaned.startswith(">\n"):
        cleaned = cleaned[2:].lstrip("\n")
    elif cleaned == ">":
        cleaned = ""
    return cleaned


def render_thinking_panel(text: str) -> Panel:
    """思考过程面板(dim 灰边框,标题 Thinking)。

    think=on 流式结束后渲染,与 render_assistant_panel 并列显示。
    思考段走纯 Text(非 markdown):思考多为意识流长段落,markdown 渲染反而
    干扰(把思考里的 #/* 当标题/加粗),dim 纯文本更接近「草稿」的视觉语义。
    """
    return Panel(
        Text(text, style="dim italic"),
        border_style="dim",
        title="[dim]Thinking[/dim]",
        title_align="left",
        padding=(0, 1),
    )


def dim_summary(summary_line: str) -> Text:
    """技术摘要行(stop/tokens/tps)用 dim 灰,不抢正文。"""
    return Text(summary_line, style="dim")


def user_prompt_label() -> Text:
    """You> 提示标签(淡青)。"""
    return Text("You> ", style=USER_LABEL_COLOR)


def render_config_table(config_groups: list[tuple[str, list[tuple[str, str]]]]) -> Table:
    """渲染调参面板为对齐表格。

    Args:
      config_groups: [(组名, [(字段名, 值), ...]), ...]
        例 [("模板", [("template", "qa"), ("state", "off")]),
            ("采样", [("temperature", "0.6"), ...])]

    布局:左列 dim 标签,右列正常值;组分隔靠空行。
        用 Text 对象避免值里的方括号被 rich 当 markup 吃掉。
    """
    # 测量字段名列最大宽度(grid 的 no_wrap 自动测宽对方括号不安全)。
    label_width = max(
        (len(name) for _, fields in config_groups for name, _ in fields),
        default=0,
    )
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", no_wrap=True, width=label_width, overflow="fold")
    table.add_column(overflow="fold")
    for group_idx, (_group_name, fields) in enumerate(config_groups):
        if group_idx > 0:
            table.add_row("", "")  # 组间空行分隔
        for name, value in fields:
            table.add_row(Text(name, no_wrap=True), Text(value))
    return table


# 启动横幅用的字段别名映射(缩短显示;不影响 config_groups 的真实字段名)。
_CONFIG_DISPLAY_ALIASES = {
    "temperature": "temp",
    "max_tokens": "max",
    "presence_penalty": "presence",
    "frequency_penalty": "frequency",
    "penalty_decay": "decay",
}
_SEP = ("  ·  ", "dim")


def render_config_compact(config_groups: list[tuple[str, list[tuple[str, str]]]]) -> Text:
    """启动横幅用的紧凑配置显示(两行,dim 标签 + 正常值,· 分隔)。

    相比 render_config_table 的 11 行表格,这个压成 2 行,启动时更清爽。
    /config 命令仍走 render_config_table(详尽模式)。
    """
    flat: list[tuple[str, str]] = []
    for _, fields in config_groups:
        flat.extend(fields)
    # 按语义重排成两行:第一行模板+采样,第二行 state/ab+重复惩罚
    first_keys = ("template", "reasoning", "temperature", "top_p", "max_tokens", "seed")
    field_map = dict(flat)
    line1_keys = [k for k in first_keys if k in field_map]
    line2_keys = [k for k, _ in flat if k not in first_keys]

    def _build(keys: list[str]) -> Text:
        parts: list = []
        for i, k in enumerate(keys):
            alias = _CONFIG_DISPLAY_ALIASES.get(k, k)
            parts.append((f"{alias} ", "dim"))
            parts.append((field_map[k], "white"))
            if i < len(keys) - 1:
                parts.append(_SEP)
        return Text.assemble(*parts)

    combined = Text()
    combined.append_text(_build(line1_keys))
    combined.append("\n")
    combined.append_text(_build(line2_keys))
    return combined


def render_help_table(help_lines: list[str]) -> Optional[Table]:
    """渲染 help 命令列表为两列对齐表格(命令 | 说明)。

    help_lines 格式:"/cmd ARGS   说明文字"(空格分隔,首个多空格为分界)。
    解析失败或格式不符的行原样透传(交给调用方 echo)。
    """
    if not help_lines:
        return None
    import re

    # 先解析所有行,测量命令列最大宽度,显式设列宽(grid 的 no_wrap 自动测宽不可靠)。
    rows: list[tuple[str, str]] = []
    parsed_any = False
    for line in help_lines:
        m = re.split(r" {2,}", line.strip(), maxsplit=1)
        if len(m) == 2 and m[0].startswith("/"):
            rows.append((m[0], m[1]))
            parsed_any = True
        else:
            # 不符合"命令  说明"格式的行(如分组标题/空行)用占位保持对齐
            rows.append(("", line))
    if not parsed_any:
        return None
    cmd_width = max((len(cmd) for cmd, _ in rows), default=0)
    table = Table.grid(padding=(0, 2))
    # markup=False:命令含方括号(如 /ab [on|off])不能被当 rich 标记吃掉。
    table.add_column(
        style="bold cyan", no_wrap=True, width=cmd_width, overflow="fold"
    )
    table.add_column(overflow="fold")
    for cmd, desc in rows:
        # 命令列禁用 markup([on|off] 不能当样式标记);说明列保持默认。
        table.add_row(Text(cmd, no_wrap=True), desc)
    return table
