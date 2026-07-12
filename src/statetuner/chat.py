"""模型常驻的交互会话控制器；不感知 Typer 和具体终端。"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

from .core import StateInput
from .inference import (
    GenerationConfig,
    InferenceEngine,
    render_prompt,
    with_template_stops,
)

StateLoader = Callable[[Path], StateInput]
TextCallback = Callable[[str], None]


@dataclass(frozen=True)
class ChatReply:
    lines: list[str]
    exit: bool = False
    payload: Optional[dict] = None


class ChatSession:
    """每轮 fresh cache；模型常驻，state 可在轮次之间动态替换。"""

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        config: Optional[GenerationConfig] = None,
        template: str = "qa",
        reasoning: bool = False,
        think: str = "off",
        state: StateInput = None,
        state_label: Optional[str] = None,
        state_loader: Optional[StateLoader] = None,
        ab: bool = False,
    ):
        self.engine = engine
        base_config = config or GenerationConfig(max_tokens=200, temperature=0.8)
        self.config = with_template_stops(base_config, template)
        self.config.validate()
        self.template = template
        self.reasoning = reasoning
        self.think = think
        self.state = state
        self.state_label = state_label
        self.state_loader = state_loader or engine.load_state
        self.ab = ab

    @property
    def _has_reasoning_dialect(self) -> bool:
        """是否需要 reasoning 方言的显示清洗(前导 \\n 是空 think 标签后的自然换行)。

        旧实现硬判某旧模板字符串;新世界 reasoning 是正交开关,
        且仅与 qa 模板组合(instruction/raw 不与 reasoning 组合,由 render_prompt 拒绝)。
        """
        return self.reasoning and self.think in ("fast", "on") and self.template != "raw"

    def handle(
        self, text: str, *, on_text: Optional[TextCallback] = None
    ) -> ChatReply:
        text = text.strip()
        if not text:
            return ChatReply([])
        if text.startswith("/"):
            return self._command(text)

        wrapped = render_prompt(
            text, self.template, reasoning=self.reasoning, think=self.think
        )
        if self.ab:
            if self.state is None:
                return ChatReply(["A/B 已开启，但当前没有 state；请先使用 /state PATH。"])
            result = self.engine.compare(wrapped, state=self.state, config=self.config)
            lines = [
                "=== 有 state ===",
                self._display_text(result.with_state.text),
                self._summary(result.with_state),
                "=== 无 state（基线）===",
                self._display_text(result.baseline.text),
                self._summary(result.baseline),
            ]
            return ChatReply(lines, payload=result.to_dict())

        # reasoning 方言输出开头常有 \n(</think> 后的自然换行)，stream 首个非空 chunk 去掉它。
        stream_callback = self._wrap_stream_callback(on_text) if on_text else None
        result = self.engine.generate(
            wrapped, state=self.state, config=self.config, on_text=stream_callback
        )
        if on_text is not None:
            return ChatReply([self._summary(result)], payload=result.to_dict())
        return ChatReply(
            [self._display_text(result.text), self._summary(result)],
            payload=result.to_dict(),
        )

    def _display_text(self, text: str) -> str:
        """模板相关的显示清洗。reasoning 方言去掉前导换行(</think> 后的自然换行)。"""
        if self._has_reasoning_dialect:
            return text.lstrip("\n")
        return text

    def _wrap_stream_callback(self, callback: TextCallback) -> TextCallback:
        """包装 stream 回调：首个非空 chunk 去掉模板前导换行。

        inference.generate 的 emit_safe_text 只发增量 delta，首个 delta 通常就是
        开头的 \\n（reasoning 方言空 think 标签后的自然换行）。这里一次性消费掉它，
        后续 chunk 原样透传。
        """
        if not self._has_reasoning_dialect:
            return callback
        stripped = False

        def _wrapped(chunk: str) -> None:
            nonlocal stripped
            if not stripped:
                chunk = chunk.lstrip("\n")
                stripped = True
                if not chunk:
                    return
            callback(chunk)

        return _wrapped

    @staticmethod
    def _summary(result) -> str:
        return result.summary_line()

    def _command(self, text: str) -> ChatReply:
        try:
            parts = shlex.split(text)
        except ValueError as exc:
            return ChatReply([f"命令解析失败: {exc}"])
        command = parts[0].lower()
        args = parts[1:]

        if command in ("/quit", "/exit"):
            return ChatReply(["会话结束。"], exit=True)
        if command == "/help":
            return ChatReply(self.help_lines())
        if command == "/clear":
            return ChatReply(["当前为独立问答模式；下一轮本来就会从当前 S₀ 重新开始。"])
        if command == "/state":
            return self._state_command(args)
        if command == "/ab":
            return self._toggle_ab(args)
        if command in ("/temperature", "/temp"):
            return self._set_float("temperature", args, minimum=0.0)
        if command == "/top-p":
            return self._set_float("top_p", args, minimum=0.0, maximum=1.0, strict_min=True)
        if command == "/max-tokens":
            return self._set_int("max_tokens", args, minimum=1)
        if command == "/seed":
            return self._set_int("seed", args)
        if command == "/config":
            return ChatReply([self.config_line()])
        return ChatReply([f"未知命令: {command}；输入 /help 查看帮助。"])

    def _state_command(self, args: list[str]) -> ChatReply:
        if not args:
            return ChatReply([f"当前 state: {self.state_label or 'off'}"])
        value = args[0]
        if value.lower() in ("off", "none", "zero"):
            self.state = None
            self.state_label = None
            return ChatReply(["state 已关闭；后续轮次使用零 state 基线。"])
        path = Path(value).expanduser()
        try:
            loaded = self.state_loader(path)
        except Exception as exc:
            return ChatReply([f"state 加载失败: {exc}"])
        self.state = loaded
        self.state_label = str(path)
        return ChatReply([f"state 已加载: {path}"])

    def _toggle_ab(self, args: list[str]) -> ChatReply:
        if args:
            value = args[0].lower()
            if value not in ("on", "off"):
                return ChatReply(["用法: /ab [on|off]"])
            self.ab = value == "on"
        else:
            self.ab = not self.ab
        return ChatReply([f"A/B: {'on' if self.ab else 'off'}"])

    def _set_float(
        self,
        field: str,
        args: list[str],
        *,
        minimum: float,
        maximum: Optional[float] = None,
        strict_min: bool = False,
    ) -> ChatReply:
        if len(args) != 1:
            return ChatReply([f"用法: /{field.replace('_', '-')} VALUE"])
        try:
            value = float(args[0])
        except ValueError:
            return ChatReply([f"{field} 必须是数字"])
        if value < minimum or (strict_min and value == minimum):
            return ChatReply([f"{field} 超出范围"])
        if maximum is not None and value > maximum:
            return ChatReply([f"{field} 超出范围"])
        self.config = replace(self.config, **{field: value})
        return ChatReply([self.config_line()])

    def _set_int(self, field: str, args: list[str], *, minimum: Optional[int] = None) -> ChatReply:
        if len(args) != 1:
            return ChatReply([f"用法: /{field.replace('_', '-')} VALUE"])
        try:
            value = int(args[0])
        except ValueError:
            return ChatReply([f"{field} 必须是整数"])
        if minimum is not None and value < minimum:
            return ChatReply([f"{field} 必须 >= {minimum}"])
        self.config = replace(self.config, **{field: value})
        return ChatReply([self.config_line()])

    def config_line(self) -> str:
        reasoning_part = (
            f" reasoning=on think={self.think}" if self.reasoning else ""
        )
        return (
            f"template={self.template}{reasoning_part} "
            f"state={self.state_label or 'off'} ab={'on' if self.ab else 'off'} "
            f"temperature={self.config.temperature} top_p={self.config.top_p} "
            f"max_tokens={self.config.max_tokens} seed={self.config.seed}"
        )

    @staticmethod
    def help_lines() -> list[str]:
        return [
            "/state PATH       动态加载 npz/pth，下一轮生效",
            "/state off        关闭 state，切回零 state 基线",
            "/state            查看当前 state",
            "/ab [on|off]      切换 A/B 输出",
            "/temperature N    调整温度（0=贪心）",
            "/top-p N          调整 top-p",
            "/max-tokens N     调整单轮生成上限",
            "/seed N           调整采样 seed",
            "/config           查看当前配置",
            "/clear            说明独立问答的 cache 语义",
            "/quit             退出",
        ]
