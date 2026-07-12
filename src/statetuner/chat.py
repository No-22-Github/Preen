"""模型常驻的交互会话控制器；不感知 Typer 和具体终端。

Phase 3 §2 多轮改造(2026-07-12):
  会话持有 history / cache / cache_clean,实现 state 续传(主路径)+ 历史重放(修复路径)。
  续传分级(docs/g1g-decode-alignment.md §8.4 实测 + 裁决):
    - 纯 qa(template=qa, reasoning=False):续传,cache 轮间保留。
    - reasoning(任意 think 档):全量重放(历史剥 think 是 reasoning 模型品类属性,
      续传会偏离训练分布)。continuation_safe = (template=="qa" and not reasoning)。
  脏 cache(stop_sequence 停止)、/rewind、/state 切换 → 触发重放。
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

from .core import StateInput
from .inference import (
    GenerationConfig,
    InferenceEngine,
    render_prompt,
    with_template_stops,
)
from .templates import QA

StateLoader = Callable[[Path], StateInput]
TextCallback = Callable[[str], None]


@dataclass(frozen=True)
class Turn:
    """单轮对话记录(role + display 文本)。

    display 文本是唯一事实源(§2.3):重放时以 history 的文本重新渲染编码,
    不用 fed_token_ids 拼接。token 只是审计记录。
    """

    role: str  # "user" / "assistant"
    text: str


@dataclass(frozen=True)
class ChatReply:
    lines: list[str]
    exit: bool = False
    payload: Optional[dict] = None


class ChatSession:
    """模型常驻的交互会话控制器;多轮 cache 续传 + 历史重放(Phase 3 §2)。

    续传/重放决策(§2.4 + docs/g1g-decode-alignment.md §8.4 裁决):
      continuation_safe = (template=="qa" and not reasoning)
      每轮 handle 时:若 continuation_safe 且 cache 干净且历史未被改 → 续传;
      否则 → 重放(从 S₀ + 完整 history 文本重新 prefill,cache=None)。
    """

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
        base_config = config or GenerationConfig(max_tokens=200, temperature=0.6, top_p=0.7)
        self.config = with_template_stops(base_config, template)
        self.config.validate()
        self.template = template
        self.reasoning = reasoning
        self.think = think
        self.state = state
        self.state_label = state_label
        self.state_loader = state_loader or engine.load_state
        self.ab = ab
        # Phase 3 §2 多轮状态
        self.history: list[Turn] = []
        self.cache: object = None       # running cache,None=需从 S₀ 新建(重放)
        self.cache_clean: bool = True   # 上一轮结束时 cache 是否干净可续传

    @property
    def _has_reasoning_dialect(self) -> bool:
        """是否需要 reasoning 方言的显示清洗(前导 \\n 是空 think 标签后的自然换行)。

        旧实现硬判某旧模板字符串;新世界 reasoning 是正交开关,
        且仅与 qa 模板组合(instruction/raw 不与 reasoning 组合,由 render_prompt 拒绝)。
        """
        return self.reasoning and self.think in ("fast", "on") and self.template != "raw"

    @property
    def _continuation_safe(self) -> bool:
        """续传安全性判定(Phase 3 §2.1 修订 + docs §8.4 裁决)。

        纯 qa(template=qa, reasoning=False)可续传:已实测 encode(prefix)+
        encode(' A1')+encode(continuation) == encode(整体),token 级等价。
        reasoning(任意 think 档)走重放:续传会固化 think 标签,偏离训练分布。
        """
        return self.template == "qa" and not self.reasoning

    def _can_continue(self) -> bool:
        """本轮是否可走续传:方言安全 + cache 干净 + 有可用 cache + 历史非空。"""
        return (
            self._continuation_safe
            and self.cache_clean
            and self.cache is not None
            and len(self.history) >= 1
        )

    def handle(
        self, text: str, *, on_text: Optional[TextCallback] = None
    ) -> ChatReply:
        text = text.strip()
        if not text:
            return ChatReply([])
        if text.startswith("/"):
            return self._command(text)

        # 渲染当前轮 prompt(首轮用 prefix 模板,后续轮在续传/重放里拼胶水)
        if self.ab:
            return self._handle_ab(text, on_text)
        return self._handle_single(text, on_text)

    def _handle_single(
        self, user_text: str, on_text: Optional[TextCallback]
    ) -> ChatReply:
        """单路生成(非 A/B),含续传/重放决策。

        状态机(三种路径,history 追加时机统一在生成后):
          首轮(history 空)     :prompt = render_prompt(user),cache=None
          续传(_can_continue)  :prompt = last_assistant + continuation_glue(user),
                                  cache = 上一轮 cache
          重放(其他)            :prompt = _build_replay_prompt(含本轮 user),
                                  cache = None
        生成成功后统一追加 [user, assistant] 两个 turn(首轮/续传)或 [assistant]
        一个 turn(重放,因 user 已在 _build_replay_prompt 前追加)。
        """
        is_first_turn = len(self.history) == 0
        is_continuation = self._can_continue()
        stream_cb = self._wrap_stream_callback(on_text) if on_text else None

        if is_first_turn:
            prompt = render_prompt(
                user_text, self.template, reasoning=self.reasoning, think=self.think
            )
            result = self.engine.generate(
                prompt, state=self.state, cache=None, config=self.config, on_text=stream_cb
            )
        elif is_continuation:
            prompt = self._build_continuation_prompt(user_text)
            result = self.engine.generate(
                prompt, state=self.state, cache=self.cache, config=self.config, on_text=stream_cb
            )
        else:
            # 重放:先把本轮 user 追加进 history,再渲染完整历史
            self.history.append(Turn(role="user", text=user_text))
            prompt = self._build_replay_prompt()
            result = self.engine.generate(
                prompt, state=self.state, cache=None, config=self.config, on_text=stream_cb
            )

        display = self._display_text(result.text)

        # 统一记录 history(文本是唯一事实源,§2.3)
        if is_first_turn or is_continuation:
            self.history.append(Turn(role="user", text=user_text))
        # 重放路径 user turn 已在上面追加,这里只补 assistant
        self.history.append(Turn(role="assistant", text=display))

        # 更新 cache 状态(供下一轮决策)
        self.cache = result.cache
        self.cache_clean = result.cache_clean

        if on_text is not None:
            return ChatReply([self._summary(result)], payload=result.to_dict())
        return ChatReply(
            [display, self._summary(result)],
            payload=result.to_dict(),
        )

    def _handle_ab(self, user_text: str, on_text: Optional[TextCallback]) -> ChatReply:
        """A/B 对比(不参与多轮 cache,A/B 推迟 v1.5,沿用单轮语义)。"""
        if self.state is None:
            return ChatReply(["A/B 已开启，但当前没有 state；请先使用 /state PATH。"])
        wrapped = render_prompt(
            user_text, self.template, reasoning=self.reasoning, think=self.think
        )
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

    def _build_continuation_prompt(self, user_text: str) -> str:
        """续传 prompt = 只有本轮的 continuation 胶水(不含上一轮回答)。

        上一轮回答的 token 已在生成循环里逐个喂入前向、固化进 cache
        (见 inference.generate 的 `input_ids = mx.array([[next_token]])`)。
        所以续传时 prompt 只需 "胶水 + 本轮 user",再 prefill 这一小段进现有 cache。

        QA.continuation_prefix_template = "\\n\\nUser: {q}\\n\\nAssistant:"。
        若把上一轮回答也拼进 prompt,会让模型"看到两遍回答"(一遍在 cache,一遍在
        新 prompt),上下文错乱、退化成复读机。
        """
        if not self.history:
            # 首轮(理论上 _can_continue 已排除,这里兜底):用 prefix 模板
            return render_prompt(
                user_text, self.template, reasoning=self.reasoning, think=self.think
            )
        # 只有胶水,不含 last_assistant(已在 cache 里)
        return QA.continuation_prefix_template.format(q=user_text)

    def _build_replay_prompt(self) -> str:
        """重放 prompt = 从 S₀ 重新渲染完整 history 文本。

        以 history 的 display 文本为准(§2.3:文本是唯一事实源,token 只审计)。
        首轮 user 用 render_prompt(含 bos 若 reasoning),后续 assistant 用裸文本、
        user 用 continuation 胶水。调用方应已把本轮 user turn 追加进 history。
        """
        if not self.history:
            return render_prompt(
                "", self.template, reasoning=self.reasoning, think=self.think
            )
        # 首轮 user turn → prefix 模板
        parts = [
            render_prompt(
                self.history[0].text, self.template,
                reasoning=self.reasoning, think=self.think,
            )
        ]
        # 后续 turns:assistant 裸文本 + user continuation 胶水交替
        for turn in self.history[1:]:
            if turn.role == "assistant":
                parts.append(turn.text)
            elif turn.role == "user":
                parts.append(QA.continuation_prefix_template.format(q=turn.text))
        return "".join(parts)

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
            return self._clear_command()
        if command == "/rewind":
            return self._rewind_command(args)
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
        if command == "/presence":
            return self._set_float("presence_penalty", args, minimum=0.0)
        if command == "/frequency":
            return self._set_float("frequency_penalty", args, minimum=0.0)
        if command == "/config":
            return ChatReply([self.config_line()])
        return ChatReply([f"未知命令: {command}；输入 /help 查看帮助。"])

    def _clear_command(self) -> ChatReply:
        """/clear 真实语义(§2.4):清空 history、丢弃 cache、回到 S₀。"""
        self.history = []
        self.cache = None
        self.cache_clean = True
        return ChatReply(["已清空会话:历史与 cache 已重置,下一轮从 S₀ 重新开始。"])

    def _rewind_command(self, args: list[str]) -> ChatReply:
        """/rewind [n]:截断最后 n 轮(默认 1),触发重放。

        一"轮" = 一个 user+assistant 对。截断后 cache 丢弃(历史被改,续传失效),
        下一轮自动走重放。n 超过轮数时 clamp 到 0。
        """
        n = 1
        if args:
            try:
                n = int(args[0])
            except ValueError:
                return ChatReply([f"/rewind 参数必须是正整数,收到: {args[0]}"])
        if n < 1:
            return ChatReply(["/rewind 参数必须 >= 1"])
        # history 按 [user, assistant, user, assistant, ...] 排列,
        # 每轮 = 2 个 turn。截断 n 轮 = 删最后 2n 个 turn。
        total_turns = len(self.history)
        turns_per_round = 2
        clamp = min(n * turns_per_round, total_turns)
        if clamp == 0:
            return ChatReply(["无历史可撤销。"])
        rounds_removed = clamp // turns_per_round
        self.history = self.history[: total_turns - clamp]
        # 历史被改 → cache 失效,下一轮重放
        self.cache = None
        self.cache_clean = False
        return ChatReply([f"已撤销 {rounds_removed} 轮;下一轮将重放剩余历史({len(self.history)} 条记录)。"])

    def _state_command(self, args: list[str]) -> ChatReply:
        if not args:
            return ChatReply([f"当前 state: {self.state_label or 'off'}"])
        value = args[0]
        if value.lower() in ("off", "none", "zero"):
            self.state = None
            self.state_label = None
            # §2.4: 切换 state = 换 S₀ = 换人设,续传旧对话无意义 → 重置会话
            self.history = []
            self.cache = None
            self.cache_clean = True
            return ChatReply(["state 已关闭,已重置会话;后续轮次使用零 state 基线。"])
        path = Path(value).expanduser()
        try:
            loaded = self.state_loader(path)
        except Exception as exc:
            return ChatReply([f"state 加载失败: {exc}"])
        self.state = loaded
        self.state_label = str(path)
        # §2.4: 多轮中途切换 state → 清空会话(换 S₀ = 换人设)
        self.history = []
        self.cache = None
        self.cache_clean = True
        return ChatReply([f"state 已加载: {path}(已重置会话)"])

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
            f"max_tokens={self.config.max_tokens} seed={self.config.seed} "
            f"presence={self.config.presence_penalty} "
            f"frequency={self.config.frequency_penalty} "
            f"decay={self.config.penalty_decay}"
        )

    @staticmethod
    def help_lines() -> list[str]:
        return [
            "/state PATH       动态加载 npz/pth(重置会话,换 S₀=换人设)",
            "/state off        关闭 state(重置会话)",
            "/state            查看当前 state",
            "/ab [on|off]      切换 A/B 输出",
            "/temperature N    调整温度（0=贪心）",
            "/top-p N          调整 top-p",
            "/presence N       调整 presence 重复惩罚(0=关)",
            "/frequency N      调整 frequency 重复惩罚(0=关)",
            "/max-tokens N     调整单轮生成上限",
            "/seed N           调整采样 seed",
            "/rewind [n]       撤销最后 n 轮(默认 1),触发重放",
            "/clear            清空历史与 cache,回到 S₀",
            "/config           查看当前配置",
            "/quit             退出",
        ]
