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
    AbortChecker,
    GenerationAborted,
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


@dataclass(frozen=True)
class StateChange:
    """set_state 的结构化结果(T3 public API)。

    ok=True 表示切换成功(off 或加载路径);False 表示加载失败(路径无效等)。
    state_label 切换后的标签(off / 路径);history_cleared 因换 S₀ 会重置会话。
    message 是给人看的中文说明(CLI/chat 用),协议层不依赖它。
    """

    ok: bool
    state_label: Optional[str]
    history_cleared: bool
    message: str


@dataclass(frozen=True)
class RewindChange:
    """rewind 的结构化结果(T3 public API)。"""

    rounds_removed: int
    history_len: int
    message: str


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
        # 默认值(高创造力档):temp=1.2 + top_p=0.5 是 RWKV 聊天的实测好档,
        # max_tokens=300 给多轮对话足够回旋空间。
        base_config = config or GenerationConfig(
            max_tokens=300, temperature=1.2, top_p=0.5
        )
        self.config = with_template_stops(base_config, template)
        self.config.validate()
        self.template = template
        self.reasoning = reasoning
        self.think = think
        self.state = state
        self.state_label = state_label
        self.state_loader = state_loader or engine.load_state
        self.ab = ab
        # abort 钩子(§3.3):serve 协议设此项,handle 内部 generate 透传给引擎,
        # 每步检查 → GenerationAborted。默认 None,CLI/chat 命令零影响。
        self.abort_checker: Optional[AbortChecker] = None
        # Phase 3 §2 多轮状态
        self.history: list[Turn] = []
        self.cache: object = None  # running cache,None=需从 S₀ 新建(重放)
        self.cache_clean: bool = True  # 上一轮结束时 cache 是否干净可续传

    @property
    def _has_reasoning_dialect(self) -> bool:
        """是否需要 reasoning 方言的显示清洗(前导 \\n 是空 think 标签后的自然换行)。

        旧实现硬判某旧模板字符串;新世界 reasoning 是正交开关,
        且仅与 qa 模板组合(instruction/raw 不与 reasoning 组合,由 render_prompt 拒绝)。
        """
        return (
            self.reasoning and self.think in ("fast", "on") and self.template != "raw"
        )

    @property
    def _continuation_safe(self) -> bool:
        """续传安全性判定(Phase 3 §2.1 修订 + docs §8.4 裁决)。

        纯 qa(template=qa, reasoning=False)可续传:已实测 encode(prefix)+
        encode(' A1')+encode(continuation) == encode(整体),token 级等价。
        reasoning(任意 think 档)走重放:续传会固化 think 标签,偏离训练分布。
        """
        return self.template == "qa" and not self.reasoning

    @property
    def _supports_multiturn(self) -> bool:
        """模板是否支持多轮胶水(S2)。

        QA 有 continuation_prefix_template;INSTRUCTION 的 continuation 是 None
        (单任务,模板 docstring 明确禁多轮);raw 无角色锚点。
        非 qa 模板每轮独立:prompt 永远是 render_prompt(user)(无历史拼接),
        避免用 QA 的胶水给 instruction/raw 拼多轮(旧 bug,会偏离训练分布)。
        history 仍记录(供 /rewind、A/B 展示),但不进 prompt。
        """
        return self.template == "qa"

    def _can_continue(self) -> bool:
        """本轮是否可走续传:方言安全 + cache 干净 + 有可用 cache + 历史非空。"""
        return (
            self._continuation_safe
            and self.cache_clean
            and self.cache is not None
            and len(self.history) >= 1
        )

    def handle(self, text: str, *, on_text: Optional[TextCallback] = None) -> ChatReply:
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

        状态机(三种路径):
          首轮(history 空)     :prompt = render_prompt(user),cache=None
          续传(_can_continue)  :prompt = continuation_glue(user),
                                  cache = 上一轮 cache(续传安全方言才走)
          重放(其他)            :prompt = _build_replay_prompt(pending_user=user),
                                  cache = None

        P0-2 会话原子性:**所有 history/cache 变更都在 generate 成功之后**。
        旧实现在重放路径 generate 前就 append user turn → abort 时留下孤儿 user
        turn → 下轮重放拼出两段 User: 中间无回答;_rewind 的"每轮=2 turn"假设破产。
        续传路径传 self.cache(同一对象)→ 前向就地写 cache[1],abort 时已污染,
        而 cache_clean 仍 True → 下轮 _can_continue() 为真 → 从不存在的上下文续。
        现在统一在成功后 append + 更新 cache;abort 时 history/cache 不变,
        cache_clean 显式置 False(强制下轮重放)。
        """
        is_first_turn = len(self.history) == 0
        # S2:非 qa 模板(instruction/raw)不支持多轮胶水,每轮独立 prompt。
        # 用 self._supports_multiturn 把"是否首轮"扩成"本轮是否走独立 prompt"。
        is_independent_turn = is_first_turn or not self._supports_multiturn
        is_continuation = self._can_continue()
        stream_cb = self._wrap_stream_callback(on_text) if on_text else None

        try:
            if is_independent_turn:
                # 首轮 或 非多轮模板(S2):每轮独立 prompt,无历史拼接。
                prompt = render_prompt(
                    user_text, self.template, reasoning=self.reasoning, think=self.think
                )
                result = self.engine.generate(
                    prompt,
                    state=self.state,
                    cache=None,
                    config=self.config,
                    on_text=stream_cb,
                    should_abort=self.abort_checker,
                )
            elif is_continuation:
                prompt = self._build_continuation_prompt(user_text)
                result = self.engine.generate(
                    prompt,
                    state=self.state,
                    cache=self.cache,
                    config=self.config,
                    on_text=stream_cb,
                    should_abort=self.abort_checker,
                )
            else:
                # 重放:渲染含本轮 user 的完整历史,但不改 self.history(P0-2)。
                prompt = self._build_replay_prompt(pending_user=user_text)
                result = self.engine.generate(
                    prompt,
                    state=self.state,
                    cache=None,
                    config=self.config,
                    on_text=stream_cb,
                    should_abort=self.abort_checker,
                )
        except GenerationAborted:
            # P0-2:abort 时 history 未变(所有 append 都在成功后);但续传路径
            # 传入了 self.cache(同一对象),前向可能已就地写 cache[1] → 已污染。
            # 置 cache=None + cache_clean=False,强制下轮重放(§2.2 脏 cache 语义)。
            # 重放后 re-raise,让 serve 发 error{aborted}。
            self.cache = None
            self.cache_clean = False
            raise

        display = self._display_text(result.text)

        # 统一记录 history(文本是唯一事实源,§2.3)—— 全部在 generate 成功后。
        # 三条路径都 append user + assistant(P0-2:重放路径不再提前 append)。
        self.history.append(Turn(role="user", text=user_text))
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

    def _build_replay_prompt(self, *, pending_user: Optional[str] = None) -> str:
        """重放 prompt = 从 S₀ 重新渲染完整 history 文本。

        以 history 的 display 文本为准(§2.3:文本是唯一事实源,token 只审计)。
        首轮 user 用 render_prompt(含 bos 若 reasoning),后续 assistant 用裸文本、
        user 用 continuation 胶水。

        pending_user(P0-2):本轮 user 文本。调用方不再提前 append 进 history
        (abort 时会留孤儿),而是通过此参数参与渲染,生成成功后才 append。
        """
        # 构造本次渲染用的 turns = 历史 + (可选)本轮 user
        turns: list[Turn] = list(self.history)
        if pending_user is not None:
            turns.append(Turn(role="user", text=pending_user))
        if not turns:
            return render_prompt(
                "", self.template, reasoning=self.reasoning, think=self.think
            )
        # 首轮 user turn → prefix 模板
        parts = [
            render_prompt(
                turns[0].text,
                self.template,
                reasoning=self.reasoning,
                think=self.think,
            )
        ]
        # 后续 turns:assistant 裸文本 + user continuation 胶水交替
        for turn in turns[1:]:
            if turn.role == "assistant":
                parts.append(turn.text)
            elif turn.role == "user":
                parts.append(QA.continuation_prefix_template.format(q=turn.text))
        return "".join(parts)

    def _display_text(self, text: str) -> str:
        """模板相关的显示清洗,统一用于 history 记录与 ChatReply.lines。

        - think=on:只保留 ``</think>`` 之后的 answer(品类铁律:reasoning 模型
          重放时历史 assistant 必须是裸 answer,think 标签只在当前生成轮。
          docs/g1g-decode-alignment.md §8.4 实测 + 官方 chat_template 同惯例)。
          无 ``</think>``(max_tokens 截断等)→ 返回空 answer:半截思考不该当
          历史回答重放(展示层 cli.py 会把已生成内容 dim 显示为思考 + 标注截断)。
        - think=fast/off + reasoning:去掉前导换行(空 think 标签后的自然换行)。

        注意:GenerationResult.text 保留完整 raw(含 think),供展示层(cli.py)
        拆两段做双 panel;这里剥出的 answer 才是进 history 的文本。

        T1:think=on 拆分走 thinking.py 单一事实源(旧实现 rfind 与 console 的 find
        不一致;现统一 find,reasoning 模型 think 段内部不会合法出现闭合标签)。
        """
        if self.reasoning and self.think == "on":
            from .thinking import split_thinking
            _thinking, answer = split_thinking(text)
            return answer.lstrip("\n")
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

    # ── public API(T3:协议层/未来 UI 共用,不依赖 CLI 斜杠命令)──────────
    # 斜杠命令退化成这些方法的 parser;协议 payload 取结构化字段,
    # 不再从 reply.lines[0] 捞中文人话(改 CLI 文案 → 协议跟着变)。

    def set_state(self, path: Optional[str]) -> StateChange:
        """切换 state(T3 public API)。

        path=None 关闭 state(零基线);path=字符串 加载该 npz/pth。
        换 S₀ = 换人设,续传旧对话无意义 → 重置会话(history/cache 清空,§2.4)。

        返回 StateChange(ok/state_label/history_cleared/message)。
        ok=False 表示加载失败(路径无效、格式不符等),state 维持原状。
        """
        if path is None:
            self.state = None
            self.state_label = None
            self.history = []
            self.cache = None
            self.cache_clean = True
            return StateChange(
                ok=True, state_label=None, history_cleared=True,
                message="state 已关闭,已重置会话;后续轮次使用零 state 基线。",
            )
        resolved = Path(path).expanduser()
        try:
            loaded = self.state_loader(resolved)
        except Exception as exc:
            return StateChange(
                ok=False, state_label=self.state_label, history_cleared=False,
                message=f"state 加载失败: {exc}",
            )
        self.state = loaded
        self.state_label = str(resolved)
        self.history = []
        self.cache = None
        self.cache_clean = True
        return StateChange(
            ok=True, state_label=str(resolved), history_cleared=True,
            message=f"state 已加载: {resolved}(已重置会话)",
        )

    def rewind(self, n: int = 1) -> RewindChange:
        """撤销最后 n 轮(T3 public API)。一"轮" = 一个 user+assistant 对。

        历史被改 → cache 失效,下一轮自动重放。
        n 超过现有轮数时 clamp 到 0(不报错)。
        """
        if n < 1:
            n = 1
        total_turns = len(self.history)
        turns_per_round = 2
        clamp = min(n * turns_per_round, total_turns)
        if clamp == 0:
            return RewindChange(
                rounds_removed=0, history_len=total_turns,
                message="无历史可撤销。",
            )
        rounds_removed = clamp // turns_per_round
        self.history = self.history[: total_turns - clamp]
        self.cache = None
        self.cache_clean = False
        return RewindChange(
            rounds_removed=rounds_removed, history_len=len(self.history),
            message=f"已撤销 {rounds_removed} 轮;下一轮将重放剩余历史"
                    f"({len(self.history)} 条记录)。",
        )

    def reset(self) -> None:
        """清空会话(T3 public API,/clear 真实语义)。

        清空 history、丢弃 cache、回到 S₀。无返回(纯副作用,协议层发 ok 即可)。
        """
        self.history = []
        self.cache = None
        self.cache_clean = True

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
            return self._set_float(
                "top_p", args, minimum=0.0, maximum=1.0, strict_min=True
            )
        if command == "/max-tokens":
            return self._set_int("max_tokens", args, minimum=1)
        if command == "/seed":
            return self._set_int("seed", args)
        if command == "/presence":
            return self._set_float("presence_penalty", args, minimum=0.0)
        if command == "/frequency":
            return self._set_float("frequency_penalty", args, minimum=0.0)
        if command == "/penalty-decay":
            return self._set_float(
                "penalty_decay", args, minimum=0.0, maximum=1.0, strict_min=True
            )
        if command == "/config":
            return ChatReply([self.config_line()])
        return ChatReply([f"未知命令: {command}；输入 /help 查看帮助。"])

    def _clear_command(self) -> ChatReply:
        """/clear 真实语义(§2.4):清空 history、丢弃 cache、回到 S₀。

        T3:薄包装,核心逻辑在 reset() public API。
        """
        self.reset()
        return ChatReply(["已清空会话:历史与 cache 已重置,下一轮从 S₀ 重新开始。"])

    def _rewind_command(self, args: list[str]) -> ChatReply:
        """/rewind [n] parser(T3):解析 n → rewind(n) public API。"""
        n = 1
        if args:
            try:
                n = int(args[0])
            except ValueError:
                return ChatReply([f"/rewind 参数必须是正整数,收到: {args[0]}"])
        result = self.rewind(n)
        return ChatReply([result.message])

    def _state_command(self, args: list[str]) -> ChatReply:
        """/state parser(T3):解析 path/off → set_state(path|None) public API。"""
        if not args:
            return ChatReply([f"当前 state: {self.state_label or 'off'}"])
        value = args[0]
        if value.lower() in ("off", "none", "zero"):
            result = self.set_state(None)
        else:
            result = self.set_state(value)
        return ChatReply([result.message])

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

    def _set_int(
        self, field: str, args: list[str], *, minimum: Optional[int] = None
    ) -> ChatReply:
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

    def config_groups(self, *, brief: bool = False) -> list[tuple[str, list[tuple[str, str]]]]:
        """当前配置,按语义分组(供 console.render_config_table 渲染)。

        三组:模板/状态、采样、重复惩罚(ChatRWKV 官方语义)。
        重复惩罚组标注官方默认值,让用户知道基准和当前偏移。
        brief=True 时去掉默认值注解,用于启动横幅的紧凑显示。
        """
        reasoning_note = (
            f"on (think={self.think})" if self.reasoning else "off"
        )
        def _penalty(value, default):
            return f"{value}" if brief else f"{value}  (ChatRWKV 默认 {default})"
        return [
            ("模板", [
                ("template", self.template),
                ("reasoning", reasoning_note),
                ("state", self.state_label or "off"),
                ("ab", "on" if self.ab else "off"),
            ]),
            ("采样", [
                ("temperature", f"{self.config.temperature}"),
                ("top_p", f"{self.config.top_p}"),
                ("max_tokens", f"{self.config.max_tokens}"),
                ("seed", f"{self.config.seed}"),
            ]),
            ("重复惩罚", [
                ("presence", _penalty(self.config.presence_penalty, "0.4")),
                ("frequency", _penalty(self.config.frequency_penalty, "0.4")),
                ("decay", _penalty(self.config.penalty_decay, "0.996")),
            ]),
        ]

    def config_line(self) -> str:
        """当前配置单行摘要(纯文本降级路径;表格渲染走 config_groups)。

        保留单行格式供 ChatReply(纯文本)和轻量日志;rich 表格用 config_groups。
        """
        reasoning_part = (
            f" reasoning=on think={self.think}" if self.reasoning else ""
        )
        return (
            f"template={self.template}{reasoning_part} "
            f"state={self.state_label or 'off'} ab={'on' if self.ab else 'off'} | "
            f"采样: temp={self.config.temperature} top_p={self.config.top_p} "
            f"max_tokens={self.config.max_tokens} seed={self.config.seed} | "
            f"重复惩罚: presence={self.config.presence_penalty} "
            f"frequency={self.config.frequency_penalty} "
            f"decay={self.config.penalty_decay}"
        )

    @staticmethod
    def help_lines() -> list[str]:
        """命令帮助。每行 "命令  说明"(≥2 空格分界,供表格解析)。

        分组标题行不以 / 开头,渲染时作为分隔/标题显示。
        重复惩罚组标注官方默认值,降低调参认知门槛。
        """
        return [
            "会话控制",
            "/state PATH       动态加载 npz/pth(重置会话,换 S₀=换人设)",
            "/state off        关闭 state(重置会话)",
            "/state            查看当前 state",
            "/ab [on|off]      切换 A/B 输出",
            "/rewind [n]       撤销最后 n 轮(默认 1),触发重放",
            "/clear            清空历史与 cache,回到 S₀",
            "/config           查看当前配置",
            "/quit             退出",
            "",
            "采样参数",
            "/temperature N    调整温度(0=贪心;越高越随机)",
            "/top-p N          调整 top-p(0-1;越小越聚焦)",
            "/max-tokens N     调整单轮生成上限",
            "/seed N           调整采样 seed(同 seed 可复现)",
            "",
            "重复惩罚(ChatRWKV 官方默认 0.4/0.4/0.996)",
            "/presence N       presence 惩罚:对已出现 token 固定惩罚(0=关)",
            "/frequency N      frequency 惩罚:按出现次数累加(0=关)",
            "/penalty-decay N  历史计数衰减率(接近 1=记忆长,默认 0.996)",
        ]
