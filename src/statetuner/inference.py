"""独立推理引擎。

这一层不感知 Typer / SwiftUI，只接收模型、tokenizer、prompt、state 与生成配置，
返回结构化结果。CLI 和未来 sidecar 共用同一 API。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Literal, Optional

import mlx.core as mx
import mlx.nn as nn

from .core import StateInput, _load_state_dict, build_state_cache
from .templates import INSTRUCTION, QA

TemplateName = Literal["raw", "qa", "instruction"]
# think 档位:仅 reasoning 模型生效,只影响 prompt 尾部渲染(Spec §1.1)。
#   off  → Assistant: 后追加 ""            (直答)
#   fast → Assistant: 后追加 " <think>\n</think>"  (空 think 标签,跳过思考)
#   on   → Assistant: 后追加 " <think"     (模型续写思考段)
ThinkMode = Literal["off", "fast", "on"]
# reasoning 方言前缀(World tokenizer bos,= token 0):RWKV 训练每轮以此起始。
# 旧 API 把这层打包进整包模板,新世界拆出来(参数语义 = "reasoning 模型需要 bos+think 外壳",
# 不写死模型版本号:G1 是版本号会迭代)。
REASONING_BOS = "<|rwkv_tokenizer_end_of_text|>"
ThinkSuffix = {
    "off": "",
    "fast": " <think>\n</think>",
    "on": " <think",
}
StopReason = Literal["eos", "stop_sequence", "max_tokens"]
TextCallback = Callable[[str], None]
AbortChecker = Callable[[], bool]


class GenerationAborted(Exception):
    """生成被外部中断(serve abort 协议指令触发,§3.3)。

    InferenceEngine.generate 每步检查 should_abort 回调,True 则抛此异常。
    serve 层 ServeProtocol 捕获并发 error{code:aborted} 终结事件。
    """


@dataclass(frozen=True)
class GenerationConfig:
    """单次生成配置。temperature=0 保持可复现的贪心解码。

    重复惩罚(对齐 ChatRWKV 官方 v2/chat.py 的 occurrence penalty):
      presence_penalty:  对已出现过的 token 施加固定惩罚(官方默认 0.4)。
      frequency_penalty:  按出现次数累加惩罚(官方默认 0.4)。
      penalty_decay:      每步对历史计数做指数衰减(官方默认 0.996),
                          老的 token 惩罚递减,避免过度抑制早期内容。
      三者都为 0 时无惩罚(贪心/纯采样)。
    """

    max_tokens: int = 80
    temperature: float = 0.0
    top_p: float = 0.9
    seed: int = 42
    eos_token: int = 0
    stop_sequences: tuple[str, ...] = ()
    # 重复惩罚(ChatRWKV 官方语义,默认值对齐官方推荐)
    presence_penalty: float = 0.4
    frequency_penalty: float = 0.4
    penalty_decay: float = 0.996

    @property
    def has_penalty(self) -> bool:
        """是否启用重复惩罚(任一参数非零)。"""
        return self.presence_penalty > 0 or self.frequency_penalty > 0

    def validate(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须 > 0")
        if self.temperature < 0:
            raise ValueError("temperature 必须 >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p 必须在 (0, 1] 范围内")
        if self.presence_penalty < 0:
            raise ValueError("presence_penalty 必须 >= 0")
        if self.frequency_penalty < 0:
            raise ValueError("frequency_penalty 必须 >= 0")
        if not 0 < self.penalty_decay <= 1:
            raise ValueError("penalty_decay 必须在 (0, 1] 范围内")


@dataclass(frozen=True)
class GenerationResult:
    """单次生成的结果(Phase 3 §2 多轮改造)。

    token 账本拆分(§2.3):
      display_token_ids: 干净展示文本对应的 token(旧 token_ids 改名,不含角色边界/污染)。
      fed_token_ids:     实际走过前向的完整 token 序列(含 stop_sequence 污染部分)。
        - eos/max_tokens 停止:fed == display(eos 不进 cache)
        - stop_sequence 停止:fed ⊃ display(污染 token 已进 cache)

    cache 字段(§2.4):
      cache:       前向结束后的 running cache,供下轮续传传入。None = 未产出(重放场景)。
      cache_clean: cache 洁净性(§2.2)。eos/max_tokens 干净可续传;stop_sequence 脏需重放。
    """
    text: str
    display_token_ids: list[int]
    stop_reason: StopReason
    elapsed: float
    used_state: bool
    config: GenerationConfig
    # 计时分段(对齐 llama.cpp 的 prompt eval / generation eval 口径)：
    #   prompt_time        = 首次前向(整个 prompt 并行 prefill)耗时
    #   generation_time    = 后续逐 token 串行 decode 耗时(不含 prefill)
    #   elapsed            = 总耗时 ≈ prompt_time + generation_time
    prompt_tokens: int = 0
    prompt_time: float = 0.0
    generation_time: float = 0.0
    # Phase 3 §2 新增字段(带默认值,旧的无多轮构造点零改动)
    fed_token_ids: list[int] = field(default_factory=list)
    cache: object = None
    cache_clean: bool = True

    @property
    def token_count(self) -> int:
        return len(self.display_token_ids)

    @property
    def prompt_tps(self) -> float:
        """Prompt prefill 速率（t/s）。无 prefill 记录时为 0。"""
        return self.prompt_tokens / self.prompt_time if self.prompt_time > 0 else 0.0

    @property
    def generation_tps(self) -> float:
        """Decode 生成速率（t/s）。无生成或未计时时为 0。"""
        return (
            self.token_count / self.generation_time
            if self.generation_time > 0
            else 0.0
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["token_count"] = self.token_count
        data["prompt_tps"] = self.prompt_tps
        data["generation_tps"] = self.generation_tps
        # cache 是不透明的模型对象,不可序列化;to_dict 用于 JSON 输出(serve/CLI),
        # 这里排除 cache 字段,只保留可序列化的审计字段。
        data.pop("cache", None)
        return data

    def summary_line(self) -> str:
        """摘要行的单一事实源（cli.py preview / chat.py 共用）。

        格式对齐 llama.cpp 的 prompt eval / generation 分段：
          [stop=..., tokens=..., 总耗时 | Prompt: .. t/s | Generation: .. t/s]
        """
        return (
            f"[stop={self.stop_reason}, tokens={self.token_count}, "
            f"{self.elapsed:.2f}s | "
            f"Prompt: {self.prompt_tps:.1f} t/s | "
            f"Generation: {self.generation_tps:.1f} t/s]"
        )


@dataclass(frozen=True)
class ABResult:
    prompt: str
    with_state: GenerationResult
    baseline: GenerationResult

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "with_state": self.with_state.to_dict(),
            "baseline": self.baseline.to_dict(),
        }


def render_prompt(
    prompt: str,
    template: TemplateName = "raw",
    *,
    reasoning: bool = False,
    think: ThinkMode = "off",
    instruction_input: str = "",
) -> str:
    """按训练同源模板 + reasoning 方言 + think 档位渲染推理 prompt。

    单一事实源(验收 a/b/c)。

    Args:
        prompt: 用户问题(qa)或指令(instruction)或裸文本(raw)。
        template: 训练/推理模板。
        reasoning: 是否套 reasoning 方言(前缀加 REASONING_BOS + 尾部按 think 追加)。
                   仅对 qa 模板有意义;raw/instruction 传 True 报错(v1 不做)。
        think: think 档位。仅当 reasoning=True 时合法(reasoning=False 时强制 off)。

    Kwargs:
        instruction_input: instruction 模板的 Input 字段(空则自动降级)。

    渲染规则(对齐 RWKV 官方文档,Spec §1.1):
      raw        → prompt 原样(reasoning/think 对 raw 无意义,传了报错)
      qa         → "User: {prompt}\\n\\nAssistant:"
        + reasoning=True: "{BOS}User: {prompt}\\n\\nAssistant:{think_suffix}"
      instruction→ INSTRUCTION.format_prefix(instruction=prompt, input=instruction_input)
                   (空 input 自动降级,验收 d)
    """
    if think != "off" and not reasoning:
        raise ValueError(
            f"--think 仅在 reasoning 模型上生效(reasoning=False 时 think 必须 off),收到 think={think!r}"
        )
    if reasoning and think not in ThinkSuffix:
        raise ValueError(f"不支持的 think 档位: {think!r}(合法: off/fast/on)")

    if template == "raw":
        if reasoning or think != "off":
            raise ValueError(
                "raw 模板不与 reasoning/think 组合(裸文本无 Assistant: 锚点)"
            )
        return prompt

    if template == "qa":
        prefix = QA.format_prefix(q=prompt)
        if reasoning:
            prefix = REASONING_BOS + prefix + ThinkSuffix[think]
        return prefix

    if template == "instruction":
        if reasoning or think != "off":
            raise ValueError(
                "instruction 模板不与 reasoning/think 组合(v1 不做 reasoning 指令)"
            )
        return INSTRUCTION.format_prefix(
            instruction=prompt, input=instruction_input
        )

    raise ValueError(f"不支持的模板: {template!r}")


def with_template_stops(
    config: GenerationConfig, template: TemplateName
) -> GenerationConfig:
    """把模板定义的角色边界加入生成配置。

    reasoning 方言不改 stop 边界(只是 prompt 外壳),所以旧 reasoning 整包模板的
    ("\\nUser:", "\\nSystem:") 合并到 QA 的 stops 里——历史实测只用
    "\\nUser:" 触发停,"\\nSystem:" 是冗余兜底;新世界 qa + reasoning 复用
    QA.inference_stop_sequences。
    """
    if template == "raw":
        return config
    if template == "qa":
        return replace(config, stop_sequences=QA.inference_stop_sequences)
    if template == "instruction":
        return replace(config, stop_sequences=INSTRUCTION.inference_stop_sequences)
    raise ValueError(f"不支持的模板: {template!r}")


class InferenceEngine:
    """RWKV state 注入推理。模型只加载一次，可连续 preview/eval。"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_state(self, state: StateInput):
        """把 npz/pth 一次性载入内存，供常驻 chat 动态切换。"""
        return _load_state_dict(state)

    def generate(
        self,
        prompt: str,
        *,
        state: StateInput = None,
        cache=None,
        config: Optional[GenerationConfig] = None,
        on_text: Optional[TextCallback] = None,
        should_abort: Optional[AbortChecker] = None,
    ) -> GenerationResult:
        """单次生成。

        Phase 3 §2 多轮改造:
          cache=None  → 按 state 新建 running cache(零 state 走 model.make_cache())
          cache=<obj> → 续传:复用传入的 cache,只 prefill 新 prompt
        返回的 GenerationResult.cache 是前向结束后的 cache,供下轮续传。
        cache_clean(§2.2):eos/max_tokens 干净可续传;stop_sequence 脏需重放。

        should_abort(§3.3 abort 机制):每步前检查,True 则抛 GenerationAborted。
        默认 None → 不检查(所有现有调用方:CLI/ChatSession/service 零改动)。
        """
        cfg = config or GenerationConfig()
        cfg.validate()

        # cache 续传优先:传入 cache 则复用,否则按 state 新建。
        # state 仍要传入(用于 used_state 标记 + 续传场景下 state 已固化为 cache)。
        if cache is not None:
            caches = cache
        else:
            state_dict = _load_state_dict(state)
            caches = (
                build_state_cache(state_dict, batch_size=1)
                if state_dict is not None
                else self.model.make_cache()
            )

        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt 编码后为空")

        mx.random.seed(cfg.seed)
        sampler = None
        if cfg.temperature > 0:
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=cfg.temperature, top_p=cfg.top_p)

        input_ids = mx.array([prompt_ids])
        generated: list[int] = []          # 展示文本对应的 token(干净)
        fed_token_ids: list[int] = []      # 实际喂入前向的完整 token(含污染,§2.3)
        stop_reason: StopReason = "max_tokens"
        emitted_text = ""
        final_text = ""
        t0 = time.time()
        # 重复惩罚 occurrence 表(ChatRWKV 官方语义): {token_id: 衰减后计数}
        occurrence: dict[int, float] = {}

        def emit_safe_text(safe_text: str) -> None:
            nonlocal emitted_text
            if on_text is None:
                return
            if safe_text.startswith(emitted_text):
                delta = safe_text[len(emitted_text) :]
            else:
                # 极少数 tokenizer 可能因上下文修正已解码文本；不重复输出旧内容。
                common = 0
                for old, new in zip(emitted_text, safe_text):
                    if old != new:
                        break
                    common += 1
                delta = safe_text[common:]
            if delta:
                on_text(delta)
            emitted_text = safe_text

        def pending_stop_prefix_length(text: str) -> int:
            """保留可能继续长成 stop sequence 的字符后缀。"""
            pending = 0
            for sequence in stop_sequences:
                limit = min(len(text), len(sequence) - 1)
                for size in range(limit, 0, -1):
                    if text.endswith(sequence[:size]):
                        pending = max(pending, size)
                        break
            return pending

        stop_sequences = tuple(
            sequence for sequence in cfg.stop_sequences if sequence
        )

        # 计时分段(对齐 llama.cpp)：
        #   step 0  → 首次前向消化整个 prompt(prefill,并行)
        #   step>0  → 逐 token decode(串行),累加到 t_gen
        # t_step 窗口覆盖「前向 + 采样得 next_token」,int() 隐式触发 MLX eval,
        # 保证 GPU 计算完成才停表。
        t_prefill = 0.0
        t_gen = 0.0
        prompt_token_count = len(prompt_ids)

        for step in range(cfg.max_tokens):
            # abort 检查(§3.3):serve 协议中断信号,每步前检查一次。
            # 默认 None 时短路,零开销。延迟到下一 step 边界(MLX 前向 ~50-200ms)。
            if should_abort is not None and should_abort():
                raise GenerationAborted()
            t_step_start = time.time()
            logits = self.model(input_ids, caches)[0, -1]
            # 重复惩罚(X3 向量化):ChatRWKV 官方语义 logits[tok] -= presence+cnt*freq。
            # 旧实现是 per-token Python 循环,300 token 对话后期每步几百次 dispatch;
            # 改成 scatter(权重)+ 一次相减,避免每步往图里塞 len(occurrence) 个小 op。
            if cfg.has_penalty and occurrence:
                tok_ids = mx.array(list(occurrence.keys()))
                counts = mx.array(list(occurrence.values()))
                penalties = cfg.presence_penalty + counts * cfg.frequency_penalty
                # scatter-add 取负 = 在 tok_ids 位置减去 penalties(原地语义)
                logits[tok_ids] -= penalties
            if sampler is None:
                next_token = int(mx.argmax(logits, axis=-1))
            else:
                # logsumexp 替 nn.log_softmax(mlx_lm 做法,更轻量:减一次 max)
                logprobs = logits - mx.logsumexp(logits, keepdims=True)
                next_token = int(sampler(logprobs))
            t_step = time.time() - t_step_start
            if step == 0:
                t_prefill = t_step
            else:
                t_gen += t_step
            if next_token == cfg.eos_token:
                stop_reason = "eos"
                break
            generated.append(next_token)
            # fed_token_ids 记录所有走过前向的生成 token(含将被 stop_sequence
            # 污染的部分),用于多轮审计(§2.3)。eos 不进 cache,不记入。
            fed_token_ids.append(next_token)
            # 更新 occurrence:先衰减所有历史计数,再给当前 token 计数+1。
            # X3:衰减也批处理(字典推导式一次性重建),避免 per-key Python 循环。
            if cfg.has_penalty:
                occurrence = {
                    t_id: cnt * cfg.penalty_decay for t_id, cnt in occurrence.items()
                }
                occurrence[next_token] = occurrence.get(next_token, 0.0) + 1.0
            decoded_text = self.tokenizer.decode(generated)
            stop_positions = [
                decoded_text.find(sequence) for sequence in stop_sequences
            ]
            stop_positions = [position for position in stop_positions if position >= 0]
            if stop_positions:
                final_text = decoded_text[: min(stop_positions)]
                emit_safe_text(final_text)
                # display_token_ids 表示实际返回文本，不包含角色边界。
                # fed_token_ids 保留污染部分(供审计),不重编码。
                generated = self.tokenizer.encode(final_text)
                stop_reason = "stop_sequence"
                break
            pending = pending_stop_prefix_length(decoded_text)
            safe_text = decoded_text[:-pending] if pending else decoded_text
            emit_safe_text(safe_text)
            input_ids = mx.array([[next_token]])

        # flush：EOS/max_tokens 或 stop_sequence 前仍可能有尚未确认的安全文本。
        if stop_reason != "stop_sequence":
            final_text = self.tokenizer.decode(generated)
            emit_safe_text(final_text)

        # cache 洁净性(§2.2):eos/max_tokens 未产生越界 token,干净;stop_sequence 脏。
        cache_clean = stop_reason != "stop_sequence"

        return GenerationResult(
            text=final_text,
            display_token_ids=generated,
            stop_reason=stop_reason,
            elapsed=time.time() - t0,
            used_state=state is not None,
            config=cfg,
            prompt_tokens=prompt_token_count,
            prompt_time=t_prefill,
            generation_time=t_gen,
            fed_token_ids=fed_token_ids,
            cache=caches,
            cache_clean=cache_clean,
        )

    def compare(
        self,
        prompt: str,
        *,
        state: StateInput,
        config: Optional[GenerationConfig] = None,
    ) -> ABResult:
        """相同配置/seed 下生成 tuned state 与零 state 基线。"""
        if state is None:
            raise ValueError("A/B 对比必须提供 state")
        cfg = config or GenerationConfig()
        return ABResult(
            prompt=prompt,
            with_state=self.generate(prompt, state=state, config=cfg),
            baseline=self.generate(prompt, state=None, config=cfg),
        )
