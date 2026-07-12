"""独立推理引擎。

这一层不感知 Typer / SwiftUI，只接收模型、tokenizer、prompt、state 与生成配置，
返回结构化结果。CLI 和未来 sidecar 共用同一 API。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
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


@dataclass(frozen=True)
class GenerationConfig:
    """单次生成配置。temperature=0 保持可复现的贪心解码。"""

    max_tokens: int = 80
    temperature: float = 0.0
    top_p: float = 0.9
    seed: int = 42
    eos_token: int = 0
    stop_sequences: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须 > 0")
        if self.temperature < 0:
            raise ValueError("temperature 必须 >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p 必须在 (0, 1] 范围内")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    stop_reason: StopReason
    elapsed: float
    used_state: bool
    config: GenerationConfig
    # 计时分段(对齐 llama.cpp 的 prompt eval / generation eval 口径)：
    #   prompt_time        = 首次前向(整个 prompt 并行 prefill)耗时
    #   generation_time    = 后续逐 token 串行 decode 耗时(不含 prefill)
    #   elapsed            = 总耗时 ≈ prompt_time + generation_time
    # 字段都带默认值，旧的构造点(tests/FakeEngine)无需改动。
    prompt_tokens: int = 0
    prompt_time: float = 0.0
    generation_time: float = 0.0

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

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
        config: Optional[GenerationConfig] = None,
        on_text: Optional[TextCallback] = None,
    ) -> GenerationResult:
        cfg = config or GenerationConfig()
        cfg.validate()

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
        generated: list[int] = []
        stop_reason: StopReason = "max_tokens"
        emitted_text = ""
        final_text = ""
        t0 = time.time()

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
            t_step_start = time.time()
            logits = self.model(input_ids, caches)[0, -1]
            if sampler is None:
                next_token = int(mx.argmax(logits, axis=-1))
            else:
                logprobs = nn.log_softmax(logits, axis=-1)
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
            decoded_text = self.tokenizer.decode(generated)
            stop_positions = [
                decoded_text.find(sequence) for sequence in stop_sequences
            ]
            stop_positions = [position for position in stop_positions if position >= 0]
            if stop_positions:
                final_text = decoded_text[: min(stop_positions)]
                emit_safe_text(final_text)
                # token_ids 表示实际返回文本，不包含角色边界。
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

        return GenerationResult(
            text=final_text,
            token_ids=generated,
            stop_reason=stop_reason,
            elapsed=time.time() - t0,
            used_state=state is not None,
            config=cfg,
            prompt_tokens=prompt_token_count,
            prompt_time=t_prefill,
            generation_time=t_gen,
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
