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
from .templates import NEKO_QA

TemplateName = Literal["raw", "nekoqa"]
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

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["token_count"] = self.token_count
        return data


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


def render_prompt(prompt: str, template: TemplateName = "raw") -> str:
    """按训练同源模板渲染推理 prompt。"""
    if template == "raw":
        return prompt
    if template == "nekoqa":
        return NEKO_QA.format_prefix(q=prompt)
    raise ValueError(f"不支持的模板: {template!r}")


def with_template_stops(
    config: GenerationConfig, template: TemplateName
) -> GenerationConfig:
    """把模板定义的角色边界加入生成配置。"""
    if template == "raw":
        return config
    if template == "nekoqa":
        return replace(config, stop_sequences=NEKO_QA.inference_stop_sequences)
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

        for _ in range(cfg.max_tokens):
            logits = self.model(input_ids, caches)[0, -1]
            if sampler is None:
                next_token = int(mx.argmax(logits, axis=-1))
            else:
                logprobs = nn.log_softmax(logits, axis=-1)
                next_token = int(sampler(logprobs))
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
