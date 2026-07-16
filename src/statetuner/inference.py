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
            raise ValueError("max_tokens must be > 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the range (0, 1]")
        if self.presence_penalty < 0:
            raise ValueError("presence_penalty must be >= 0")
        if self.frequency_penalty < 0:
            raise ValueError("frequency_penalty must be >= 0")
        if not 0 < self.penalty_decay <= 1:
            raise ValueError("penalty_decay must be in the range (0, 1]")


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
    #   generation_time    = 首 token 就绪到最后一个 decode token 就绪的墙钟时间
    #                        (不含 prefill；包含 GPU/CPU 流水重叠后的实际间隔)
    #   decode_steps       = generation_time 实际覆盖的前向次数
    #   elapsed            = 总耗时 ≈ prompt_time + generation_time
    prompt_tokens: int = 0
    prompt_time: float = 0.0
    generation_time: float = 0.0
    decode_steps: int = 0
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
        """连续 decode 出字速率（t/s）。无 decode step 或未计时时为 0。

        step 0 同时完成 prompt prefill 和首 token 采样，计入 prompt_time；
        generation_time 从首 token 就绪开始，只覆盖 step > 0 的 token 间隔，
        所以分子必须是 decode_steps，不能用包含首 token 的 token_count。
        """
        return (
            self.decode_steps / self.generation_time
            if self.decode_steps > 0 and self.generation_time > 0
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
            f"--think applies only to reasoning models (think must be off when reasoning=False); received think={think!r}"
        )
    if reasoning and think not in ThinkSuffix:
        raise ValueError(f"Unsupported think mode: {think!r} (valid: off/fast/on)")

    if template == "raw":
        if reasoning or think != "off":
            raise ValueError(
                "The raw template cannot be combined with reasoning/think (plain text has no Assistant: anchor)"
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
                "The instruction template cannot be combined with reasoning/think (reasoning instructions are not supported in v1)"
            )
        return INSTRUCTION.format_prefix(
            instruction=prompt, input=instruction_input
        )

    raise ValueError(f"Unsupported template: {template!r}")


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
    raise ValueError(f"Unsupported template: {template!r}")


def _last_token_logits(model, input_ids, caches):
    """RWKV7 只投影最后一个 hidden，避免构造整段词表 logits。

    prefill 的后续生成只使用最后一个位置的 logits。mlx-lm RWKV7
    默认会先对 [batch, sequence, hidden] 整段做 lm_head，再由调用方取
    ``[0, -1]``。这里保留完整 RWKV 主体和 cache 更新，只在进词表投影
    前取 last hidden。未知模型/测试 mock 仍走原始外层前向。
    """
    inner = getattr(model, "model", None)
    args = getattr(model, "args", None)
    if getattr(model, "model_type", None) != "rwkv7" or inner is None or args is None:
        return model(input_ids, caches)[0, -1]

    if getattr(args, "tie_word_embeddings", False):
        embeddings = getattr(inner, "embeddings", None)
        projection = getattr(embeddings, "as_linear", None)
    else:
        projection = getattr(model, "lm_head", None)

    # 所有能力检查必须在 inner 前向前完成；否则 fallback 会让同一段
    # input 重复写入 RWKV cache。
    if not callable(projection):
        return model(input_ids, caches)[0, -1]

    hidden = inner(input_ids, caches)
    logits = projection(hidden[:, -1:, :])
    return logits[0, -1]


def _has_quantized_modules(model) -> bool:
    """量化模型暂不启用整步编译。

    1.5B int8 的稳定 A/B 中整步 ``mx.compile`` 没有可复现收益，而 bf16
    有约 4% decode 提升。按实测边界保留 int8 原路径，避免用复杂度换零收益。
    """
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return False
    return any(
        type(module).__name__.startswith("Quantized")
        for _, module in named_modules()
    )


def _read_cache_state(caches):
    """把 ArraysCache / state 注入用普通 list 统一成可编译的 array tree。

    每层做一次浅拷贝，防止首次 tracing 时 holder 的 ``cache[i] = value``
    改写调用方正在持有的 Python list；array 本身保持零拷贝。
    """
    return [
        list(cache.state) if hasattr(cache, "state") else list(cache)
        for cache in caches
    ]


def _write_cache_state(caches, state) -> None:
    """把编译图返回的 state 写回原 cache，并保留外层对象身份。"""
    for cache, layer_state in zip(caches, state):
        if hasattr(cache, "state"):
            cache.state = list(layer_state)
        else:
            cache[:] = layer_state


class _CompiledRwkv7Decode:
    """可跨请求复用的纯函数 RWKV7 单 token 前向。

    cache state 是显式输入/输出，不能作为 ``mx.compile(outputs=...)`` 的
    闭包状态捕获；否则编译图会绑定某一次生成的 cache，无法安全用于多轮或
    并列请求。重复惩罚、采样和停止判断仍留在 Python 生成循环中。
    """

    def __init__(self, model):
        self._model = model
        self._cache_holders = model.make_cache()
        if not self._cache_holders or not all(
            hasattr(cache, "state") for cache in self._cache_holders
        ):
            raise TypeError("RWKV7 compiled decode requires an ArraysCache-style holder")

        def decode_step(input_ids, cache_state):
            for holder, layer_state in zip(self._cache_holders, cache_state):
                holder.state = layer_state
            logits = _last_token_logits(
                self._model, input_ids, self._cache_holders
            )
            return logits, [holder.state for holder in self._cache_holders]

        # 权重是捕获输入；cache 是每次调用显式传入/返回的动态状态。
        self._step = mx.compile(decode_step, inputs=model.state)

    def __call__(self, input_ids, cache_state):
        return self._step(input_ids, cache_state)


def _make_compiled_rwkv7_decode(model):
    """能力探测式创建 compiled decode；不支持的模型透明回退原路径。"""
    if getattr(model, "model_type", None) != "rwkv7":
        return None
    if not callable(getattr(model, "make_cache", None)):
        return None
    if not hasattr(model, "state") or _has_quantized_modules(model):
        return None
    try:
        return _CompiledRwkv7Decode(model)
    except (AttributeError, TypeError, ValueError):
        return None


class InferenceEngine:
    """RWKV state 注入推理。模型只加载一次，可连续 preview/eval。"""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        compile_decode: bool = True,
        async_decode: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._compiled_decode = (
            _make_compiled_rwkv7_decode(model) if compile_decode else None
        )
        self._async_decode = async_decode and self._compiled_decode is not None

    @property
    def compiled_decode_enabled(self) -> bool:
        """当前模型是否启用了可复用的 RWKV7 单 token 编译图。"""
        return self._compiled_decode is not None

    @property
    def decode_backend(self) -> str:
        """实际 decode 后端名称，供 benchmark 审计运行口径。"""
        if self._async_decode:
            return "mx.compile+async"
        if self._compiled_decode is not None:
            return "mx.compile"
        return "eager"

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
            raise ValueError("Prompt is empty after encoding")

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
        t0 = time.perf_counter()
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
        #   step>0  → 首 token 之后的连续 decode 墙钟区间
        # prompt_time 计到首 token 对 CPU 可见；generation_time 从首 token
        # 就绪计到最后一个 decode token 就绪。后者用墙钟区间而非逐 step
        # 阻塞时间求和，才能正确计算 async GPU / CPU 流水重叠后的真实出字速度。
        t_prefill = 0.0
        t_gen = 0.0
        decode_steps = 0
        prompt_token_count = len(prompt_ids)
        compiled_cache_state = None
        pipelined_logits = None
        generation_started_at = None

        for step in range(cfg.max_tokens):
            # abort 检查(§3.3):serve 协议中断信号,每步前检查一次。
            # 默认 None 时短路,零开销。延迟到下一 step 边界(MLX 前向 ~50-200ms)。
            if should_abort is not None and should_abort():
                raise GenerationAborted()
            t_step_start = time.perf_counter()
            if pipelined_logits is not None:
                # 上一轮已把依赖 next_token 的 compiled graph 提交给 GPU；
                # CPU 做 tokenizer/stop/stream 时，这一轮前向可并行执行。
                logits = pipelined_logits
            elif step > 0 and self._compiled_decode is not None:
                # benchmark 的同步 compiled 对照路径；产品默认走上面的 async
                # pipeline。cache 仍是显式输入/输出，语义完全相同。
                if compiled_cache_state is None:
                    compiled_cache_state = _read_cache_state(caches)
                logits, compiled_cache_state = self._compiled_decode(
                    input_ids, compiled_cache_state
                )
            else:
                logits = _last_token_logits(self.model, input_ids, caches)
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
                next_token_array = mx.argmax(logits, axis=-1)
            else:
                # logsumexp 替 nn.log_softmax(mlx_lm 做法,更轻量:减一次 max)
                logprobs = logits - mx.logsumexp(logits, keepdims=True)
                next_token_array = sampler(logprobs)

            # BF16 compiled path 做一拍流水：next_token 保持在 GPU 上，先提交
            # 下一次 RWKV 前向，再让 CPU 同步读取当前 token。最后一个已知
            # max_tokens step 不预取，避免固定多算一次；EOS/stop 未知，只能
            # 投机提交，但命中时丢弃 future_state，不污染返回 cache。
            future_logits = None
            future_cache_state = None
            if self._async_decode and step + 1 < cfg.max_tokens:
                if compiled_cache_state is None:
                    # step 0 prefill 完成后，将 ArraysCache / State list 提升为
                    # compiled graph 的显式 array tree。只做浅拷贝，不搬数据。
                    compiled_cache_state = _read_cache_state(caches)
                future_logits, future_cache_state = self._compiled_decode(
                    next_token_array.reshape((1, 1)), compiled_cache_state
                )
                mx.async_eval(future_logits)

            # 唯一 CPU 同步点放在下一步图提交之后。GPU 一旦算出 argmax / sample
            # 就能继续跑 future_logits，不必等待下面的文本和停止条件处理。
            next_token = int(next_token_array)
            token_ready_at = time.perf_counter()
            t_step = token_ready_at - t_step_start
            if step == 0:
                t_prefill = t_step
                # Decode 吞吐按“首 token 已交给 CPU → 后续 token 就绪”的
                # 用户可见区间计时；所有后端统一口径。
                generation_started_at = token_ready_at
            else:
                decode_steps += 1
                t_gen = token_ready_at - generation_started_at
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
            if future_logits is not None:
                # 只有确认当前 token 非终止后才推进 state；终止路径保留
                # compiled_cache_state(即喂入当前 token 之前的干净 cache)。
                pipelined_logits = future_logits
                compiled_cache_state = future_cache_state

        # flush：EOS/max_tokens 或 stop_sequence 前仍可能有尚未确认的安全文本。
        if stop_reason != "stop_sequence":
            final_text = self.tokenizer.decode(generated)
            emit_safe_text(final_text)

        # cache 洁净性(§2.2):eos/max_tokens 未产生越界 token,干净;stop_sequence 脏。
        cache_clean = stop_reason != "stop_sequence"

        # 编译路径内部以纯函数 state 续传；成功生成后一次性写回调用方原 cache，
        # 保留 ChatSession 依赖的 cache 对象身份。abort 抛异常时不提交，且
        # ChatSession 会按既有规则丢弃该 cache、强制下轮重放。
        if compiled_cache_state is not None:
            _write_cache_state(caches, compiled_cache_state)

        return GenerationResult(
            text=final_text,
            display_token_ids=generated,
            stop_reason=stop_reason,
            elapsed=time.perf_counter() - t0,
            used_state=state is not None,
            config=cfg,
            prompt_tokens=prompt_token_count,
            prompt_time=t_prefill,
            generation_time=t_gen,
            decode_steps=decode_steps,
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
            raise ValueError("A/B comparison requires a state")
        cfg = config or GenerationConfig()
        return ABResult(
            prompt=prompt,
            with_state=self.generate(prompt, state=state, config=cfg),
            baseline=self.generate(prompt, state=None, config=cfg),
        )
