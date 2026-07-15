#!/usr/bin/env python3
"""推理速度 benchmark —— 三档(1k/2k/4k token)× 三指标表格测 prefill 与 decode 速度。

引擎(InferenceEngine.generate)内置了 prefill/decode 分段计时:
  step 0 = 整个 prompt 并行 prefill
  step>0 = 首 token 就绪后的连续 decode 墙钟区间，按实际前向次数计数
所以无需自己插桩,直接读 GenerationResult.prompt_tps / generation_tps。

默认一次跑 3 个 prefill 档位(1024 / 2048 / 4096 token),每档多次取中位数,
最后打印一张 3×3 表格:

  Prefill tokens │  prefill t/s  │  decode t/s  │  decode ms/token

用法(PYTHONPATH=src 是必须的,src layout):
  PYTHONPATH=src .venv/bin/python tools/bench_inference.py
  PYTHONPATH=src .venv/bin/python tools/bench_inference.py --model models/converted/rwkv7-g1d-0.4b
  PYTHONPATH=src .venv/bin/python tools/bench_inference.py --runs 7 --max-tokens 256

默认配置:
  - 模型 rwkv7-g1h-1.5b(g1h 是 reasoning 模型 → 默认开 reasoning 方言,否则降智)
  - 三档 prefill: 1024 / 2048 / 4096 token(素材库约 1900 token,超出自动重复填充)
  - qa 模板 + reasoning + think fast(对齐官方 enable_thinking=False 渲染)
  - 贪心 temperature=0(可复现),关掉重复惩罚(纯测速度)
  - 正式测量前先用首档全局 warmup 4 次,排除进程级编译/频率爬升
  - 每档再跑 1 次 shape/allocator warmup 丢弃
  - 正式 runs 连续执行,中间不 clear allocator cache,测常驻 serve 的稳态速度
  - 档位切换时清空空闲 memory cache,随后先 warmup 再计时

慢测模式(--slow):全局预热后、以及后续档位之间均冷却 60s
(可带数字如 --slow 90),使每个正式档位都从“已编译、已散热”的状态开始。
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from statetuner.core import load_model
from statetuner.inference import (
    GenerationConfig,
    InferenceEngine,
    render_prompt,
    with_template_stops,
)
from statetuner.runtime import apply_cache_limit

DEFAULT_MODEL = "models/converted/rwkv7-g1h-1.5b"
# 真实语义素材:让模型有内容可总结,prompt 不会显得突兀。
# prefill 内容语义对速度没影响(只取决于 token 数 × 模型大小),但保持有意义。
# 展开到 ~2000 token,使 1024 截取前半、2048 刚好一遍、4096 最多重复一次,
# 避免长 prompt 下单调重复同一段落。
_FILLER_TEXT = """量子计算是一种利用量子力学原理处理信息的计算范式,其理论基石包括叠加、纠缠和干涉三大核心资源。经典比特只能取 0 或 1 两种确定状态,而量子比特可以同时处于 0 和 1 的叠加态。当多个量子比特相互纠缠时,它们构成的状态空间随比特数呈指数增长,n 个量子比特可以同时表示 2 的 n 次方种状态的叠加。量子算法正是通过精心设计的干涉,让正确答案的概率幅相互增强、错误答案相互抵消,从而在某些问题上实现超越经典计算机的加速。

量子计算的概念最早由物理学家理查德·费曼在 1981 年提出。他指出,用经典计算机模拟量子系统会遭遇指数级的计算复杂度瓶颈,而用量子系统本身来模拟则自然高效。1985 年,大卫·多伊奇形式化了通用量子图灵机的概念,并提出了第一个量子算法——多伊奇算法,证明了量子计算在某些问题上确实可以优于经典计算。此后,量子计算从纯理论探索逐步走向算法设计和物理实现的实验阶段。

1994 年,彼得·秀尔提出了著名的 Shor 算法,能在多项式时间内完成大整数分解。这一算法对广泛使用的 RSA 加密体系构成了根本性威胁,因为 RSA 的安全性建立在经典计算机难以快速分解大整数的假设之上。Shor 算法的发现极大地推动了量子计算领域的发展,也促使各国政府和科技巨头开始认真投资量子计算研究。1996 年,洛夫·格罗弗提出了 Grover 算法,能在无序数据库搜索中提供平方级加速,虽然加速幅度不如 Shor 算法,但适用范围更广,包括密码碰撞、优化和图论问题等。

在物理实现层面,量子计算面临的核心挑战是如何在保持量子相干性的同时实现高保真度的量子门操作。目前主流的物理平台包括超导量子比特、离子阱、中性原子阵列、光量子和拓扑量子比特等。超导量子比特以谷歌的 Sycamore 和 IBM 的量子处理器为代表,利用超导电路中的约瑟夫森结构造人工原子,优势在于与现有半导体工艺兼容、门操作速度快,但需要极端低温环境(约 15 毫开尔文)且相干时间相对较短。离子阱平台以 IonQ 和 Quantinuum 为代表,用电磁场囚禁单个离子作为量子比特,相干时间长、门保真度高,但门操作速度较慢且扩展到大规模较为困难。中性原子阵列是近年来快速发展的方向,用光镊捕获中性碱土原子,天然支持高连接度的二维或三维阵列,且能在同一平台上实现数百到数千个量子比特的规模。

量子纠错是实现通用容错量子计算的关键技术。由于量子态极易受到环境噪声干扰而退相干,直接操作的物理量子比特错误率太高,无法支撑长计算的可靠运行。量子纠错码通过将多个有噪的物理量子比特编码为一个低错误率的逻辑量子比特来解决这个问题。表面码是目前研究最充分的方案,它只需要最近邻相互作用,适合二维平面架构,但代价是每个逻辑量子比特需要约一千个物理量子比特。要实现有实际价值的大规模计算,可能需要数百万个物理量子比特。近年来,中性原子平台上利用里德堡态实现了超越经典模拟的量子电路演示,谷歌和 Quantinuum 也在实验中观测到逻辑错误率随码距增加而下降的关键里程碑,标志着量子纠错正从理论走向现实。

量子计算的应用前景涵盖多个领域。在密码学领域,除了对公钥密码的威胁,量子计算也催生了量子密钥分发等新型安全通信技术。在材料科学和药物研发领域,量子计算机有望精确模拟分子和固体的电子结构,加速新材料和新药的发现。在优化和机器学习领域,量子算法可能为组合优化、采样和线性代数运算提供新的工具。然而,大多数实际应用仍需要容错级别的量子计算机,目前处于含噪中等规模量子(NISQ)时代,量子优势的实验演示主要局限于经过精心设计的人造问题,距离广泛实用的量子优越性仍有相当长的道路。

除量子计算外,量子技术的版图还包括量子通信和量子精密测量。量子通信利用量子不可克隆定理实现原理上无条件安全的密钥分发,中国的墨子号卫星完成了千公里级的星地量子密钥分发实验。量子精密测量利用量子纠缠和压缩态突破经典测量精度极限,在引力波探测、原子钟和磁场传感等领域展现出重要价值。这三者共同构成了第二次量子革命的技术核心,正在从基础研究向工程化和产业化迈进。

尽管量子计算前景广阔,但前进道路上仍存在诸多严峻挑战。退相干问题使得量子态难以长时间维持,需要发展更好的材料、隔离技术和动力学解耦方案。门保真度的提升依赖于更精细的控制电子学和校准算法。可扩展性要求在增加比特数的同时不牺牲单比特性能,这对系统集成和低温电子学提出了极高要求。此外,量子算法的实际价值仍在持续探索中,需要在硬件进步的推动下不断发现能够真正解决实际问题的量子应用。业界普遍认为,实现通用容错量子计算可能还需要十年甚至更长时间,但每一次硬件性能的提升都在缩短这一距离。"""


def build_prompt_to_tokens(
    tokenizer, target_tokens: int, question: str
) -> str:
    """构造一个 token 数 ≥ target_tokens 的 prompt。

    用 question + 重复 _FILLER_TEXT 填充,逐句追加到刚超过 target(宁可略超,不欠)。
    返回的 prompt 编码后 token 数 ≥ target,超出量 ≤ 一个句子(通常 < 60 token)。
    prefill 速度测量只在乎 token 数足够大,略超目标完全可接受。
    """
    # 粗填:重复 filler 直到刚超过 target
    full = question
    while len(tokenizer.encode(full)) < target_tokens:
        full += _FILLER_TEXT
    # 已经 ≥ target,逐句回退到"不超过 target 的最大值"再加最后一句,
    # 保证最终略超 target(或恰好等于粗填值,若粗填首轮就达标)。
    sentences = full.replace("。", "。\n").split("\n")
    # 找到"加到第 k 句开始超过 target"的分界
    acc = ""
    k = 0
    for i, s in enumerate(sentences):
        if len(tokenizer.encode(acc + s)) >= target_tokens:
            k = i
            break
        acc += s
    # 返回 acc + 第 k 句:必然 ≥ target(因为 acc+s 已超过)
    return acc + sentences[k] if k < len(sentences) else full


def quantize_predicate(path, module) -> bool:
    """nn.quantize 的 class_predicate:只量化 input_dim 可被 group_size 整除的 Linear。

    RWKV7 的 LoRA 低秩层(如 v_low_rank_dim=32)最后一维整除不了 64,
    `mx.quantize` 会抛 ValueError。这类层参数量极小,跳过不影响提速与压缩。
    0.4B 实测跳 23/334 个,1.5B 同构。
    """
    if type(module).__name__ != "Linear":
        return False
    return module.weight.shape[1] % 64 == 0


def run_once(
    engine: InferenceEngine, wrapped: str, cfg: GenerationConfig, state=None
):
    """跑一次生成,返回 GenerationResult。

    这里故意不 clear MLX memory cache：同一档的正式 runs 要测常驻进程
    的稳态速度，allocator 冷启动已由该档 warmup 排除。
    """
    return engine.generate(wrapped, state=state, config=cfg)


def detect_precision(model_path: Path, runtime_quantize: str = "none") -> str:
    """从 config.json 识别实际权重精度,避免量化目录被误标 bf16。"""
    if runtime_quantize != "none":
        return f"{runtime_quantize}-runtime"
    try:
        config = json.loads((model_path / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return "unknown"
    quantization = config.get("quantization") or config.get("quantization_config")
    if isinstance(quantization, dict):
        bits = quantization.get("bits")
        return f"int{bits}" if bits is not None else "quantized"
    dtype = str(config.get("torch_dtype", config.get("dtype", "unknown"))).lower()
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
        dtype, dtype
    )


def median_and_range(values: list[float]) -> tuple[float, float, float]:
    """返回 (p50, min, max)；空列表统一返回 0。"""
    if not values:
        return (0.0, 0.0, 0.0)
    return (statistics.median(values), min(values), max(values))


def stability_warnings(
    rows: list[dict], threshold_pct: float = 3.0
) -> list[str]:
    """按预先锁定的 3% 阈值识别启动/降频污染。

    - 单档 prefill/decode 的 (max-min)/p50 超阈值 → 该档不稳定。
    - 多档 decode p50 跨档差异超阈值 → 全局运行状态不一致。
    """
    warnings: list[str] = []

    def span_pct(center: float, bounds: tuple[float, float]) -> float:
        return (bounds[1] - bounds[0]) / center * 100 if center > 0 else 0.0

    for row in rows:
        for label, center_key, range_key in (
            ("prefill", "prefill_tps", "prefill_range"),
            ("decode", "decode_tps", "decode_range"),
        ):
            spread = span_pct(row[center_key], row[range_key])
            if spread > threshold_pct:
                warnings.append(
                    f"{row['prompt_tokens']} tok {label} 波动 {spread:.1f}% "
                    f"> {threshold_pct:.1f}%,该档不可用于性能裁决"
                )

    decode_centers = [row["decode_tps"] for row in rows if row["decode_tps"] > 0]
    if len(decode_centers) > 1:
        center = statistics.median(decode_centers)
        spread = (max(decode_centers) - min(decode_centers)) / center * 100
        if spread > threshold_pct:
            warnings.append(
                f"跨档 decode p50 差异 {spread:.1f}% > {threshold_pct:.1f}%,"
                "可能仍有启动或热降频污染"
            )
    return warnings


def cooldown(seconds: int) -> None:
    """档间冷却:清空空闲 memory cache 并等待 N 秒散热。"""
    import mlx.core as mx

    mx.clear_cache()
    print(f"[cool] 档间冷却 {seconds}s(降低热降频干扰)...", end="", flush=True)
    for remaining in range(seconds, 0, -1):
        print(f"\r[cool] 档间冷却:剩余 {remaining:>3d}s ", end="", flush=True)
        time.sleep(1)
    print("\r[cool] 冷却完成。" + " " * 64)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RWKV-7 推理速度 benchmark(prefill + decode tps)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"HF 格式模型目录(默认 {DEFAULT_MODEL})"
    )
    ap.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096],
        help="prefill token 档位列表(默认 1024 2048 4096)",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=5,
        help="每档正式跑的次数(不含 warmup,默认 5)",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=128, help="每次生成的最大 token 数(默认 128)"
    )
    ap.add_argument(
        "--template",
        choices=["raw", "qa", "instruction"],
        default="qa",
        help="prompt 模板(默认 qa)",
    )
    ap.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否套 reasoning 方言(默认开;G1 系列必须开,否则降智)",
    )
    ap.add_argument(
        "--think",
        choices=["off", "fast", "on"],
        default="fast",
        help="think 档位(默认 fast;reasoning=False 时强制 off)",
    )
    ap.add_argument(
        "--state", default=None, help="可选 state 文件(.npz/.pth),不传则无 state"
    )
    ap.add_argument(
        "--cache-limit-gb",
        default="auto",
        help="MLX cache 上限:'auto'=物理内存×25%% 或 GB 数(默认 auto)",
    )
    ap.add_argument(
        "--no-warmup",
        action="store_true",
        help="跳过全局和分档 warmup(调试用,不可用于性能裁决)",
    )
    ap.add_argument(
        "--warmup-runs",
        type=int,
        default=4,
        help="正式测量前的全局 warmup 次数(用首档,默认 4)",
    )
    ap.add_argument(
        "--slow",
        type=int,
        nargs="?",
        const=60,
        default=0,
        metavar="SECONDS",
        help="全局预热后与各档之间冷却 N 秒(不带数字默认 60s)",
    )
    ap.add_argument(
        "--quantize",
        choices=["none", "int8"],
        default="none",
        help="加载后运行时量化: none=bf16 原样, int8=group_size=64/bits=8"
        "(跳过 LoRA 低秩小层,见 quantize_predicate)",
    )
    ap.add_argument(
        "--decode-backend",
        choices=["eager", "compile", "pipeline"],
        default="pipeline",
        help="decode 后端:eager / 同步 mx.compile / mx.compile+async(默认 pipeline)",
    )
    args = ap.parse_args()

    if args.runs <= 0:
        ap.error("--runs 必须 > 0")
    if args.warmup_runs <= 0:
        ap.error("--warmup-runs 必须 > 0")
    if args.max_tokens < 2:
        ap.error("--max-tokens 必须 >= 2，否则没有 step>0 的 decode 可测")
    if any(level <= 0 for level in args.levels):
        ap.error("--levels 必须全部 > 0")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"模型目录不存在: {model_path}")

    think = args.think if args.reasoning else "off"

    # ── 时序铁律:cache_limit 必须在 load_model 之前 ──
    limit_bytes = apply_cache_limit(args.cache_limit_gb)
    if limit_bytes is not None:
        print(f"[setup] MLX cache limit = {limit_bytes / 1e9:.2f} GB")

    print(f"[setup] 加载模型: {model_path}")
    t0 = time.perf_counter()
    mdl, tok = load_model(model_path, patch=False)
    print(f"[setup] 模型加载完成({time.perf_counter() - t0:.1f}s)")

    if args.quantize == "int8":
        import mlx.nn as nn

        print("[setup] 运行时量化:int8 group_size=64 ...", end="", flush=True)
        t0 = time.perf_counter()
        # 量化整个外层模型,否则会遗漏 decode 每步必跑的 lm_head。
        nn.quantize(mdl, group_size=64, bits=8, class_predicate=quantize_predicate)
        n_q = sum(
            1
            for _, mod in mdl.named_modules()
            if type(mod).__name__ == "QuantizedLinear"
        )
        print(f" 完成({time.perf_counter() - t0:.1f}s, {n_q} 个 QuantizedLinear)")

    engine = InferenceEngine(
        mdl,
        tok,
        compile_decode=args.decode_backend != "eager",
        async_decode=args.decode_backend == "pipeline",
    )

    cfg = with_template_stops(
        GenerationConfig(
            max_tokens=args.max_tokens,
            temperature=0.0,  # 贪心,可复现
            seed=42,
            presence_penalty=0.0,  # 关掉重复惩罚,纯测速度
            frequency_penalty=0.0,
        ),
        args.template,
    )
    state_arg = str(args.state) if args.state else None
    question = "请根据以上材料,用三句话总结量子计算的核心原理与主要挑战。"
    precision = detect_precision(model_path, args.quantize)

    print(
        f"[setup] 精度={precision} 模板={args.template} "
        f"reasoning={args.reasoning} think={think} | "
        f"runs={args.runs}/档 max_tokens={args.max_tokens} mode=steady | "
        f"global_warmup={0 if args.no_warmup else args.warmup_runs} | "
        f"decode={engine.decode_backend}"
    )
    print()

    import mlx.core as mx

    # 预先构造各档 prompt：全局 warmup 和正式测量必须用同一份首档输入。
    prepared_levels = []
    for level in args.levels:
        raw_prompt = build_prompt_to_tokens(tok, level, question)
        wrapped = render_prompt(
            raw_prompt, args.template, reasoning=args.reasoning, think=think
        )
        prepared_levels.append((level, wrapped, len(tok.encode(wrapped))))

    # ── 进程级全局 warmup ──
    # 用首档完整生成多次，让 decode 图/内核缓存/频率爬升全部收敛。
    if not args.no_warmup:
        _, warmup_prompt, _ = prepared_levels[0]
        print(f"[warmup] 全局预热 {args.warmup_runs} 次(全部丢弃)")
        for i in range(args.warmup_runs):
            result = run_once(engine, warmup_prompt, cfg, state=state_arg)
            print(
                f"  warmup {i + 1}/{args.warmup_runs}: "
                f"Prompt {result.prompt_tps:.1f} t/s | "
                f"Decode {result.generation_tps:.1f} t/s"
            )
        print()
        # slow 模式下先预热再冷却，使第一档与后续档一样：
        # 内核已编译，但测量前有完整的散热窗口。
        if args.slow > 0:
            cooldown(args.slow)

    # ── 逐档测试 ──
    # rows[level] = {prompt_tokens, prefill_tps, decode_tps, ms_per_decode_token}
    rows = []
    for idx, (level, wrapped, prompt_token_count) in enumerate(prepared_levels):
        # 档间冷却(--slow > 0 时,两档之间等 N 秒;最后一档不空等)
        if args.slow > 0 and idx > 0:
            cooldown(args.slow)

        print(f"[bench] ── prefill {level} token(实际 {prompt_token_count})──")

        # 档位间可清空空闲 memory cache；随后必须 warmup，且 warmup
        # 与正式 runs 之间不再 clear，才是常驻 serve 的稳态口径。
        mx.clear_cache()
        if not args.no_warmup:
            run_once(engine, wrapped, cfg, state=state_arg)

        prefill_tps_list = []
        decode_tps_list = []
        ms_per_token_list = []
        gen_tokens_list = []
        for i in range(args.runs):
            r = run_once(engine, wrapped, cfg, state=state_arg)
            prefill_tps_list.append(r.prompt_tps)
            decode_tps_list.append(r.generation_tps)
            gen_tokens_list.append(r.token_count)
            if r.decode_steps > 0:
                ms_per_token_list.append(
                    r.generation_time * 1000 / r.decode_steps
                )
            print(
                f"  run {i + 1}/{args.runs}: {r.summary_line()}  "
                f"(gen_tokens={r.token_count}, decode_steps={r.decode_steps})"
            )

        prefill_p50, prefill_min, prefill_max = median_and_range(prefill_tps_list)
        decode_p50, decode_min, decode_max = median_and_range(decode_tps_list)
        ms_p50, ms_min, ms_max = median_and_range(ms_per_token_list)
        rows.append({
            "level": level,
            "prompt_tokens": prompt_token_count,
            "prefill_tps": prefill_p50,
            "prefill_range": (prefill_min, prefill_max),
            "decode_tps": decode_p50,
            "decode_range": (decode_min, decode_max),
            "ms_per_token": ms_p50,
            "ms_range": (ms_min, ms_max),
            "gen_tokens": statistics.median(gen_tokens_list) if gen_tokens_list else 0.0,
        })
        # 该档正式计时已结束，此处再清理不会污染数据。
        mx.clear_cache()
        print()

    # ── 3×3 表格 ──
    print_table(model_path.name, rows, args.runs, precision)


def print_table(model_name: str, rows: list[dict], runs: int, precision: str) -> None:
    """打印 p50 汇总表 + min–max 稳定性范围。"""
    qtag = f" [{precision}]"
    print("=" * 62)
    print(f"  {model_name}{qtag}  (runs={runs}/档, steady, 排除 warmup)")
    print("=" * 62)
    header = f"{'Prefill':>10} │ {'prefill p50':>12} │ {'decode p50':>11} │ {'ms/token':>9}"
    print(header)
    print("─" * len(header))
    for r in rows:
        print(
            f"{r['prompt_tokens']:>7} tok │ "
            f"{r['prefill_tps']:>12.1f} │ "
            f"{r['decode_tps']:>11.1f} │ "
            f"{r['ms_per_token']:>9.1f}"
        )
    print("=" * 62)
    print("  波动范围(min–max):")
    for r in rows:
        p_lo, p_hi = r["prefill_range"]
        d_lo, d_hi = r["decode_range"]
        m_lo, m_hi = r["ms_range"]
        print(
            f"  {r['prompt_tokens']:>7} tok │ prefill {p_lo:.1f}–{p_hi:.1f} t/s │ "
            f"decode {d_lo:.1f}–{d_hi:.1f} t/s │ {m_lo:.1f}–{m_hi:.1f} ms"
        )
    warnings = stability_warnings(rows)
    if warnings:
        print("  稳定性警告:")
        for warning in warnings:
            print(f"  [warn] {warning}")
    else:
        print("  稳定性检查:通过(单档波动与跨档 decode 差异均 <= 3%)")
    print(
        "  注:汇总值是 p50 中位数;decode 只按 step>0 的实际前向次数计算。"
    )


if __name__ == "__main__":
    main()
