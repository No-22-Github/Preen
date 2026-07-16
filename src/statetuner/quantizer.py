"""离线 int8 量化器:bf16 HF 模型目录 → int8 量化 HF 模型目录。

二阶段量化的第二步(转换器输出 bf16 后,本模块做精度转换)。
设计上**复用 mlx-lm 量化基础设施**,不自己实现打包/反量化:

  load(src, return_config=True)     # 加载 bf16 模型(lazy)
  quantize_model(model, config, …)  # mlx_lm 自动调用 RWKV7 自带的 quant_predicate
                                    #   (rwkv7.py: lora.2/embeddings 强制 bits=8,
                                    #    其余默认;自动跳过 input_dim 整除不了
                                    #    group_size 的层)
  save(out, src, model, …)          # mlx_lm 写盘:config 自动加 quantization +
                                    #   quantization_config 字段;safetensors 键名
                                    #   自动处理(weight/scales/biases);tokenizer
                                    #   与 remote .py 自动复制

产物是标准 mlx-community 格式的量化模型目录,`core.load_model`(走 `mlx_lm.load`)
会**透明识别** config 里的 quantization 字段并自动替换为 QuantizedLinear——
推理层(preview/chat/serve/eval/export)零改动即可加载 int8 模型。

精度边界(见 docs/decision-precision.md + AGENTS.md):
  - **推理专用**:int8 产物只用于推理加速(1.5B 实测 decode 1.7x 提速)。
  - **训练拒绝**:state tuning 的精度契约是「权重 bf16 + state fp32」,
    量化模型不能训练——`service.validate_training_request` 会拦下(config 层)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

PathLike = Union[str, Path]

# 进度回调签名:(phase, message, current, total)
ProgressCallback = Optional[Callable[[str, str, Optional[int], Optional[int]], None]]
LogCallback = Optional[Callable[[str], None]]


def quantize(
    src_model: PathLike,
    out_path: PathLike,
    *,
    bits: int = 8,
    group_size: int = 64,
    progress_callback: ProgressCallback = None,
    log: LogCallback = None,
) -> Dict[str, Any]:
    """加载 bf16 模型 → int8 量化 → 存量化模型目录。返回 summary。

    Args:
        src_model: 源 bf16 HF 模型目录(Preen convert-model 的产物)。
        out_path:  输出量化模型目录。**必须不存在或为空**(mlx_lm.save 约束)。
        bits:      量化位数(默认 8 = int8)。
        group_size: 量化组大小(默认 64;须能整除待量化层的 input_dim)。
        progress_callback: (phase, message, current, total) 四元回调,phase 取值
                           load / quantize / save,供 CLI 发 tool_events 进度。
        log: 人类可读日志回调(stderr 用)。

    Returns:
        summary dict:{bits, group_size, quantized_layers, src, out}
    """
    src = Path(src_model)
    out = Path(out_path)
    say = log or (lambda _msg: None)
    notify = progress_callback or (lambda *_a, **_k: None)

    # ── 1. 校验 ──
    if not src.is_dir():
        raise ValueError(f"Source model directory does not exist: {src}")
    _src_config = src / "config.json"
    if not _src_config.is_file():
        raise ValueError(f"Source directory has no config.json and is not a valid HF model directory: {src}")
    # 输出目录约束:mlx_lm.save 要求不存在或空(它内部会 makedirs)。
    if out.exists() and out.is_dir() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}; clear it or choose another path")
    if out.exists() and not out.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {out}")

    # ── 2. 加载源模型(lazy,避免权重实例化浪费)──
    say(f"Loading model {src}")
    notify("load", "Loading BF16 model", None, None)
    import mlx_lm

    model, tokenizer, config = mlx_lm.load(
        str(src),
        tokenizer_config={"trust_remote_code": True},
        return_config=True,
    )

    # ── 3. 量化(mlx_lm 自动调用 RWKV7 的 quant_predicate)──
    say(f"Quantizing: bits={bits} group_size={group_size}")
    notify("quantize", f"Quantizing model (bits={bits})", None, None)
    from mlx_lm.utils import quantize_model

    model, config = quantize_model(
        model, config, group_size=group_size, bits=bits
    )

    # 统计量化层数(QuantizedLinear 替换数)
    inner = getattr(model, "model", model)
    quantized_layers = sum(
        1
        for _, mod in inner.named_modules()
        if type(mod).__name__ == "QuantizedLinear"
    )
    say(f"Quantization complete: {quantized_layers} QuantizedLinear layers")

    # ── 4. 写盘(mlx_lm.save 自动处理 config/safetensors/remote .py)──
    say(f"Saving quantized model -> {out}")
    notify("save", "Writing quantized model directory", None, None)
    from mlx_lm.utils import save as mlx_save

    mlx_save(str(out), str(src), model, tokenizer, config)

    # ── 5. 同步 tokenizer 文件(覆盖 mlx_lm.save 的 tokenizer 重写)──
    # mlx_lm.save 用 HF tokenizer 标准流程保存,会改写 tokenizer_config、
    # 把 rwkv_vocab_v20230424.txt 重命名成 vocab.txt、丢失 added_tokens.json /
    # special_tokens_map.json。但 RWKV 自定义 tokenizer(hf_rwkv_tokenizer.py)
    # 硬编码要找 rwkv_vocab_v20230424.txt(VOCAB_FILES_NAMES),重命名后加载失败。
    # 解法:权重/config 是量化产物(保留),tokenizer 相关文件从源目录原样覆盖。
    _sync_tokenizer_files(src, out, say)

    notify("save", "Completed", None, None)
    return {
        "bits": bits,
        "group_size": group_size,
        "quantized_layers": quantized_layers,
        "src": str(src),
        "out": str(out),
    }


# tokenizer 相关文件:这些从源目录原样复制到量化产物,覆盖 mlx_lm.save 的重写。
# (权重 model.safetensors* + config.json + remote .py 是量化产物,不动。)
_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "rwkv_vocab_v20230424.txt",  # RWKV 自定义 tokenizer 硬编码的 vocab 文件名
    "chat_template.jinja",       # 若源目录有则复制(部分目录没有)
)


def _sync_tokenizer_files(src: Path, out: Path, say: Callable[[str], None]) -> None:
    """把源目录的 tokenizer 文件原样覆盖到量化产物目录。

    删除 mlx_lm.save 产生的非标准文件(vocab.txt, README.md 等 tokenizer 侧副产物),
    再从源目录复制 _TOKENIZER_FILES 列出的文件。
    """
    import os

    # 删除 mlx_lm.save 产生的、源目录没有的 tokenizer 副产物
    for stale in ("vocab.txt",):
        stale_path = out / stale
        if stale_path.exists() and not (src / stale).exists():
            stale_path.unlink()

    copied = []
    for name in _TOKENIZER_FILES:
        src_file = src / name
        if src_file.is_file():
            import shutil

            shutil.copy2(src_file, out / name)
            copied.append(name)
    if copied:
        say(f"Synchronized tokenizer files: {', '.join(copied)}")
