"""
RWKV-7 原生 .pth → fla HF(safetensors)正式转换模块。

不依赖 fla/triton/torch —— 用纯 Python mmap 流式读 pth 权重
(statetuner.pth_io.iter_pth,与 torch.load 逐字节等价),按官方
convert_from_rwkv7.py 的键名映射规则搬运,
用仓库内置 fixture(从同架构 0.1B fla 模型生成的 ndim 模板)做 ground truth 维度校验,
最后存 safetensors + config.json。tokenizer 文件也已 vendor 进 assets/,转换时无需任何外部下载。

映射规则 (与 fla-org/flash-linear-attention 的 convert_from_rwkv7.py 等价):
  - 顶层:  emb.weight→model.embeddings.weight
           ln_out.{weight,bias}→model.norm.{weight,bias}
           head.weight→lm_head.weight
  - 层内:  ln0→pre_norm, ln1→attn_norm, ln2→ffn_norm
           att.{receptance,key,value,output}→{r,k,v,o}_proj
           att.ln_x→g_norm
           att.{w,a,g,v}{0,1,2}→{w,a,g,v}_lora.lora.{2.bias,0.weight,2.weight}
               (其中 *1/*2 的 weight 要转置)
           att.x_*、att.r_k、att.k_a、att.k_k 保持位置名
           blocks.0.att.{v0,v1,v2} 被丢弃 (layer 0 无 v_lora)
  - shape: [1,1,hidden] 的非 x_ 键 squeeze; x_ 键保留 (copy_ 广播)

用法(最短形式,fixture + vendored tokenizer 已内置仓库):
  python convert_rwkv7_to_hf.py \
      --rwkv7 models/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
      --output models/converted/rwkv7-g1d-0.4b --precision bf16

可选覆盖(上游 schema 漂移时的逃生通道):
  --reference <fla model.safetensors>   活模型校验,覆盖内置 fixture
  --tokenizer-src <dir>                 指定 tokenizer 来源目录
"""
import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import ml_dtypes

from .pth_io import PthTensorInfo, iter_pth, peek_pth_tensors


ProgressCallback = Callable[[str, str, Optional[int], Optional[int]], None]


def normalize_layer_key(k):
    """键名归一化: model.layers.N.X → (True, 'model.layers.{N}.X');
    其他 → (False, k)。供 load_reference_template 和 fixture 生成共用。"""
    m = re.match(r"model\.layers\.(\d+)\.(.+)", k)
    if m:
        return True, "model.layers.{N}." + m.group(2)
    return False, k


def load_reference_template(ref_path):
    """从同架构的 fla 模型读取键名+shape 作为 ground truth 模板。

    返回 dict[str, tuple]: 每层只取 layer 0 的相对键名 → shape (去掉层数)。
    """
    from safetensors import safe_open
    f = safe_open(ref_path, framework="np")
    template = {}      # 相对键名 (无 layer idx) → shape
    top_keys = {}      # 顶层键 → shape
    for k in sorted(f.keys()):
        t = f.get_tensor(k)
        is_layer, rel = normalize_layer_key(k)
        if is_layer:
            if rel not in template:
                template[rel] = tuple(t.shape)
        else:
            if k not in top_keys:
                top_keys[k] = tuple(t.shape)
    f.__exit__(None, None, None)
    return template, top_keys


def _bundled_path(*parts: str) -> Path:
    """开发态读仓库 assets，wheel/app 内读 statetuner/assets。"""
    packaged = Path(__file__).resolve().parent.joinpath("assets", *parts)
    if packaged.exists():
        return packaged
    repository = Path(__file__).resolve().parents[2]
    if parts[0] == "rwkv7_hf_template.json":
        return repository / "tools" / "fixtures" / parts[0]
    return repository / "assets" / Path(*parts)


_DEFAULT_FIXTURE = _bundled_path("rwkv7_hf_template.json")
_DEFAULT_TOKENIZER_SRC = _bundled_path("rwkv_world_tokenizer")


def load_template_from_fixture(fixture_path):
    """从仓库内置 fixture JSON 加载校验模板。

    fixture 只存 ndim(与现有校验逻辑一致——只比维度数,不比绝对 shape)。
    返回与 load_reference_template 同构的 (template, top_keys):把每个 ndim
    包装成 (0,) * ndim 的伪 shape,使校验判定处的 len(ref_shape) == weight.ndim
    对 fixture 和活模型完全同构,无需改校验代码。
    """
    with open(fixture_path, "r", encoding="utf-8") as fh:
        fx = json.load(fh)
    template = {k: (0,) * v for k, v in fx["layer_keys"].items()}
    top_keys = {k: (0,) * v for k, v in fx["top_keys"].items()}
    return template, top_keys


def infer_config(weights):
    """从 pth 权重的 shape 推断 RWKV7Config 所需字段。"""
    config = {}
    config["vocab_size"] = weights["emb.weight"].shape[0]
    config["hidden_size"] = weights["blocks.0.ffn.key.weight"].shape[1]
    config["intermediate_size"] = weights["blocks.0.ffn.key.weight"].shape[0]
    config["hidden_ratio"] = (
        weights["blocks.0.ffn.key.weight"].shape[0]
        / weights["blocks.0.ffn.key.weight"].shape[1]
    )
    # 层数
    n = 0
    while f"blocks.{n}.ffn.key.weight" in weights:
        n += 1
    config["num_hidden_layers"] = n
    config["decay_low_rank_dim"] = weights["blocks.0.att.w1"].shape[1]
    config["gate_low_rank_dim"] = weights["blocks.0.att.g1"].shape[1]
    config["a_low_rank_dim"] = weights["blocks.0.att.a1"].shape[1]
    try:
        config["v_low_rank_dim"] = weights["blocks.1.att.v1"].shape[1]
    except KeyError:
        config["v_low_rank_dim"] = 32
    config["head_dim"] = 64
    config["num_heads"] = config["hidden_size"] // 64
    config["value_dim"] = [config["hidden_size"]] * n
    return config


# ── 键名映射 (与官方 convert_from_rwkv7.translate_into_fla 等价) ──
EMB_HEAD = {
    "emb.weight": "model.embeddings.weight",
    "ln_out.weight": "model.norm.weight",
    "ln_out.bias": "model.norm.bias",
    "head.weight": "lm_head.weight",
}
PROJ = {
    "receptance": "r_proj",
    "key": "k_proj",
    "value": "v_proj",
    "ln_x": "g_norm",
    "output": "o_proj",
}
UNUSED = ["blocks.0.att.v0", "blocks.0.att.v1", "blocks.0.att.v2"]


def translate(src_name, num_layers):
    """返回 (fla_name, transposed)。空字符串表示丢弃。"""
    if src_name in UNUSED:
        return "", False
    if src_name in EMB_HEAD:
        return EMB_HEAD[src_name], False

    parts = src_name.split(".")
    assert parts[0] == "blocks", f"unexpected key: {src_name}"
    parts[0] = "model.layers"
    li = int(parts[1])
    assert 0 <= li < num_layers
    parts[1] = "{N}"  # 占位,稍后替换为真实层号
    layer_map = {
        "att": "attn", "ffn": "ffn",
        "ln0": "pre_norm", "ln1": "attn_norm", "ln2": "ffn_norm",
    }
    assert parts[2] in layer_map, f"unexpected sub: {src_name}"
    parts[2] = layer_map[parts[2]]

    transposed = False
    # [wvag][012] → {typ}_lora.lora.{位置}
    # 官方映射: 0→2.bias, 1→0.weight, 2→2.weight; num in (1,2) 转置
    if re.match(r"^[wvag][012]$", parts[3]):
        typ, num = parts[3][0], parts[3][1]
        parts[3] = f"{typ}_lora.lora." + {"0": "2.bias", "1": "0.weight", "2": "2.weight"}[num]
        transposed = num in ("1", "2")
    elif parts[2] == "attn" and parts[3] in PROJ:
        parts[3] = PROJ[parts[3]]
    # 其余 (x_*, r_k, k_a, k_k, ffn.x_k) 保持
    return ".".join(parts), transposed


# safetensors numpy dtype → 格式字符串(与 safetensors 官方映射一致)
_NP_TO_ST_DTYPE = {
    "float64": "F64", "float32": "F32", "float16": "F16",
    "bfloat16": "BF16",  # ml_dtypes.bfloat16 的 .name
    "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8",
    "uint8": "U8", "bool": "BOOL",
}


@dataclass(frozen=True)
class _ConvertedTensorSpec:
    source_name: str
    target_name: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    transposed: bool
    dtype: np.dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.target_shape) * self.dtype.itemsize


def _safetensors_header(specs: Sequence[_ConvertedTensorSpec]) -> bytes:
    """由预检 manifest 构造官方兼容、按目标键排序的 safetensors header。"""
    import struct

    names = [spec.target_name for spec in specs]
    seen = set()
    for name in names:
        if name in seen:
            raise ValueError(f"Duplicate target tensor name: {name}")
        seen.add(name)
    if names != sorted(names):
        raise ValueError("safetensors manifest must be sorted by target name")

    meta = {"__metadata__": {"format": "pt"}}
    offset = 0
    for spec in specs:
        meta[spec.target_name] = {
            "dtype": _NP_TO_ST_DTYPE[spec.dtype.name],
            "shape": list(spec.target_shape),
            "data_offsets": [offset, offset + spec.nbytes],
        }
        offset += spec.nbytes
    header = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    header += b" " * (-(8 + len(header)) % 8)
    return struct.pack("<Q", len(header)) + header


def _stream_save_safetensors(
    tensors_iter: Iterable[tuple[str, np.ndarray]],
    path: str,
    specs: Sequence[_ConvertedTensorSpec],
) -> int:
    """根据预检 manifest 直接写单个 safetensors，不落第二份权重中转文件。

    manifest 已包含全部名称、dtype、shape 和 offset，因此可以先写 header，再把
    mmap tensor 按目标键顺序直接追加到目标文件。文件位于转换 staging 目录，失败时
    整个 staging 会被清理。数据写入使用 memoryview，避免 `arr.tobytes()` 整块复制。
    """
    tensor_iter = iter(tensors_iter)
    with open(path, "xb") as out:
        out.write(_safetensors_header(specs))
        for spec in specs:
            try:
                name, arr = next(tensor_iter)
            except StopIteration as exc:
                raise ValueError(
                    f"Missing converted tensor for {spec.target_name}"
                ) from exc
            arr = np.ascontiguousarray(arr)
            if name != spec.target_name:
                raise ValueError(
                    f"Tensor order mismatch: expected {spec.target_name}, got {name}"
                )
            if arr.dtype != spec.dtype or tuple(arr.shape) != spec.target_shape:
                raise ValueError(
                    f"Tensor metadata mismatch for {name}: "
                    f"expected {spec.dtype.name}{spec.target_shape}, "
                    f"got {arr.dtype.name}{tuple(arr.shape)}"
                )
            # ml_dtypes.bfloat16 的 PEP 3118 format='E' 不能直接建 memoryview；
            # 先做零拷贝 uint8 view，其他 dtype 也走同一路径。
            raw = memoryview(arr.view(np.uint8)).cast("B")
            for start in range(0, len(raw), 64 * 1024 * 1024):
                out.write(raw[start:start + 64 * 1024 * 1024])
        try:
            extra_name, _ = next(tensor_iter)
        except StopIteration:
            pass
        else:
            raise ValueError(f"Unexpected converted tensor: {extra_name}")
        out.flush()
        os.fsync(out.fileno())
    return len(specs)


def _build_config(samples: dict[str, tuple[int, ...]], num_hidden_layers: int) -> dict:
    """从收集到的关键张量样本(emb/blocks.0.*/blocks.1.att.v1)推断 RWKV7Config 字段。

    与 `infer_config` 等价,但接受「部分样本 dict」而非全量 weights,供流式转换
    在读完 storage/0 的前几个 tensor 后即调用,无需等全量载入。
    """
    cfg = {}
    ffn_key_shape = samples["blocks.0.ffn.key.weight"]  # tuple
    cfg["vocab_size"] = samples["emb.weight"][0]
    cfg["hidden_size"] = ffn_key_shape[1]
    cfg["intermediate_size"] = ffn_key_shape[0]
    cfg["hidden_ratio"] = ffn_key_shape[0] / ffn_key_shape[1]
    cfg["num_hidden_layers"] = num_hidden_layers
    cfg["decay_low_rank_dim"] = samples["blocks.0.att.w1"][1]
    cfg["gate_low_rank_dim"] = samples["blocks.0.att.g1"][1]
    cfg["a_low_rank_dim"] = samples["blocks.0.att.a1"][1]
    if "blocks.1.att.v1" in samples:
        cfg["v_low_rank_dim"] = samples["blocks.1.att.v1"][1]
    else:
        cfg["v_low_rank_dim"] = 32
    cfg["head_dim"] = 64
    cfg["num_heads"] = cfg["hidden_size"] // 64
    cfg["value_dim"] = [cfg["hidden_size"]] * num_hidden_layers
    return cfg


def _commit_output_directory(stage: Path, output: Path) -> None:
    """提交完整 staging；overwrite 时先保留旧目录，提交失败则立即回滚。"""
    if not output.exists():
        os.replace(stage, output)
        return

    backup = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.preen-backup-",
        dir=output.parent,
    ))
    backup.rmdir()  # os.replace 的目标必须不存在
    try:
        os.replace(output, backup)
        os.replace(stage, output)
    except BaseException:
        if backup.exists():
            # rename 已完成但 Python 尚未返回时也可能收到 SIGINT；先把新目录移回
            # staging，再恢复旧目录，外层 finally 会清理 staging。
            if output.exists() and not stage.exists():
                os.replace(output, stage)
            if not output.exists():
                os.replace(backup, output)
        raise
    else:
        # 新目录已经完整提交；旧目录清理失败不应把成功任务降级成失败。
        shutil.rmtree(backup, ignore_errors=True)


def convert(
    rwkv7_path,
    output,
    ref_path=None,
    tokenizer_src=None,
    precision="bf16",
    *,
    overwrite: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    log: Optional[Callable[[str], None]] = print,
):
    """转换并返回产物摘要；progress_callback 供 CLI/App 工具协议消费。

    先只读 data.pkl 建立完整 manifest 并完成配置、映射、shape、tokenizer 与磁盘预检；
    再从 mmap 按目标键顺序逐 tensor 转换，直接写入同级 staging 目录中的单文件
    safetensors。所有文件验证成功后才提交整个目录，失败或取消不会污染旧产物。
    """
    notify = progress_callback or (lambda phase, message, current, total: None)
    say = log or (lambda message: None)
    rwkv7_path = Path(rwkv7_path)
    output = Path(output)

    # ── 阶段 0：仅扫描元数据并完成所有可预检项 ──
    notify("read", "Reading native .pth weights", None, None)
    say(f"Loading source weights: {rwkv7_path}")
    source_infos = peek_pth_tensors(rwkv7_path, require_stored=True)
    source_shapes = {item.name: item.shape for item in source_infos}
    all_keys = set(source_shapes)
    num_hidden_layers = 0
    while f"blocks.{num_hidden_layers}.ffn.key.weight" in all_keys:
        num_hidden_layers += 1
    if num_hidden_layers == 0:
        raise ValueError("Failed to infer config: source .pth has no RWKV-7 blocks")

    precision = {
        "bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
    }.get(precision, precision)
    dtype_aliases = {
        "bf16": ml_dtypes.bfloat16,
        "fp16": np.float16,
        "fp32": np.float32,
    }
    try:
        dtype = np.dtype(dtype_aliases[precision])
    except KeyError as exc:
        raise ValueError(f"Unsupported precision: {precision}") from exc

    required_config_keys = {
        "emb.weight",
        "blocks.0.ffn.key.weight",
        "blocks.0.att.w1", "blocks.0.att.g1", "blocks.0.att.a1",
    }
    missing_config = sorted(required_config_keys - all_keys)
    if missing_config:
        raise ValueError(
            "Failed to infer config: source .pth missing required keys: "
            + ", ".join(missing_config)
        )
    config_samples = {
        key: source_shapes[key]
        for key in required_config_keys | {"blocks.1.att.v1"}
        if key in source_shapes
    }
    config = _build_config(config_samples, num_hidden_layers)
    say(f"Inferred configuration: layers={config['num_hidden_layers']} "
        f"hidden={config['hidden_size']} vocab={config['vocab_size']} "
        f"ffn={config['intermediate_size']}")

    if ref_path is not None:
        template, top_template = load_reference_template(ref_path)
        say(f"Reference template (live model {ref_path}): "
            f"{len(template)} per-layer keys + {len(top_template)} top-level keys")
    else:
        template, top_template = load_template_from_fixture(_DEFAULT_FIXTURE)
        say(f"Reference template (bundled fixture): "
            f"{len(template)} per-layer keys + {len(top_template)} top-level keys")

    if tokenizer_src is None:
        tokenizer_src = _DEFAULT_TOKENIZER_SRC
    tokenizer_src = Path(tokenizer_src)
    tokenizer_files = [
        "hf_rwkv_tokenizer.py", "tokenizer_config.json",
        "rwkv_vocab_v20230424.txt", "special_tokens_map.json",
        "added_tokens.json",
    ]
    missing_tokenizer = [
        str(tokenizer_src / name)
        for name in tokenizer_files
        if not (tokenizer_src / name).is_file()
    ]
    if missing_tokenizer:
        raise FileNotFoundError(
            "Tokenizer files are missing: " + ", ".join(missing_tokenizer)
        )

    # 由 shape 元数据建立完整目标 manifest；translate/rank/重复键错误均在写盘前暴露。
    specs = []
    reported_layer0 = set()
    target_names = set()
    for item in source_infos:
        rel_name, transposed = translate(item.name, num_hidden_layers)
        if not rel_name:
            say(f"  [skip] {item.name} (unused)")
            continue
        if "{N}" in rel_name:
            li = int(item.name.split(".")[1])
            target_name = rel_name.replace("{N}", str(li))
        else:
            li = -1
            target_name = rel_name
        if target_name in target_names:
            raise ValueError(f"Duplicate target tensor name: {target_name}")
        target_names.add(target_name)

        target_shape = tuple(reversed(item.shape)) if transposed else item.shape
        is_x = "attn.x_" in target_name
        if (not is_x
                and target_shape == (1, 1, config["hidden_size"])):
            target_shape = tuple(dim for dim in target_shape if dim != 1)

        if li == 0 and rel_name in template:
            ref_shape = template[rel_name]
            if len(ref_shape) != len(target_shape):
                raise ValueError(
                    f"Rank validation failed for {target_name}: "
                    f"reference={ref_shape}(ndim={len(ref_shape)}) "
                    f"actual={target_shape}(ndim={len(target_shape)})"
                )
            reported_layer0.add(rel_name)
        if target_name in top_template:
            ref_shape = top_template[target_name]
            if len(ref_shape) != len(target_shape):
                raise ValueError(
                    f"Top-level rank validation failed for {target_name}: "
                    f"reference={ref_shape} actual={target_shape}"
                )
        specs.append(_ConvertedTensorSpec(
            source_name=item.name,
            target_name=target_name,
            source_shape=item.shape,
            target_shape=target_shape,
            transposed=transposed,
            dtype=dtype,
        ))
    specs.sort(key=lambda item: item.target_name)

    uncovered = set(template) - reported_layer0
    for key in sorted(uncovered):
        if "pre_norm" in key:
            say(f"  [note] layer0 is missing {key} (ln0; allowed)")
        else:
            say(f"  [warning] layer0 template key was not covered: {key}")

    output_parent = output.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()) and not overwrite:
        raise ValueError(f"Output directory is not empty: {output}")

    output_bytes = sum(spec.nbytes for spec in specs)
    output_bytes += sum((tokenizer_src / name).stat().st_size for name in tokenizer_files)
    reserve = max(64 * 1024 * 1024, output_bytes // 20)
    free_bytes = shutil.disk_usage(output_parent).free
    if free_bytes < output_bytes + reserve:
        raise OSError(
            "Insufficient disk space for model conversion: "
            f"need about {(output_bytes + reserve) / 1e9:.2f} GB, "
            f"available {free_bytes / 1e9:.2f} GB"
        )

    # config.json
    config_json = {
        "model_type": "rwkv7",
        "architect": ["RWKV7ForCausalLM"],
        "auto_map": {
            "AutoConfig": "fla.models.rwkv7.configuration_rwkv7.RWKV7Config",
            "AutoModelForCausalLM": "fla.models.rwkv7.modeling_rwkv7.RWKV7ForCausalLM",
        },
        "attn_mode": "chunk",
        "hidden_size": config["hidden_size"],
        "hidden_ratio": config["hidden_ratio"],
        "intermediate_size": config["intermediate_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "head_dim": config["head_dim"],
        "num_heads": config["num_heads"],
        "decay_low_rank_dim": config["decay_low_rank_dim"],
        "gate_low_rank_dim": config["gate_low_rank_dim"],
        "a_low_rank_dim": config["a_low_rank_dim"],
        "v_low_rank_dim": config["v_low_rank_dim"],
        "value_dim": config["value_dim"],
        "hidden_act": "sqrelu",
        "max_position_embeddings": 2048,
        "norm_first": True,
        "norm_bias": True,
        "norm_eps": 1e-5,
        "use_cache": True,
        "tie_word_embeddings": False,
        "fuse_norm": True,
        "fuse_cross_entropy": True,
        "fuse_linear_cross_entropy": False,
        "use_l2warp": True,
        "vocab_size": config["vocab_size"],
        "torch_dtype": {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[precision],
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
    }

    # ── 阶段 1：单文件直写 staging；失败/取消只删除 staging ──
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.preen-convert-",
        dir=output_parent,
    ))
    committed = False
    try:
        total = len(specs)

        def _convert_iter():
            source_order = [spec.source_name for spec in specs]
            for index, ((source_name, source_arr), spec) in enumerate(
                zip(iter_pth(rwkv7_path, source_order), specs), 1
            ):
                if source_name != spec.source_name:
                    raise ValueError(
                        f"Source tensor order mismatch: expected {spec.source_name}, "
                        f"got {source_name}"
                    )
                if index == 1 or index == total or index % max(1, total // 100) == 0:
                    notify(
                        "convert", f"Converting tensor {index}/{total}",
                        index, total,
                    )
                weight = source_arr.T if spec.transposed else source_arr
                if tuple(weight.shape) != spec.target_shape:
                    if (len(weight.shape) == 3 and weight.shape[0] == 1
                            and weight.shape[1] == 1):
                        weight = weight.squeeze()
                converted = np.ascontiguousarray(weight.astype(dtype, copy=False))
                yield spec.target_name, converted

        staged_weights = stage / "model.safetensors"
        tensor_count = _stream_save_safetensors(
            _convert_iter(), str(staged_weights), specs,
        )
        notify("write", "Finalizing model files", None, None)
        say(f"Saving weights: {output / 'model.safetensors'} "
            f"({tensor_count} tensors, {precision})")

        with open(stage / "config.json", "x", encoding="utf-8") as fh:
            json.dump(config_json, fh, indent=2, ensure_ascii=False)
        say("Saving config.json")
        for name in tokenizer_files:
            shutil.copy2(tokenizer_src / name, stage / name)
            say(f"Copying tokenizer file: {name}")

        # safe_open 会完整验证 header、offset 连续性与文件长度；提交前至少读一项 shape。
        from safetensors import safe_open
        with safe_open(staged_weights, framework="np") as handle:
            keys = list(handle.keys())
            if len(keys) != tensor_count:
                raise ValueError(
                    f"Safetensors validation failed: expected {tensor_count} tensors, "
                    f"found {len(keys)}"
                )
            if keys:
                handle.get_slice(keys[0]).get_shape()

        _commit_output_directory(stage, output)
        committed = True
    finally:
        if not committed and stage.exists():
            shutil.rmtree(stage)

    say(f"\nConversion complete -> {output}")
    say("Next: validate by loading with mlx_lm.load or transformers")
    return {
        "output_path": str(output),
        "weights_path": str(output / "model.safetensors"),
        "tensor_count": tensor_count,
        "precision": precision,
        "num_hidden_layers": config["num_hidden_layers"],
        "hidden_size": config["hidden_size"],
        "vocab_size": config["vocab_size"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rwkv7", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference", default=None,
                   help="Optional safetensors from an FLA model with the same architecture for live ground-truth validation "
                        "(defaults to tools/fixtures/rwkv7_hf_template.json)")
    p.add_argument("--tokenizer-src", default=None,
                   help="Optional tokenizer source directory (defaults to assets/rwkv_world_tokenizer/)")
    p.add_argument("--precision", default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    convert(
        args.rwkv7, args.output, args.reference, args.tokenizer_src, args.precision,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
