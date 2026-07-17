"""model_converter 流式转换回归测试(快,不加载模型)。

覆盖:
  - iter_pth / peek_pth_keys 与 read_pth 的等价性(逐字节 + 键名顺序)
  - convert 流式写单文件:产物始终为 model.safetensors(无分片),临时文件清理干净
  - convert 数值正确性:与 read_pth + 旧映射逻辑逐 tensor 完全一致
  - convert 的键名映射 / squeeze / transpose 与官方规则等价
  - _stream_save_safetensors 与官方 save_file 数值等价(逐 tensor tobytes 相等)

真实 0.4B 模型存在时,额外验证 mlx 可加载转换产物。
"""
import json
import zipfile

import numpy as np
import ml_dtypes
import pytest

from statetuner.pth_io import (
    iter_pth, peek_pth_keys, peek_pth_tensors, read_pth, write_pth,
)
from statetuner.model_converter import (
    _ConvertedTensorSpec, _stream_save_safetensors, convert, translate,
)


# ── 构造一个结构正确(但极小)的 mock RWKV-7 权重 dict ──
# 包含 _build_config 所需的键 + 多层 blocks,够跑完整 convert 主循环。
def _mock_rwkv_weights(num_layers=3, hidden=64, vocab=128):
    """构造结构正确的 mock RWKV-7 权重(小张量,用于触发完整转换路径)。"""
    w = {}
    w["emb.weight"] = np.random.randn(vocab, hidden).astype(np.float32)
    for i in range(num_layers):
        p = f"blocks.{i}."
        w[p + "ln0.weight"] = np.random.randn(hidden).astype(np.float32)
        w[p + "ln1.weight"] = np.random.randn(hidden).astype(np.float32)
        w[p + "ln2.weight"] = np.random.randn(hidden).astype(np.float32)
        for part in ["receptance", "key", "value", "output"]:
            w[p + f"att.{part}.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
        w[p + "att.ln_x.weight"] = np.random.randn(hidden).astype(np.float32)
        # lora [wvag][012]:0 是 bias(1,1,hidden),1/2 是 weight 矩阵
        for typ in ["w", "v", "a", "g"]:
            w[p + f"att.{typ}0"] = np.random.randn(1, 1, hidden).astype(np.float32)
            w[p + f"att.{typ}1"] = np.random.randn(16, hidden).astype(np.float32)
            w[p + f"att.{typ}2"] = np.random.randn(hidden, 16).astype(np.float32)
        # x_* 键:(1,1,hidden) 保留 3D
        for xn in ["x_w", "x_a", "x_k", "x_v", "x_r", "x_g"]:
            w[p + f"att.{xn}"] = np.random.randn(1, 1, hidden).astype(np.float32)
        w[p + "att.r_k"] = np.random.randn(hidden, 16).astype(np.float32)
        w[p + "att.k_a"] = np.random.randn(16).astype(np.float32)
        w[p + "att.k_k"] = np.random.randn(hidden).astype(np.float32)
        w[p + "ffn.key.weight"] = np.random.randn(hidden * 2, hidden).astype(np.float32)
        w[p + "ffn.value.weight"] = np.random.randn(hidden, hidden * 2).astype(np.float32)
        w[p + "ffn.receptance.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
    w["ln_out.weight"] = np.random.randn(hidden).astype(np.float32)
    w["head.weight"] = np.random.randn(vocab, hidden).astype(np.float32)
    return w


# ── iter_pth / peek_pth_keys 与 read_pth 等价性 ──

def test_iter_pth_equivalent_to_read_pth(tmp_path):
    """iter_pth 逐字节等价于 read_pth(覆盖 fp32 + bf16 + 多 shape)。"""
    p = tmp_path / "w.pth"
    tensors = {
        "emb.weight": np.random.randn(32, 64).astype(np.float32),
        "blocks.0.att.w1": np.random.randn(8, 64).astype(np.float32),
        "bias": np.arange(128, dtype=np.float32),
        "bf": (np.random.rand(16, 16) * 2 - 1).astype(ml_dtypes.bfloat16),
    }
    write_pth(tensors, p)

    ref = read_pth(p)
    streamed = dict(iter_pth(p))
    assert set(streamed) == set(ref), f"键集不同: {set(streamed) ^ set(ref)}"
    for k in ref:
        assert streamed[k].dtype == ref[k].dtype, f"{k} dtype 不一致"
        assert streamed[k].shape == ref[k].shape, f"{k} shape 不一致"
        assert streamed[k].tobytes() == ref[k].tobytes(), f"{k} 非逐字节等价"


def test_peek_pth_keys_only_metadata(tmp_path):
    """peek_pth_keys 只读 data.pkl,返回键名顺序 + 总数,不读 storage。"""
    p = tmp_path / "w.pth"
    tensors = {
        "emb.weight": np.zeros((4, 4), dtype=np.float32),
        "blocks.0.att.w1": np.zeros((2, 4), dtype=np.float32),
        "head.weight": np.zeros((4, 4), dtype=np.float32),
    }
    write_pth(tensors, p)
    keys, total = peek_pth_keys(p)
    # write_pth 用 OrderedDict,键顺序与插入一致
    assert keys == list(tensors.keys())
    assert total == len(tensors)


# ── convert 流式写单文件 ──

def test_convert_always_single_file(tmp_path):
    """convert 始终输出单文件 model.safetensors(无分片、无 index.json)。"""
    p = tmp_path / "src.pth"
    write_pth(_mock_rwkv_weights(), p)
    out = tmp_path / "out"

    result = convert(p, out, precision="bf16", log=lambda m: None)

    assert (out / "model.safetensors").is_file(), "应有单文件 model.safetensors"
    assert not (out / "model.safetensors.index.json").exists(), "不应有 index.json"
    assert not list(out.glob("model-*-of-*.safetensors")), "不应分片"
    assert result["tensor_count"] > 0
    assert (out / "config.json").is_file()


def test_convert_cleans_temp_files(tmp_path):
    """转换完成后同级 staging / backup 目录必须清理干净。"""
    p = tmp_path / "src.pth"
    write_pth(_mock_rwkv_weights(), p)
    out = tmp_path / "out"

    convert(p, out, precision="bf16", log=lambda m: None)

    leftover = list(tmp_path.glob(".out.preen-*-*"))
    assert not leftover, f"残留临时文件: {leftover}"


def test_convert_no_temp_left_on_error(tmp_path):
    """转换中途出错(如遇到无法映射的键)时,临时文件也应清理(_stream_save try/finally)。"""
    # 构造一个含无法映射键的 pth,让 translate 抛 AssertionError
    bad_tensors = {"misc.weight": np.zeros((4, 4), dtype=np.float32)}
    p = tmp_path / "bad.pth"
    write_pth(bad_tensors, p)
    out = tmp_path / "out"

    with pytest.raises((AssertionError, ValueError, KeyError)):
        convert(p, out, precision="bf16", log=lambda m: None)

    leftover = list(tmp_path.glob(".out.preen-*-*"))
    assert not leftover, f"出错后残留临时文件: {leftover}"


def test_convert_one_layer_uses_default_v_rank(tmp_path):
    """blocks.1.att.v1 是可选配置样本；单层模型应使用 v_low_rank_dim=32。"""
    source = tmp_path / "one-layer.pth"
    output = tmp_path / "out"
    write_pth(_mock_rwkv_weights(num_layers=1), source)

    convert(source, output, precision="bf16", log=None)

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["num_hidden_layers"] == 1
    assert config["v_low_rank_dim"] == 32


def test_convert_failure_keeps_existing_output_untouched(tmp_path):
    """预检失败不能破坏 overwrite 前的有效目录。"""
    source = tmp_path / "src.pth"
    output = tmp_path / "out"
    tokenizer = tmp_path / "missing-tokenizer"
    write_pth(_mock_rwkv_weights(), source)
    output.mkdir()
    tokenizer.mkdir()
    (output / "sentinel.txt").write_text("old model", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        convert(
            source, output, precision="bf16", tokenizer_src=tokenizer,
            overwrite=True, log=None,
        )

    assert [path.name for path in output.iterdir()] == ["sentinel.txt"]
    assert (output / "sentinel.txt").read_text(encoding="utf-8") == "old model"
    assert not list(tmp_path.glob(".out.preen-*-*"))


def test_convert_mid_write_failure_cleans_stage_and_keeps_old_output(
    tmp_path, monkeypatch,
):
    """staging 写入中途失败也必须回收临时目录并保留旧模型。"""
    source = tmp_path / "src.pth"
    output = tmp_path / "out"
    write_pth(_mock_rwkv_weights(), source)
    output.mkdir()
    (output / "sentinel.txt").write_text("old model", encoding="utf-8")

    def _fail_writer(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(
        "statetuner.model_converter._stream_save_safetensors", _fail_writer,
    )
    with pytest.raises(OSError, match="simulated disk failure"):
        convert(source, output, precision="bf16", overwrite=True, log=None)

    assert [path.name for path in output.iterdir()] == ["sentinel.txt"]
    assert not list(tmp_path.glob(".out.preen-*-*"))


def test_convert_overwrite_replaces_stale_shards(tmp_path):
    """overwrite 提交整个 staging，不保留旧 index、分片或其他模型文件。"""
    source = tmp_path / "src.pth"
    output = tmp_path / "out"
    write_pth(_mock_rwkv_weights(), source)
    output.mkdir()
    (output / "model.safetensors.index.json").write_text("{}")
    (output / "model-00001-of-00002.safetensors").write_text("stale")

    convert(source, output, precision="bf16", overwrite=True, log=None)

    assert (output / "model.safetensors").is_file()
    assert not (output / "model.safetensors.index.json").exists()
    assert not list(output.glob("model-*-of-*.safetensors"))


def test_convert_progress_does_not_regress_to_convert_after_write(tmp_path):
    source = tmp_path / "src.pth"
    output = tmp_path / "out"
    write_pth(_mock_rwkv_weights(), source)
    phases = []

    convert(
        source, output, precision="bf16", log=None,
        progress_callback=lambda phase, *_: phases.append(phase),
    )

    assert phases[0] == "read"
    assert phases[-1] == "write"
    assert "write" not in phases[:-1]


def test_top_level_rank_validation_happens_before_output(tmp_path):
    source = tmp_path / "bad-emb.pth"
    output = tmp_path / "out"
    weights = _mock_rwkv_weights()
    weights["emb.weight"] = np.zeros((1, 128, 64), dtype=np.float32)
    write_pth(weights, source)

    with pytest.raises(ValueError, match="Top-level rank validation failed"):
        convert(source, output, precision="bf16", log=None)

    assert not output.exists()


def test_compressed_pth_is_rejected_during_preflight(tmp_path):
    stored = tmp_path / "stored.pth"
    compressed = tmp_path / "compressed.pth"
    write_pth({"x": np.arange(8, dtype=np.float32)}, stored)
    with zipfile.ZipFile(stored) as source, zipfile.ZipFile(
        compressed, "w", zipfile.ZIP_DEFLATED,
    ) as target:
        for info in source.infolist():
            target.writestr(info.filename, source.read(info.filename))

    assert read_pth(compressed)["x"].tolist() == list(range(8))
    with pytest.raises(ValueError, match="ZIP_STORED"):
        peek_pth_tensors(compressed, require_stored=True)


# ── convert 数值正确性 ──

def test_convert_numerically_correct(tmp_path):
    """convert 产物的每个 tensor 数值与「read_pth + 旧映射逻辑」逐 tensor 完全一致。

    流式实现改变了文件内 tensor 的物理排列顺序(按 storage 分组),但每个 tensor 的
    dtype/shape/数值必须与按 pkl 顺序转换的结果完全相同。用 mlx 读回逐 tensor 比对。
    """
    src = _mock_rwkv_weights(num_layers=3)
    p = tmp_path / "src.pth"
    write_pth(src, p)
    out = tmp_path / "out"
    convert(p, out, precision="bf16", log=lambda m: None)

    # mlx 读回转换产物(原生支持 bf16)
    import mlx.core as mx
    produced = mx.load(str(out / "model.safetensors"))

    # 独立计算期望值:read_pth 全量 + 手动应用同样的映射规则
    raw = read_pth(p)
    nl = sum(1 for k in raw if k.endswith(".ffn.key.weight"))
    expected = {}
    for src_name, src_arr in raw.items():
        rel_name, transposed = translate(src_name, nl)
        if not rel_name:
            continue
        if "{N}" in rel_name:
            li = int(src_name.split(".")[1])
            fla_name = rel_name.replace("{N}", str(li))
        else:
            fla_name = rel_name
        weight = src_arr.T if transposed else src_arr
        is_x = "attn.x_" in fla_name
        if not is_x and weight.ndim == 3 and weight.shape[0] == 1 and weight.shape[1] == 1:
            weight = weight.squeeze()
        expected[fla_name] = mx.array(np.ascontiguousarray(
            weight.astype(ml_dtypes.bfloat16)))

    assert set(produced) == set(expected), \
        f"键集不同: 产物多={set(produced)-set(expected)}, 期望多={set(expected)-set(produced)}"
    for k in expected:
        assert produced[k].shape == expected[k].shape, f"{k} shape 不一致"
        assert mx.array_equal(produced[k].astype(mx.float32),
                              expected[k].astype(mx.float32)), f"{k} 数值不一致"


def test_stream_save_equivalent_to_official(tmp_path):
    """_stream_save_safetensors 与官方 save_file **逐字节一致**(sha256 相同)。

    流式实现按 name 字典序排列 + 8 字节对齐 padding,产出与官方 save_file 完全相同的
    二进制(header key 顺序、data_offsets、padding、数据段排列全部一致)。

    注意:真实 RWKV 转换产物是单一精度(全 BF16 或全 fp16/fp32),此时官方 save_file
    的 dtype 分组排序退化为纯名字字典序,与本流式实现完全对齐。混合 dtype 场景下官方
    按 Rust 端 Dtype enum 排序(实现细节),本测试用单一精度覆盖真实使用场景。
    """
    import hashlib
    from safetensors.numpy import save_file

    # 单一精度(BF16,对齐真实 RWKV 转换产物)+ 故意用非字典序插入,验证流式版会重排
    tensors = {
        "zzz_last": (np.random.rand(32, 64) * 2 - 1).astype(ml_dtypes.bfloat16),
        "aaa_first": (np.random.rand(128) * 2 - 1).astype(ml_dtypes.bfloat16),
        "mid": (np.random.rand(16, 16) * 2 - 1).astype(ml_dtypes.bfloat16),
    }
    stream_path = str(tmp_path / "stream.safetensors")
    official_path = str(tmp_path / "official.safetensors")

    ordered = sorted(tensors.items())
    specs = [
        _ConvertedTensorSpec(
            source_name=name, target_name=name,
            source_shape=tuple(value.shape), target_shape=tuple(value.shape),
            transposed=False, dtype=np.dtype(value.dtype),
        )
        for name, value in ordered
    ]
    _stream_save_safetensors(iter(ordered), stream_path, specs)
    save_file(tensors, official_path, metadata={"format": "pt"})

    s_hash = hashlib.sha256(open(stream_path, "rb").read()).hexdigest()
    o_hash = hashlib.sha256(open(official_path, "rb").read()).hexdigest()
    assert s_hash == o_hash, \
        f"流式产物与官方 save_file 不逐字节一致: stream={s_hash[:16]} official={o_hash[:16]}"


# ── 键名映射 / squeeze 语义不变 ──

def test_translate_semantics():
    """translate() 键名映射 + transpose 标志与官方规则一致(回归保护)。

    注意:translate 返回的层内键含 {N} 占位符(真实层号在 convert 主循环里替换)。
    """
    nl = 24
    assert translate("emb.weight", nl) == ("model.embeddings.weight", False)
    assert translate("head.weight", nl) == ("lm_head.weight", False)
    # 层内 lora:0→2.bias(不转置),1→0.weight(转置),2→2.weight(转置)
    r, t = translate("blocks.0.att.w0", nl)
    assert r == "model.layers.{N}.attn.w_lora.lora.2.bias" and t is False
    r, t = translate("blocks.3.att.w1", nl)
    assert r == "model.layers.{N}.attn.w_lora.lora.0.weight" and t is True
    r, t = translate("blocks.3.att.w2", nl)
    assert r == "model.layers.{N}.attn.w_lora.lora.2.weight" and t is True
    # blocks.0.att.v* 丢弃
    assert translate("blocks.0.att.v0", nl) == ("", False)
    assert translate("blocks.0.att.v1", nl) == ("", False)
    # x_* 保留位置名
    r, t = translate("blocks.1.att.x_w", nl)
    assert r == "model.layers.{N}.attn.x_w" and t is False


# ── 真实 0.4B 模型(若存在):mlx 可加载 + config 正确 ──

REAL_PTH = "models/rwkv7-g1d-0.4b-20260210-ctx8192.pth"


@pytest.mark.skipif(
    not __import__("os").path.exists(REAL_PTH),
    reason="真实 0.4B pth 不存在,跳过产物对比",
)
def test_real_04b_product(tmp_path):
    """真实 0.4B 转换:单文件产物 + mlx 可加载 + config 推断正确。"""
    out = tmp_path / "out"
    result = convert(REAL_PTH, out, precision="bf16", log=lambda m: None)
    assert (out / "model.safetensors").is_file()
    assert not (out / "model.safetensors.index.json").exists()
    # 0.4B 结构断言
    assert result["num_hidden_layers"] == 24
    assert result["hidden_size"] == 1024
    assert result["vocab_size"] == 65536

    # mlx 能加载(下游兼容性)
    import mlx.core as mx
    w = mx.load(str(out / "model.safetensors"))
    assert len(w) > 0
    assert w["model.embeddings.weight"].shape == (65536, 1024)
