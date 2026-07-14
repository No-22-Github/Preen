"""int8 量化端到端测试(慢,需 --slow)。

覆盖 quantizer.quantize() 的完整 roundtrip:
  bf16 模型 → 量化 → int8 目录 → 重新 load → 生成文本

量化产物标准性:
  - config.json 有 quantization + quantization_config 字段
  - 重新 load 能生成文本(mlx_lm.load 自动识别)

模型缺失时自动 skip(和 conftest.py 的 app fixture 一致)。
"""
import json

import pytest

pytestmark = pytest.mark.slow

from tests.conftest import MODEL_PATH  # noqa: E402


def test_quantize_int8_roundtrip(tmp_path):
    """0.4B bf16 → int8 量化 → 目录结构 + config 字段 + 重新 load 生成。"""
    if not (MODEL_PATH / "model.safetensors").exists():
        pytest.skip(f"模型不存在: {MODEL_PATH}")

    from statetuner.quantizer import quantize

    out = tmp_path / "rwkv7-g1d-0.4b-int8"
    result = quantize(MODEL_PATH, out)

    # summary 关键字段
    assert result["bits"] == 8
    assert result["group_size"] == 64
    assert result["quantized_layers"] > 0

    # 产物目录结构
    assert (out / "model.safetensors").is_file()
    assert (out / "config.json").is_file()

    # config.json 有 quantization + quantization_config 字段(mlx-lm 标准)
    cfg = json.loads((out / "config.json").read_text())
    assert "quantization" in cfg
    assert cfg["quantization"]["bits"] == 8
    assert cfg["quantization"]["group_size"] == 64
    # HF 镜像字段(#957)
    assert cfg.get("quantization_config") == cfg["quantization"]

    # 重新 load 能识别量化模型并生成文本
    from statetuner.core import load_model

    model, tok = load_model(str(out), patch=False)
    inner = getattr(model, "model", model)
    n_quantized = sum(
        1
        for _, mod in inner.named_modules()
        if type(mod).__name__ == "QuantizedLinear"
    )
    assert n_quantized > 0, "加载 int8 模型后应含量化层"

    from statetuner.inference import GenerationConfig, InferenceEngine

    engine = InferenceEngine(model, tok)
    cfg_gen = GenerationConfig(max_tokens=20, temperature=0.0)
    res = engine.generate("User: 你好\n\nAssistant: ", config=cfg_gen)
    assert len(res.display_token_ids) > 0
    assert res.text  # 非空输出


def test_quantize_rejects_nonempty_out(tmp_path):
    """输出目录非空应报错(mlx_lm.save 约束)。"""
    if not (MODEL_PATH / "model.safetensors").exists():
        pytest.skip(f"模型不存在: {MODEL_PATH}")

    from statetuner.quantizer import quantize

    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("x")
    with pytest.raises(ValueError, match="非空"):
        quantize(MODEL_PATH, out)
