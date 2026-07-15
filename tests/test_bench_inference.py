"""推理 benchmark 统计口径快测(不加载模型)。"""
import json

from tools.bench_inference import (
    detect_precision,
    median_and_range,
    stability_warnings,
)


def test_median_and_range_uses_p50_not_mean():
    # 离群的冷启动样本不应拖低主汇总值。
    assert median_and_range([33.1, 39.4, 40.6]) == (39.4, 33.1, 40.6)
    assert median_and_range([]) == (0.0, 0.0, 0.0)


def test_detect_precision_reads_model_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"torch_dtype": "bfloat16"})
    )
    assert detect_precision(tmp_path) == "bf16"

    (tmp_path / "config.json").write_text(
        json.dumps({"quantization": {"bits": 8, "group_size": 64}})
    )
    assert detect_precision(tmp_path) == "int8"


def test_detect_precision_runtime_override_is_explicit(tmp_path):
    assert detect_precision(tmp_path, "int8") == "int8-runtime"


def test_stability_warnings_rejects_noisy_level_and_cross_level_decode():
    rows = [
        {
            "prompt_tokens": 1081,
            "prefill_tps": 2141.8,
            "prefill_range": (1566.6, 2160.4),
            "decode_tps": 32.8,
            "decode_range": (28.5, 37.5),
        },
        {
            "prompt_tokens": 2114,
            "prefill_tps": 2374.9,
            "prefill_range": (2373.8, 2383.8),
            "decode_tps": 40.7,
            "decode_range": (40.5, 40.8),
        },
    ]
    warnings = stability_warnings(rows)
    assert any("1081 tok prefill" in warning for warning in warnings)
    assert any("1081 tok decode" in warning for warning in warnings)
    assert any("跨档 decode" in warning for warning in warnings)


def test_stability_warnings_accepts_tight_measurements():
    rows = [
        {
            "prompt_tokens": 2114,
            "prefill_tps": 2374.9,
            "prefill_range": (2373.8, 2383.8),
            "decode_tps": 40.7,
            "decode_range": (40.5, 40.8),
        },
        {
            "prompt_tokens": 4122,
            "prefill_tps": 2335.5,
            "prefill_range": (2333.1, 2338.2),
            "decode_tps": 40.6,
            "decode_range": (40.4, 40.7),
        },
    ]
    assert stability_warnings(rows) == []
