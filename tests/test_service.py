from types import SimpleNamespace

import pytest

from statetuner.service import TrainingRequest, validate_training_request


def _request(tmp_path, **overrides):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    data = tmp_path / "data.json"
    data.write_text("[]", encoding="utf-8")
    cfg = SimpleNamespace(
        lr=0.01,
        lr_floor=1e-4,
        warmup=10,
        ctx_len=512,
        epochs=3,
        grad_clip=1.0,
    )
    values = {
        "model": model,
        "data": data,
        "out": tmp_path / "state.npz",
        "train_config": cfg,
    }
    values.update(overrides)
    return TrainingRequest(**values)


def test_validate_training_request_accepts_normal_config(tmp_path):
    validate_training_request(_request(tmp_path))


def test_validate_training_request_rejects_pth_out_without_export(tmp_path):
    request = _request(tmp_path, pth_out=tmp_path / "state.pth")
    with pytest.raises(ValueError, match="export_pth"):
        validate_training_request(request)


def test_validate_training_request_rejects_unknown_template(tmp_path):
    request = _request(tmp_path, template="raw")
    with pytest.raises(ValueError, match="qa / instruction"):
        validate_training_request(request)


def test_validate_rejects_instruction_with_legacy_data(tmp_path):
    """S1:遗留数据(无 .import.json sidecar)+ instruction 模板应在校验阶段被拒。

    旧实现放行 instruction 却走 load_qa_dataset(永远 QA 编码),metadata 写
    "template":"instruction" 但实际按 QA 训练 → 静默走错。现在 validate 拦下。
    """
    request = _request(tmp_path, template="instruction")
    with pytest.raises(ValueError, match="不支持 instruction 模板"):
        validate_training_request(request)


def test_validate_accepts_instruction_with_import_sidecar(tmp_path):
    """S1:有 importer sidecar 的数据 + instruction 走标准 loader,应放行。"""
    request = _request(tmp_path, template="instruction")
    # 造 sidecar(importer 产物标记)
    sidecar = request.data.with_name(
        request.data.stem + request.data.suffix + ".import.json"
    )
    sidecar.write_text("{}", encoding="utf-8")
    # 不应抛(其余校验项已满足)
    validate_training_request(request)


def test_validate_rejects_quantized_model(tmp_path):
    """量化模型(config 含 quantization 字段)不能训练,应在校验阶段被拦下。

    训练精度契约:权重 bf16 + state fp32(docs/decision-precision.md)。
    int8 量化模型是推理专用产物。
    """
    import json

    request = _request(tmp_path)
    (request.model / "config.json").write_text(
        json.dumps({"quantization": {"bits": 8, "group_size": 64}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="量化模型"):
        validate_training_request(request)


def test_validate_rejects_quantized_model_via_config_alias(tmp_path):
    """quantization_config(HF 镜像字段)同样应触发训练拦截。"""
    import json

    request = _request(tmp_path)
    (request.model / "config.json").write_text(
        json.dumps({"quantization_config": {"bits": 8}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="量化模型"):
        validate_training_request(request)


def test_validate_accepts_unquantized_model(tmp_path):
    """无 quantization 字段的普通 bf16 模型正常放行(回归保护)。"""
    import json

    request = _request(tmp_path)
    (request.model / "config.json").write_text(
        json.dumps({"model_type": "rwkv7", "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )
    validate_training_request(request)
