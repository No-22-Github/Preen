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
