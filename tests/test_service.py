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
    with pytest.raises(ValueError, match="nekoqa"):
        validate_training_request(request)
