import json

from statetuner.metadata import file_sha256, write_state_metadata


def test_write_state_metadata(tmp_path):
    data = tmp_path / "data.json"
    data.write_text("[]", encoding="utf-8")
    state = tmp_path / "state.npz"
    state.write_bytes(b"state")
    model = tmp_path / "model"
    meta = write_state_metadata(
        state,
        model_path=model,
        data_path=data,
        template="qa",
        config={"lr": 0.01},
        data_stats={"valid": 2},
        result={"final_loss": 1.0},
    )
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["format_version"] == 2
    assert payload["model_name"] == "model"
    assert payload["model_path"] == str(model)
    assert payload["state_format"] == "npz"
    assert payload["state_dtype"] == "float32"
    assert payload["precision"]["weights"] == "bf16"
    assert payload["precision"]["train_state"] == "fp32"
    assert payload["data_sha256"] == file_sha256(data)
    assert payload["artifacts"]["state_npz"] == str(state)
