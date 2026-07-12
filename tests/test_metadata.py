import json

from statetuner.metadata import file_sha256, write_state_metadata


def test_write_state_metadata(tmp_path):
    data = tmp_path / "data.json"
    data.write_text("[]", encoding="utf-8")
    state = tmp_path / "state.npz"
    state.write_bytes(b"state")
    meta = write_state_metadata(
        state,
        model_path=tmp_path / "model",
        data_path=data,
        template="qa",
        config={"lr": 0.01},
        data_stats={"valid": 2},
        result={"final_loss": 1.0},
    )
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert payload["precision"]["weights"] == "bf16"
    assert payload["precision"]["train_state"] == "fp32"
    assert payload["data_sha256"] == file_sha256(data)
    assert payload["artifacts"]["state_npz"] == str(state)
