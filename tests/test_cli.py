"""CLI 入口快测：不加载真实模型。"""
import json

import numpy as np
from typer.testing import CliRunner

from statetuner.cli import app


runner = CliRunner()


def test_root_help_lists_product_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "train", "eval", "export", "preview", "chat",
        "doctor", "data-info", "state-info",
    ):
        assert command in result.stdout


def test_preview_ab_requires_state_before_model_load(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    result = runner.invoke(
        app,
        ["preview", "--model", str(model), "--prompt", "你好", "--ab"],
    )
    assert result.exit_code != 0
    assert "--state" in result.output


def test_preview_stream_rejects_json_before_model_load(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    result = runner.invoke(
        app,
        [
            "preview", "--model", str(model), "--prompt", "你好",
            "--stream", "--json",
        ],
    )
    assert result.exit_code != 0
    assert "不能与" in result.output


def test_train_rejects_invalid_parameter_before_model_load(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "data.json"
    data.write_text("[]", encoding="utf-8")
    result = runner.invoke(
        app,
        ["train", "--model", str(model), "--data", str(data), "--lr", "0"],
    )
    assert result.exit_code == 2
    assert "--lr 必须 > 0" in result.output


def test_train_runtime_failure_exits_1(tmp_path, monkeypatch):
    """run_training 抛常规异常时,CLI 记 failed 事件并 exit 1(不加载真实模型)。"""
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps([{"instruction": "q", "output": "a"}], ensure_ascii=False),
        encoding="utf-8",
    )

    def _boom(request, emitter, *, status=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("statetuner.service.run_training", _boom)
    result = runner.invoke(
        app,
        [
            "train", "--model", str(model), "--data", str(data),
            "--out", str(tmp_path / "state.npz"),
        ],
    )
    assert result.exit_code == 1
    assert "boom" in result.output


def test_train_interrupt_exits_130(tmp_path, monkeypatch):
    """用户中断(Ctrl-C)时,CLI 记 cancelled 事件并 exit 130。"""
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps([{"instruction": "q", "output": "a"}], ensure_ascii=False),
        encoding="utf-8",
    )

    def _interrupt(request, emitter, *, status=None):
        raise KeyboardInterrupt

    monkeypatch.setattr("statetuner.service.run_training", _interrupt)
    result = runner.invoke(
        app,
        [
            "train", "--model", str(model), "--data", str(data),
            "--out", str(tmp_path / "state.npz"),
        ],
    )
    assert result.exit_code == 130


def test_train_rejects_pth_out_without_export(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "data.json"
    data.write_text("[]", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "train", "--model", str(model), "--data", str(data),
            "--pth-out", str(tmp_path / "state.pth"),
        ],
    )
    assert result.exit_code == 2
    assert "--export-pth" in result.output


def test_state_info_json(tmp_path):
    path = tmp_path / "state.npz"
    np.savez(
        path,
        layer_0=np.zeros((2, 64, 64), dtype=np.float32),
        layer_1=np.ones((2, 64, 64), dtype=np.float32),
    )
    result = runner.invoke(app, ["state-info", "--state", str(path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["layers"] == 2
    assert payload["continuous_layers"] is True
    assert payload["rwkv7_compatible"] is True


def test_state_info_rejects_unknown_format(tmp_path):
    path = tmp_path / "state.txt"
    path.write_text("bad", encoding="utf-8")
    result = runner.invoke(app, ["state-info", "--state", str(path)])
    assert result.exit_code == 2
    assert "只支持 .npz / .pth" in result.output
