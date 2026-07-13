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
        "convert-model", "dataset-preview", "dataset-preview-page", "import",
    ):
        assert command in result.stdout


def test_convert_model_emits_tool_events(tmp_path, monkeypatch):
    source = tmp_path / "model.pth"
    source.write_bytes(b"fixture")
    out = tmp_path / "converted"

    def fake_convert(source_path, output, **kwargs):
        kwargs["progress_callback"]("convert", "转换张量", 1, 2)
        output.mkdir()
        return {"output_path": str(output), "tensor_count": 2, "precision": "bf16"}

    monkeypatch.setattr("statetuner.model_converter.convert", fake_convert)
    result = runner.invoke(app, [
        "convert-model", "--rwkv7", str(source), "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [event["type"] for event in events] == ["started", "progress", "completed"]
    assert events[-1]["tool"] == "model_conversion"
    assert events[-1]["result"]["tensor_count"] == 2


def test_import_events_support_manual_mapping(tmp_path):
    source = tmp_path / "custom.jsonl"
    source.write_text('{"ask":"你好","reply":"喵"}\n', encoding="utf-8")
    out = tmp_path / "standard.jsonl"
    result = runner.invoke(app, [
        "import", "--data", str(source), "--out", str(out),
        "--prompt-key", "ask", "--response-key", "reply", "--events",
    ])
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [event["type"] for event in events] == ["started", "completed"]
    assert events[-1]["result"]["record_count"] == 1
    assert out.exists()


def test_dataset_preview_emits_render_and_inspection(tmp_path, monkeypatch):
    class Tokenizer:
        @staticmethod
        def encode(text):
            return [ord(char) for char in text]

    model = tmp_path / "model"
    model.mkdir()
    source = tmp_path / "custom.jsonl"
    source.write_text('{"ask":"你好","reply":"喵"}\n', encoding="utf-8")
    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", lambda *args, **kwargs: Tokenizer())
    result = runner.invoke(app, [
        "dataset-preview", "--model", str(model), "--data", str(source),
        "--prompt-key", "ask", "--response-key", "reply", "--ctx-len", "128",
    ])
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    payload = events[-1]["result"]
    assert events[-1]["type"] == "completed"
    assert payload["inspection"]["valid"] == 1
    assert payload["preview"][0]["full_text"] == (
        payload["preview"][0]["prefix_text"] + payload["preview"][0]["target_text"]
    )


def test_dataset_preview_cache_pages_without_returning_all_rows(tmp_path, monkeypatch):
    class Tokenizer:
        @staticmethod
        def encode(text):
            return [ord(char) for char in text]

    model = tmp_path / "model"
    model.mkdir()
    source = tmp_path / "many.jsonl"
    source.write_text("".join(
        json.dumps({"q": f"问题 {i}", "a": f"回答 {i}"}, ensure_ascii=False) + "\n"
        for i in range(45)
    ), encoding="utf-8")
    cache = tmp_path / "preview-cache.jsonl"
    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", lambda *args, **kwargs: Tokenizer())

    initial = runner.invoke(app, [
        "dataset-preview", "--model", str(model), "--data", str(source),
        "--cache-out", str(cache), "--page-size", "20",
    ])
    assert initial.exit_code == 0, initial.output
    initial_events = [json.loads(line) for line in initial.stdout.splitlines() if line.strip()]
    payload = initial_events[-1]["result"]
    assert len(payload["preview"]) == 20
    assert payload["pagination"] == {
        "cache_path": str(cache), "total": 45, "page_size": 20, "page_count": 3,
    }

    page = runner.invoke(app, [
        "dataset-preview-page", "--cache", str(cache), "--page", "3",
    ])
    assert page.exit_code == 0, page.output
    page_events = [json.loads(line) for line in page.stdout.splitlines() if line.strip()]
    page_payload = page_events[-1]["result"]
    assert page_payload["page"] == 3
    assert page_payload["total"] == 45
    assert len(page_payload["preview"]) == 5
    assert page_payload["preview"][0]["prompt_text"] == "问题 40"


def test_data_info_routes_import_sidecar_to_standard_loader(tmp_path, monkeypatch):
    class Tokenizer:
        @staticmethod
        def encode(text):
            return [ord(char) for char in text]

    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "standard.jsonl"
    data.write_text('{"prompt":"你好","response":"喵"}\n', encoding="utf-8")
    sidecar = data.with_name(data.stem + data.suffix + ".import.json")
    sidecar.write_text('{"result":{"template":"qa"}}', encoding="utf-8")
    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", lambda *args, **kwargs: Tokenizer())
    result = runner.invoke(app, [
        "data-info", "--model", str(model), "--data", str(data), "--json",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["valid"] == 1


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


def test_export_deep_requires_model(tmp_path):
    """T6:export --deep 必须同时提供 --model(端到端 mount 校验需加载模型)。"""
    # 造一个合法的 npz(3 层,满足 load_npz_as_numpy)
    state = tmp_path / "state.npz"
    arrays = {f"layer_{i}": np.zeros((4, 8, 8), dtype=np.float32) for i in range(3)}
    np.savez(state, **arrays)
    out = tmp_path / "out.pth"
    result = runner.invoke(
        app,
        ["export", "--state", str(state), "--out", str(out), "--deep"],
    )
    # --deep 缺 --model → 拒绝(exit 2),不应进到模型加载
    assert result.exit_code == 2
    assert "--deep" in result.output and "--model" in result.output


def test_export_deep_rejects_missing_model_dir(tmp_path):
    """T6:export --deep --model <不存在> 在模型加载前拒绝。"""
    state = tmp_path / "state.npz"
    arrays = {f"layer_{i}": np.zeros((4, 8, 8), dtype=np.float32) for i in range(3)}
    np.savez(state, **arrays)
    out = tmp_path / "out.pth"
    result = runner.invoke(
        app,
        [
            "export", "--state", str(state), "--out", str(out),
            "--deep", "--model", str(tmp_path / "no-such-model"),
        ],
    )
    assert result.exit_code == 2
    assert "模型目录不存在" in result.output


def _make_train_doubles(monkeypatch, received):
    """搭 train 测试用的 MLX + service 双桩,不加载真实模型。

    received 是测试侧传入的 dict,用于收集 set_cache_limit 的入参。
    返回 run_training 的 noop stub(已 patch 到 service 模块)。
    """

    def _fake_set_cache_limit(n):
        received["bytes"] = n

    def _noop(request, emitter, *, status=None):
        from statetuner.service import TrainingJobResult

        return TrainingJobResult(
            state_path=request.out,
            metadata_path=request.out,
            pth_path=None,
            epochs_run=0,
            final_loss=0.0,
            final_state_std=0.0,
            elapsed=0.0,
        )

    monkeypatch.setattr("mlx.core.set_cache_limit", _fake_set_cache_limit)
    monkeypatch.setattr("statetuner.service.run_training", _noop)
    return _noop


def _write_train_fixture(tmp_path):
    """搭 train 命令所需的最小 model 目录 + 单条 data.json。"""
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "data.json"
    data.write_text(
        json.dumps([{"instruction": "q", "output": "a"}], ensure_ascii=False),
        encoding="utf-8",
    )
    return model, data


def test_train_applies_explicit_cache_limit(tmp_path, monkeypatch):
    """--cache-limit-gb 4 → set_cache_limit(int(4e9))(显式数字用法)。"""
    model, data = _write_train_fixture(tmp_path)
    received = {}
    _make_train_doubles(monkeypatch, received)

    result = runner.invoke(
        app,
        [
            "train", "--model", str(model), "--data", str(data),
            "--out", str(tmp_path / "state.npz"),
            "--cache-limit-gb", "4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert received.get("bytes") == int(4 * 1e9)


def test_train_cache_limit_default_is_auto(tmp_path, monkeypatch):
    """不传 --cache-limit-gb 时,默认 auto = 物理内存 × 25%。"""
    model, data = _write_train_fixture(tmp_path)
    received = {}
    _make_train_doubles(monkeypatch, received)

    # 假 16G 机器:memory_size = 16e9 bytes → auto 应得 int(4e9)。
    monkeypatch.setattr(
        "mlx.core.device_info", lambda: {"memory_size": int(16 * 1e9)}
    )

    result = runner.invoke(
        app,
        [
            "train", "--model", str(model), "--data", str(data),
            "--out", str(tmp_path / "state.npz"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert received.get("bytes") == int(16 * 1e9 * 0.25)


def test_train_cache_limit_auto_uses_quarter_of_memory(tmp_path, monkeypatch):
    """显式 --cache-limit-gb auto → 物理内存 × 25%(与默认同路径,独立覆盖)。"""
    model, data = _write_train_fixture(tmp_path)
    received = {}
    _make_train_doubles(monkeypatch, received)

    monkeypatch.setattr(
        "mlx.core.device_info", lambda: {"memory_size": int(32 * 1e9)}
    )

    result = runner.invoke(
        app,
        [
            "train", "--model", str(model), "--data", str(data),
            "--out", str(tmp_path / "state.npz"),
            "--cache-limit-gb", "auto",
        ],
    )
    assert result.exit_code == 0, result.output
    assert received.get("bytes") == int(32 * 1e9 * 0.25)


def test_train_cache_limit_rejects_bad_input(tmp_path):
    """--cache-limit-gb abc → exit 2 + 错误文案(走 _bad_input)。"""
    model, data = _write_train_fixture(tmp_path)
    result = runner.invoke(
        app,
        [
            "train", "--model", str(model), "--data", str(data),
            "--out", str(tmp_path / "state.npz"),
            "--cache-limit-gb", "abc",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "--cache-limit-gb" in result.output


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
