"""importer 端到端慢测(产物喂 train 冒烟,需 --slow)。

覆盖 Spec §4.5 验收 d:导入产物直接喂给 train 命令可跑通(端到端冒烟)。

用 alpaca fixture(5 条)+ instruction 模板 + 1 epoch + 小 ctx,
主要验证 load_standard_jsonl 能读 importer 产物并跑通前向/反向。
不做收敛断言(数据太少),只验证不崩 + 产物可读。
"""
from __future__ import annotations

import json
import subprocess
import os
from pathlib import Path

import pytest

from statetuner.importer import import_dataset

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "import"


def test_imported_jsonl_feeds_train(tmp_path):
    """§4.5.d:importer 产物(instruction 模板)喂 train 命令跑通。

    链路:alpaca fixture → import(instruction)→ load_standard_jsonl → train 1 epoch。
    验证:不崩 + state.npz 产物可被 inspection 读回。
    """
    if not (REPO_ROOT / "models" / "converted" / "rwkv7-g1d-0.4b" / "model.safetensors").exists():
        pytest.skip("模型不存在,跳过 train 冒烟")

    # 1. 导入
    src = FIXTURES / "alpaca_sample.jsonl"
    out_jsonl = tmp_path / "imported.jsonl"
    artifact, result = import_dataset(src, out_jsonl)
    assert result.template == "instruction"
    assert artifact.record_count == 5

    # 2. 喂 train(1 epoch,小 ctx,关早停)
    state_out = tmp_path / "state.npz"
    events_file = tmp_path / "events.jsonl"
    cmd = [
        PYTHON, "-m", "statetuner.cli", "train",
        "--model", str(REPO_ROOT / "models" / "converted" / "rwkv7-g1d-0.4b"),
        "--data", str(out_jsonl),
        "--template", "instruction",
        "--out", str(state_out),
        "--events-file", str(events_file),
        "--lr", "0.01", "--epochs", "1", "--ctx-len", "128",
        "--no-early-stop", "--seed", "42",
        "--warmup", "2",
    ]
    train_result = subprocess.run(
        cmd, env=ENV, capture_output=True, text=True, timeout=300,
    )
    assert train_result.returncode == 0, (
        f"train 失败:\nstdout: {train_result.stdout}\nstderr: {train_result.stderr}"
    )

    # 3. 产物可读
    assert state_out.exists(), "state.npz 未生成"
    assert events_file.exists(), "events.jsonl 未生成"

    # 4. events 里有 completed 终结
    events = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert "completed" in types, f"训练未正常完成, events types: {types}"

    # 5. state 产物可被 inspection 读回
    inspect_result = subprocess.run(
        [PYTHON, "-m", "statetuner.cli", "state-info",
         "--state", str(state_out), "--json"],
        env=ENV, capture_output=True, text=True, timeout=30,
    )
    assert inspect_result.returncode == 0, f"state-info 失败: {inspect_result.stderr}"
    state_info = json.loads(inspect_result.stdout)
    assert state_info["layers"] == 24  # 0.4B 模型 24 层
