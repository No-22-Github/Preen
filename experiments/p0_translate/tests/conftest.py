"""
P0 测试公共 fixtures。

- 模型加载(module scope): 901MB safetensors,全程只 load 一次。
- skip 机制: models/converted/ 被 gitignore,缺失时 skip 并提示转换命令。
- --slow 开关: 训练测试默认不跑,pytest --slow 显式开启。

路径: tests/ 在 experiments/p0_translate/ 下,仓库根是 parent.parent.parent.parent。
"""
from pathlib import Path

import pytest

# 仓库根 (tests/ → p0_translate/ → experiments/ → Preen/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = REPO_ROOT / "models" / "converted" / "rwkv7-g1d-0.4b"
STATE_PATH = Path(__file__).resolve().parent.parent / "final_state_v3.npz"
DATA_PATH = REPO_ROOT / "train_data" / "translate"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False,
        help="运行训练测试(slow, ~5min)",
    )


def pytest_collection_modifyitems(config, items):
    """未传 --slow 时,跳过标记为 slow 的测试。"""
    if config.getoption("--slow"):
        return
    skip_slow = pytest.mark.skip(reason="需要 --slow 开启 (训练测试,~5min)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="module")
def app():
    """加载模型 + tokenizer (module scope, 只 load 一次)。

    返回 (model, tokenizer)。模型缺失时 skip。
    """
    if not (MODEL_PATH / "model.safetensors").exists():
        pytest.skip(
            f"转换后的模型不存在: {MODEL_PATH}\n"
            f"先运行转换: python tools/convert_rwkv7_to_hf.py "
            f"--rwkv7 models/rwkv7-g1d-0.4b-20260210-ctx8192.pth "
            f"--output {MODEL_PATH} "
            f"--reference models/fla-hub-rwkv7-0.1B-g1/model.safetensors "
            f"--tokenizer-src models/fla-hub-rwkv7-0.1B-g1 --precision bf16"
        )
    from mlx_lm import load
    model, tokenizer = load(str(MODEL_PATH), tokenizer_config={"trust_remote_code": True})
    return model, tokenizer


@pytest.fixture(scope="module")
def state_file():
    """训练好的 state 文件路径。缺失时 skip。"""
    if not STATE_PATH.exists():
        pytest.skip(
            f"训练 state 不存在: {STATE_PATH}\n"
            f"先训练生成: cd experiments/p0_translate && "
            f"uv run python train_v3.py, 然后 cp checkpoints_v3/ep04.npz final_state_v3.npz"
        )
    return str(STATE_PATH)
