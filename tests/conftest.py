"""回归测试公共配置与 fixtures。

设计目标:一条命令跑完,10 分钟内知道有没有改坏。
  - 快测(默认,~1min):推理 golden + 导出 round-trip + 数据/事件单元测试。CI 友好。
  - 慢测(--slow,~4min):训练行为断言(冒烟/过拟合/收敛/翻译)。需显式开启。

模型缺失时自动 skip 并提示转换命令(checkpoints_v3 缺失同理)。
路径锚定仓库根: tests/ → Preen/
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "converted" / "rwkv7-g1d-0.4b"
DATA_PATH = REPO_ROOT / "train_data" / "translate"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# P0 归档里的标准 state(epoch4, lr=0.01, std≈0.17, 验收通过的翻译 state)
# 作为推理 golden 测试的基准 state。见 experiments/p0_translate/checkpoints_v3/ep04.npz
STATE_PATH = REPO_ROOT / "experiments" / "p0_translate" / "checkpoints_v3" / "ep04.npz"


def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="运行训练测试(slow, ~4min)",
    )


def pytest_collection_modifyitems(config, items):
    """未传 --slow 时,跳过标记为 slow 的测试。"""
    if config.getoption("--slow"):
        return
    skip_slow = pytest.mark.skip(reason="需要 --slow 开启 (训练测试, ~4min)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="module")
def app():
    """加载模型 + tokenizer(module scope,只 load 一次,kernel 路径)。

    返回 (model, tokenizer)。模型缺失时 skip 并提示转换命令。
    """
    if not (MODEL_PATH / "model.safetensors").exists():
        pytest.skip(
            f"转换后的模型不存在: {MODEL_PATH}\n"
            f"先运行: python tools/convert_rwkv7_to_hf.py "
            f"--rwkv7 models/rwkv7-g1d-0.4b-20260210-ctx8192.pth "
            f"--output {MODEL_PATH} "
            f"--reference models/fla-hub-rwkv7-0.1B-g1/model.safetensors "
            f"--tokenizer-src models/fla-hub-rwkv7-0.1B-g1 --precision bf16"
        )
    from statetuner.core import load_model

    model, tokenizer = load_model(str(MODEL_PATH), patch=False)
    return model, tokenizer


@pytest.fixture(scope="module")
def state_file():
    """训练好的 state 文件路径(P0 ep04)。缺失时 skip。"""
    if not STATE_PATH.exists():
        pytest.skip(f"state 不存在: {STATE_PATH}")
    return str(STATE_PATH)


@pytest.fixture
def tmp_pth(tmp_path):
    """导出测试用的临时 pth 路径。"""
    return tmp_path / "test.pth"
