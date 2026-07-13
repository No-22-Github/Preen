"""兼容入口：正式实现位于 statetuner.model_converter。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from statetuner.model_converter import *  # noqa: F401,F403 - 保留历史脚本导入 API
from statetuner.model_converter import main


if __name__ == "__main__":
    main()
