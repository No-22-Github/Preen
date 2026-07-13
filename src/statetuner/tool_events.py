"""离线工具任务共用的 JSON Lines 事件协议。"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, IO, Optional


@dataclass(frozen=True)
class ToolEvent:
    """工具任务事件；type 为 started/progress/warning/completed/failed/cancelled。"""

    type: str
    tool: str
    timestamp: float = field(default_factory=time.time)
    phase: Optional[str] = None
    message: Optional[str] = None
    current: Optional[int] = None
    total: Optional[int] = None
    progress: Optional[float] = None
    path: Optional[str] = None
    result: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class ToolEventEmitter:
    """逐行输出并立即 flush，Swift 可实时消费且能发送 SIGINT 取消。"""

    def __init__(self, stream: Optional[IO[str]] = None):
        self.stream = stream if stream is not None else sys.stdout

    def emit(self, event: ToolEvent) -> None:
        self.stream.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self.stream.flush()


def started(tool: str, message: str) -> ToolEvent:
    return ToolEvent(type="started", tool=tool, message=message, progress=0.0)


def progress(
    tool: str,
    phase: str,
    message: str,
    *,
    current: Optional[int] = None,
    total: Optional[int] = None,
    fraction: Optional[float] = None,
) -> ToolEvent:
    if fraction is None and current is not None and total:
        fraction = max(0.0, min(1.0, current / total))
    elif fraction is not None:
        fraction = max(0.0, min(1.0, fraction))
    return ToolEvent(
        type="progress", tool=tool, phase=phase, message=message,
        current=current, total=total, progress=fraction,
    )


def warning(tool: str, message: str) -> ToolEvent:
    return ToolEvent(type="warning", tool=tool, message=message)


def completed(tool: str, result: dict[str, Any], *, path: Optional[str] = None) -> ToolEvent:
    return ToolEvent(
        type="completed", tool=tool, message="完成", progress=1.0,
        path=path, result=result,
    )


def failed(tool: str, message: str) -> ToolEvent:
    return ToolEvent(type="failed", tool=tool, message=message)


def cancelled(tool: str, message: str = "用户取消") -> ToolEvent:
    return ToolEvent(type="cancelled", tool=tool, message=message)
