"""结构化训练事件。

每个事件是一个 JSON-serializable dict,通过 EventEmitter 发出。
目的:
  ① CLI: 把事件流以 JSON lines 输出到 stdout(人类可读 + 可 grep)
  ② sidecar IPC(Phase 3): 同一事件流直接推给 SwiftUI 进度面板/loss 曲线

事件类型:
  start          训练开始,带 config 快照
  epoch_start    epoch 开始
  step           一个训练步(按 log_every 抽样)
  epoch_end      epoch 结束,带平均 loss / state_std / lr
  std_warning    兼容旧实验的可选阈值事件(产品默认不启用)
  checkpoint     存了 checkpoint
  early_stop     held-out 早停触发
  final          Trainer 计算结束(产物可能尚未落盘)
  completed      CLI job 必需产物均已落盘
  failed         CLI job 失败
  cancelled      用户取消

字段全部是原生类型(str/int/float/bool/list/dict),json.dumps 直接序列化。
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Callable, List, Optional, Union

PathLike = Union[str, Path, None]


@dataclass
class Event:
    """单个训练事件。type 决定携带哪些可选字段。"""

    type: str
    timestamp: float = field(default_factory=time.time)
    # 通用可选字段(按 type 填充,None 则不输出)
    epoch: Optional[int] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    lr: Optional[float] = None
    state_std: Optional[float] = None
    held_out_loss: Optional[float] = None
    best: Optional[float] = None
    patience_left: Optional[int] = None
    message: Optional[str] = None
    path: Optional[str] = None
    config: Optional[dict] = None
    elapsed: Optional[float] = None

    def to_dict(self) -> dict:
        """转 dict,丢弃值为 None 的字段(type/timestamp 总保留)。"""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class EventEmitter:
    """事件分发:把每个 Event 以 JSON line 发给所有 sink。

    sink 可以是:
      - 文件路径(追加写 JSON lines,train.py 的 --events-file)
      - 已打开的文本流(sys.stdout 默认)
      - callable(Event)(程序内订阅,供测试断言用)

    默认 sink 是 sys.stdout(CLI 场景:用户/管道消费 JSON lines)。
    """

    def __init__(
        self,
        *,
        file: PathLike = None,
        stream: Optional[IO] = None,
        callback: Optional[Callable[[Event], None]] = None,
        quiet: bool = False,
    ):
        self._owns_file = False
        self._file: Optional[IO] = None
        if file is not None:
            # 覆盖写(非追加):每次训练是一个独立事件流,重跑应清空旧事件,
            # 否则不同训练的 epoch/step 会混在同一文件里造成误读。
            file_path = Path(file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(file_path, "w", encoding="utf-8")
            self._owns_file = True
        self._stream = None if quiet else (stream if stream is not None else sys.stdout)
        self._callbacks: List[Callable[[Event], None]] = (
            [callback] if callback else []
        )
        # 收集所有已发事件(测试断言用;生产环境不依赖)
        self.events: List[dict] = []

    def subscribe(self, cb: Callable[[Event], None]) -> None:
        self._callbacks.append(cb)

    def emit(self, event: Event) -> None:
        line = event.to_json()
        self.events.append(event.to_dict())
        if self._stream is not None:
            self._stream.write(line + "\n")
            self._stream.flush()
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()
        for cb in self._callbacks:
            cb(event)

    def close(self) -> None:
        if self._owns_file and self._file is not None:
            self._file.close()
            self._file = None
            self._owns_file = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ── 便捷工厂(让 train.py 调用更清晰)─────────────────────────
def start(config: dict) -> Event:
    return Event(type="start", config=config)


def epoch_start(epoch: int) -> Event:
    return Event(type="epoch_start", epoch=epoch)


def step(
    step: int,
    total_steps: int,
    loss: float,
    lr: float,
    epoch: Optional[int] = None,
) -> Event:
    return Event(
        type="step",
        step=step,
        total_steps=total_steps,
        loss=loss,
        lr=lr,
        epoch=epoch,
    )


def epoch_end(
    epoch: int,
    loss: float,
    state_std: float,
    lr: float,
    held_out_loss: Optional[float] = None,
    best: Optional[float] = None,
    patience_left: Optional[int] = None,
) -> Event:
    return Event(
        type="epoch_end",
        epoch=epoch,
        loss=loss,
        state_std=state_std,
        lr=lr,
        held_out_loss=held_out_loss,
        best=best,
        patience_left=patience_left,
    )


def std_warning(epoch: int, state_std: float, threshold: float) -> Event:
    """state std 超过给定阈值的事件。

    产品 CLI 默认不启用(max_state_std=None);只有显式传阈值时才会触发。
    文案对齐现状:state std 健康区间尚未标定,超阈值只记录不解释、不中断
    (旧文案写"可能数值爆炸"是给已废弃的 1.0 阈值背书,与 core.state_std 的
    "阈值尚未标定"注释矛盾 —— 同一事实在仓库里曾有两个版本)。
    """
    return Event(
        type="std_warning",
        epoch=epoch,
        state_std=state_std,
        message=f"state std {state_std:.3f} > {threshold} (recorded only; healthy range is not calibrated)",
    )


def checkpoint(epoch: int, path: str) -> Event:
    return Event(type="checkpoint", epoch=epoch, path=path)


def early_stop(epoch: int, best: float, held_out_loss: float) -> Event:
    return Event(
        type="early_stop",
        epoch=epoch,
        best=best,
        held_out_loss=held_out_loss,
        message="Held-out loss did not improve for the configured patience; stopping early",
    )


def final(path: str, elapsed: float, best: Optional[float] = None) -> Event:
    return Event(type="final", path=path, elapsed=elapsed, best=best)


def completed(path: str, elapsed: float, message: Optional[str] = None) -> Event:
    """整个 CLI job 的必需产物均已落盘。"""
    return Event(type="completed", path=path, elapsed=elapsed, message=message)


def failed(message: str, path: Optional[str] = None) -> Event:
    return Event(type="failed", path=path, message=message)


def cancelled(message: str = "Cancelled by user") -> Event:
    return Event(type="cancelled", message=message)
