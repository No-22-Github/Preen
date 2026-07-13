"""数据集完整预览的磁盘分页缓存。

缓存只承载已经套用训练模板、完成 tokenizer 统计的展示数据。App 每次只读取
一页，避免通过 JSON Lines 一次传输上千个长字符串。
"""
from __future__ import annotations

import json
import math
from itertools import islice
from pathlib import Path
from typing import Any, Optional


FORMAT_VERSION = 1


def metadata_path(cache_path: Path) -> Path:
    cache_path = Path(cache_path)
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


class PreviewCacheWriter:
    """流式写入渲染样本，并只在内存保留首页。"""

    def __init__(self, path: Path, *, page_size: int):
        if page_size <= 0:
            raise ValueError("page_size 必须 > 0")
        self.path = Path(path)
        self.page_size = page_size
        self.first_page: list[dict[str, Any]] = []
        self.total = 0
        self._tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self._file = None
        self._committed = False

    def __enter__(self) -> "PreviewCacheWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_path.unlink(missing_ok=True)
        self._file = self._tmp_path.open("w", encoding="utf-8")
        return self

    def append(self, sample: dict[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError("PreviewCacheWriter 尚未打开")
        self._file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        if len(self.first_page) < self.page_size:
            self.first_page.append(sample)
        self.total += 1

    def commit(self, *, template: str, ctx_len: int) -> dict[str, Any]:
        if self._file is None:
            raise RuntimeError("PreviewCacheWriter 尚未打开")
        self._file.close()
        self._file = None
        self._tmp_path.replace(self.path)

        meta = {
            "format_version": FORMAT_VERSION,
            "total": self.total,
            "page_size": self.page_size,
            "page_count": math.ceil(self.total / self.page_size),
            "template": template,
            "ctx_len": ctx_len,
        }
        meta_path = metadata_path(self.path)
        meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
        meta_tmp.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        meta_tmp.replace(meta_path)
        self._committed = True
        return meta

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if not self._committed:
            self._tmp_path.unlink(missing_ok=True)


def read_preview_cache_page(
    cache_path: Path, *, page: int, page_size: Optional[int] = None,
) -> dict[str, Any]:
    """读取一页缓存；不加载 tokenizer，也不把其他页留在内存。"""
    cache_path = Path(cache_path)
    if page <= 0:
        raise ValueError("page 必须 >= 1")
    if not cache_path.is_file():
        raise ValueError("预览缓存已失效，请重新检查数据集")
    meta_path = metadata_path(cache_path)
    if not meta_path.is_file():
        raise ValueError("预览缓存元数据缺失，请重新检查数据集")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError("预览缓存版本不兼容，请重新检查数据集")
    effective_size = int(page_size or meta["page_size"])
    if effective_size <= 0 or effective_size > 200:
        raise ValueError("page_size 必须在 1...200 之间")
    total = int(meta["total"])
    page_count = math.ceil(total / effective_size)
    if page_count and page > page_count:
        raise ValueError(f"页码超出范围: {page}/{page_count}")

    start = (page - 1) * effective_size
    with cache_path.open("r", encoding="utf-8") as handle:
        samples = [json.loads(line) for line in islice(handle, start, start + effective_size)]
    return {
        "preview": samples,
        "page": page,
        "page_size": effective_size,
        "page_count": page_count,
        "total": total,
    }
