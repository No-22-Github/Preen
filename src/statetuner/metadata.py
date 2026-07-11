"""训练产物旁挂元数据。"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_state_metadata(
    state_path: Path,
    *,
    model_path: Path,
    data_path: Path,
    template: str,
    config: dict,
    data_stats: dict,
    result: dict,
    pth_path: Path | None = None,
) -> Path:
    """在 state 旁写 `<stem>.meta.json`，供 CLI/未来 UI 读取。"""
    meta_path = state_path.with_suffix(".meta.json")
    payload = {
        "format_version": 1,
        "created_at": time.time(),
        "model": str(model_path),
        "data": str(data_path),
        "data_sha256": file_sha256(data_path),
        "template": template,
        "precision": {"weights": "bf16", "train_state": "fp32", "export": "fp32"},
        "config": config,
        "data_stats": data_stats,
        "result": result,
        "artifacts": {
            "state_npz": str(state_path),
            "state_pth": str(pth_path) if pth_path else None,
        },
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(meta_path)
    return meta_path
