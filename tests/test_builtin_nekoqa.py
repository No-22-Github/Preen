from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / "macos/Preen/Resources/Datasets/NekoQA200"


def _load():
    manifest = json.loads((RESOURCE / "manifest.json").read_text(encoding="utf-8"))
    data_bytes = (RESOURCE / manifest["data_file"]).read_bytes()
    return manifest, data_bytes, json.loads(data_bytes)


def test_builtin_nekoqa_release_artifact_is_complete_and_immutable():
    manifest, data_bytes, records = _load()
    assert manifest["id"] == "builtin:nekoqa_200"
    assert manifest["subset_version"] == "1.1.0-1"
    assert manifest["license"] == "Apache-2.0"
    assert len(records) == manifest["sample_count"] == 200
    assert hashlib.sha256(data_bytes).hexdigest() == manifest["sha256"]
    assert (RESOURCE / "LICENSE").read_text().startswith("                                 Apache License")
    assert "MindsRiverPonder" in (RESOURCE / "NOTICE.md").read_text()
    assert len(manifest["selection"]["source_indices"]) == 200


def test_builtin_nekoqa_records_match_fixed_unmodified_source_indices():
    source_path = ROOT / "train_data/NekoQA_10k/NekoQA-10K.json"
    if not source_path.exists():
        pytest.skip("full upstream source is not part of release checkouts")
    manifest, _, records = _load()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    indices = manifest["selection"]["source_indices"]
    assert records == [source[index] for index in indices]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == manifest["source"]["source_file_sha256"]


def test_builtin_nekoqa_content_gate():
    _, _, records = _load()
    assert all(set(record) == {"instruction", "output"} for record in records)
    assert all(record["instruction"].strip() and record["output"].strip() for record in records)

    forbidden = re.compile(
        "永远|离不开|不能没有|只属于|属于主人|绝对服从|无条件服从|什么都愿意|"
        "抛弃|结婚|娶你|嫁给|老婆|老公|恋人|女朋友|男朋友|我爱你|只爱|领证|"
        "私人财产|奴隶|自杀|自残|毒药|炸弹|枪支|犯罪|违法|做爱|裸体|乳房|"
        "胸部|内裤|接吻|亲一口|三围|舔你的脚|医疗建议|法律建议|投资建议|股票|"
        "贷款|身份证|手机号|住址|银行卡|主人说什么都"
    )
    assert not [record for record in records if forbidden.search(record["instruction"] + record["output"])]

    def normalized(value: str) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff]", "", value).lower()

    for field, threshold in (("instruction", 0.86), ("output", 0.92)):
        values = [normalized(record[field]) for record in records]
        for index, left in enumerate(values):
            assert all(
                difflib.SequenceMatcher(None, left, right).ratio() < threshold
                for right in values[index + 1:]
            )


def test_builtin_nekoqa_real_tokenizer_preflight_when_fixture_model_exists():
    model = ROOT / "models/converted/rwkv7-g1d-0.4b"
    if not model.exists():
        pytest.skip("local tokenizer fixture model is unavailable")
    from mlx_lm.utils import load_tokenizer
    from statetuner.inspection import inspect_data

    tokenizer = load_tokenizer(str(model), tokenizer_config_extra={"trust_remote_code": True})
    result = inspect_data(RESOURCE / "nekoqa_200.json", tokenizer, ctx_len=512)
    assert result.valid == result.total == 200
    assert result.target_fully_truncated == 0
    assert result.truncated == 0
    assert result.template == "qa"
