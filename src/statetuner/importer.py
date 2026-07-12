"""数据导入与格式转换(Phase 3 Spec §4)。

把外部数据集(Alpaca / ShareGPT / Messages / 裸 QA)探测格式 → 转内部标准 jsonl,
供 train 命令消费。UI 与 CLI 共用同一 service 层(§4.1 契约)。

内部标准 jsonl 字段名(Spec §1.3,与 data.py docstring 钉死的契约一致):
  - qa 模板:        {"prompt": ..., "response": ...}
  - instruction 模板: {"instruction": ..., "input": ..., "response": ...}

探测规则(§4.2,按优先级命中即停,采样前 N=50 行):
  1. Messages/ChatML:  messages: [{role, content}]
  2. ShareGPT:        conversations: [{from, value}]
  3. Alpaca:          instruction + output 键(input 可选)
  4. 裸 QA:           命中别名表中的一对键
  5. 全部不中 → unknown(走手动字段映射)

采样中 >10% 行不符合命中的 schema → 降级 unknown。

不支持 parquet(pyarrow ~100MB 伤 bundle),报错信息里给一句转换提示。
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

from .metadata import file_sha256

# ── 别名表(§4.2.4)──────────────────────────────────────────
# prompt 侧别名:仅当无 output 键时才认 instruction(避免与 Alpaca 的 instruction 冲突)
PROMPT_ALIASES: tuple[str, ...] = (
    "prompt", "question", "q", "query", "user", "问", "instruction",
)
# response 侧别名
RESPONSE_ALIASES: tuple[str, ...] = (
    "response", "answer", "a", "completion", "output", "assistant", "答",
)

# ShareGPT / Messages 里 user 角色的别名
USER_ROLES: frozenset[str] = frozenset({"user", "human"})
ASSISTANT_ROLES: frozenset[str] = frozenset({"assistant", "gpt", "model", "chatgpt"})
SYSTEM_ROLES: frozenset[str] = frozenset({"system", "prompt"})

SchemaName = Literal["messages", "sharegpt", "alpaca", "bare_qa", "unknown"]
TurnPolicy = Literal["first", "all"]

SAMPLE_LIMIT = 50         # 采样行数(§4.2)
CONFIDENCE_FLOOR = 0.9    # 命中率阈值(<此值降级 unknown,§4.2 末段:>10% 不符合)


# ── 数据结构 ────────────────────────────────────────────────

@dataclass(frozen=True)
class SchemaDetection:
    """探测结果(可序列化,写进 import.json sidecar)。

    sample 是前 3 条原文(§4.2.5:UI 在探测失败时展示前 3 行原始数据)。
    """

    schema: SchemaName
    prompt_keys: List[str]   # 命中的 prompt 侧键名(审计/UI 回显)
    response_keys: List[str]
    confidence: float        # 命中行占比
    sample: List[dict]       # 前 3 条原文
    total_sampled: int       # 实际采样行数

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "prompt_keys": self.prompt_keys,
            "response_keys": self.response_keys,
            "confidence": self.confidence,
            "total_sampled": self.total_sampled,
            # sample 一起序列化(serve detect_import 指令需要把前3条原文发回 UI,
            # 供探测失败时展示;write_import 落盘的 sidecar 不调 to_dict 的 sample
            # 字段——sidecar_payload 里只取 detection.to_dict() 的非 sample 部分,
            # 实际写盘不含样本原文,避免 sidecar 膨胀)。
            "sample": self.sample,
        }


@dataclass(frozen=True)
class ImportResult:
    """转换结果(内存态,write_import 落盘后产 ImportArtifact)。"""

    records: List[dict]                  # 标准记录(qa 两字段 / instruction 三字段)
    template: Literal["qa", "instruction"]
    dropped_system: int                  # 丢弃的 system 消息计数
    dropped_other: int                   # 其他无法归类的计数(如孤儿 assistant)
    qa_degradation_hint: bool            # Alpaca 且 input 全空 → 可选降级 qa
    detection: SchemaDetection
    turn_policy: TurnPolicy

    def to_dict(self) -> dict:
        return {
            "template": self.template,
            "turn_policy": self.turn_policy,
            "dropped_system": self.dropped_system,
            "dropped_other": self.dropped_other,
            "qa_degradation_hint": self.qa_degradation_hint,
            "record_count": len(self.records),
            "detection": self.detection.to_dict(),
        }


@dataclass(frozen=True)
class ImportArtifact:
    """落盘产物路径 + 校验信息。"""

    jsonl_path: Path
    sidecar_path: Path
    sha256: str            # 源文件 hash(可复现性验证,验收 e)
    record_count: int


@dataclass(frozen=True)
class RenderedSample:
    """单条样本套模板后的渲染预览(§4.4,UI 用颜色区分 prefix/target 段)。

    prefix_len 是 token 级边界(encode 后),不是字符级——颜色分界以此为准,
    与训练 mask 边界(encode_template_sample 的 prefix_len)完全一致。
    """

    full_text: str          # prefix + target 拼接(最终喂给模型的文本)
    prefix_len: int         # token 级 prefix 长度(颜色分界)
    prompt_text: str        # 原始 prompt(审计)
    response_text: str      # 原始 response(审计)
    truncated: bool         # 是否因 ctx_len 截断(预览阶段一般 False,留扩展)

    def to_dict(self) -> dict:
        return {
            "full_text": self.full_text,
            "prefix_len": self.prefix_len,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "truncated": self.truncated,
        }


# ── 读取:多格式入口(§4.1)────────────────────────────────────

def read_records(path: Path) -> List[dict]:
    """读取 .jsonl / .json(数组或对象包 list)/ .csv(utf-8,stdlib)。

    parquet 显式不支持,报错信息里给转换提示(§4.1 末段)。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        raise ValueError(
            "parquet 暂不支持(pyarrow 依赖 ~100MB,伤 bundle 体积)。"
            "请先用 `python -c \"import pandas,json,sys; "
            "print(json.dumps(pandas.read_parquet(sys.argv[1]).to_dict('records')))\" "
            f"{path} > out.json` 转成 jsonl/json 后再导入。"
        )
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    if suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict):
            # 对象包 list:取第一个 list 值(常见 {"data": [...]} / {"train": [...]} 包装)
            for value in loaded.values():
                if isinstance(value, list):
                    return value
            return [loaded]
        raise ValueError(f"{path}: 顶层 JSON 既非数组也非对象")
    # 默认按 jsonl 读(无后缀也走这里)
    items: List[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {lineno} 行 JSON 解析失败: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path} 第 {lineno} 行不是 JSON 对象")
        items.append(item)
    return items


# ── 探测(§4.2)──────────────────────────────────────────────

def _looks_like_messages(item: dict) -> bool:
    msgs = item.get("messages")
    return isinstance(msgs, list) and len(msgs) > 0 and all(
        isinstance(m, dict) and "role" in m and "content" in m for m in msgs
    )


def _looks_like_sharegpt(item: dict) -> bool:
    convs = item.get("conversations")
    return isinstance(convs, list) and len(convs) > 0 and all(
        isinstance(c, dict) and "from" in c and "value" in c for c in convs
    )


def _looks_like_alpaca(item: dict) -> bool:
    return "instruction" in item and "output" in item


def _find_bare_qa_keys(item: dict) -> Optional[Tuple[str, str]]:
    """裸 QA:命中别名表中的一对键。返回 (prompt_key, response_key) 或 None。"""
    # response 侧先匹配(优先级:显式 response > answer > output ...)
    response_key = None
    for alias in RESPONSE_ALIASES:
        if alias in item:
            response_key = alias
            break
    if response_key is None:
        return None
    # prompt 侧:instruction 仅当无 output 键时才认(避免抢 Alpaca 的)
    prompt_key = None
    for alias in PROMPT_ALIASES:
        if alias == "instruction" and "output" in item:
            continue  # 让给 Alpaca
        if alias in item:
            prompt_key = alias
            break
    if prompt_key is None or prompt_key == response_key:
        return None
    return (prompt_key, response_key)


def detect_schema(items: List[dict]) -> SchemaDetection:
    """按优先级探测 schema(§4.2)。

    采样前 SAMPLE_LIMIT 行,按 Messages → ShareGPT → Alpaca → 裸 QA → unknown 顺序,
    命中即停;命中率 < CONFIDENCE_FLOOR(90%)→ 降级 unknown(§4.2 末段)。
    """
    sample_items = items[:SAMPLE_LIMIT]
    total = len(sample_items)
    if total == 0:
        return SchemaDetection(
            schema="unknown", prompt_keys=[], response_keys=[],
            confidence=0.0, sample=[], total_sampled=0,
        )

    # 按优先级尝试每个 schema,统计命中率
    for detector, schema_name in (
        (_looks_like_messages, "messages"),
        (_looks_like_sharegpt, "sharegpt"),
        (_looks_like_alpaca, "alpaca"),
    ):
        hits = sum(1 for item in sample_items if detector(item))
        confidence = hits / total
        if confidence >= CONFIDENCE_FLOOR:
            return SchemaDetection(
                schema=schema_name,
                prompt_keys=_schema_prompt_keys(schema_name),
                response_keys=_schema_response_keys(schema_name),
                confidence=round(confidence, 3),
                sample=sample_items[:3],
                total_sampled=total,
            )

    # 裸 QA:键名可能因数据集不同,从实际命中行里收
    bare_hits = 0
    prompt_keys_seen: set[str] = set()
    response_keys_seen: set[str] = set()
    for item in sample_items:
        keys = _find_bare_qa_keys(item)
        if keys is not None:
            bare_hits += 1
            prompt_keys_seen.add(keys[0])
            response_keys_seen.add(keys[1])
    confidence = bare_hits / total
    if confidence >= CONFIDENCE_FLOOR:
        return SchemaDetection(
            schema="bare_qa",
            prompt_keys=sorted(prompt_keys_seen),
            response_keys=sorted(response_keys_seen),
            confidence=round(confidence, 3),
            sample=sample_items[:3],
            total_sampled=total,
        )

    return SchemaDetection(
        schema="unknown",
        prompt_keys=[],
        response_keys=[],
        confidence=round(confidence, 3),  # 裸 QA 的残余命中率(信息性)
        sample=sample_items[:3],
        total_sampled=total,
    )


def _schema_prompt_keys(schema: SchemaName) -> List[str]:
    return {
        "messages": ["messages[].role==user"],
        "sharegpt": ["conversations[].from==human"],
        "alpaca": ["instruction"],
        "bare_qa": [],
        "unknown": [],
    }.get(schema, [])


def _schema_response_keys(schema: SchemaName) -> List[str]:
    return {
        "messages": ["messages[].role==assistant"],
        "sharegpt": ["conversations[].from==gpt"],
        "alpaca": ["output"],
        "bare_qa": [],
        "unknown": [],
    }.get(schema, [])


# ── 转换(§4.3)──────────────────────────────────────────────

def convert(
    items: List[dict],
    detection: SchemaDetection,
    *,
    turn_policy: TurnPolicy = "first",
) -> ImportResult:
    """按探测到的 schema 把原始记录转成内部标准记录。

    turn_policy(ShareGPT/Messages 多轮拆分,§4.3):
      first(默认):只取首对 user/assistant
      all:         每个相邻 user/assistant 对独立成样本
    system 消息:两种策略下都丢弃并计数(§4.3)。
    """
    if turn_policy not in ("first", "all"):
        raise ValueError(f"turn_policy 只支持 first / all, 收到 {turn_policy!r}")

    records: List[dict] = []
    dropped_system = 0
    dropped_other = 0

    if detection.schema == "alpaca":
        template = "instruction"
        qa_degradation_hint = True
        for item in items:
            instruction = (item.get("instruction") or "").strip()
            inp = (item.get("input") or "").strip()
            output = (item.get("output") or "").strip()
            if not instruction or not output:
                dropped_other += 1
                continue
            records.append({
                "instruction": instruction,
                "input": inp,
                "response": output,
            })
            if inp:
                qa_degradation_hint = False  # 有非空 input → 不降级
        return ImportResult(
            records=records, template=template,
            dropped_system=0, dropped_other=dropped_other,
            qa_degradation_hint=qa_degradation_hint,
            detection=detection, turn_policy=turn_policy,
        )

    if detection.schema in ("sharegpt", "messages"):
        template = "qa"
        conv_key = "conversations" if detection.schema == "sharegpt" else "messages"
        role_key = "from" if detection.schema == "sharegpt" else "role"
        text_key = "value" if detection.schema == "sharegpt" else "content"
        for item in items:
            convs = item.get(conv_key) or []
            if not isinstance(convs, list):
                dropped_other += 1
                continue
            # 先过滤 system(丢弃计数),保留 user/assistant 序列
            pairs: List[Tuple[str, str]] = []
            pending_user: Optional[str] = None
            for turn in convs:
                if not isinstance(turn, dict):
                    dropped_other += 1
                    continue
                role = str(turn.get(role_key, "")).lower()
                text = (turn.get(text_key) or "").strip()
                if role in SYSTEM_ROLES or not text:
                    if role in SYSTEM_ROLES:
                        dropped_system += 1
                    continue
                if role in USER_ROLES:
                    if pending_user is not None:
                        dropped_other += 1  # 连续两个 user,前一个孤儿
                    pending_user = text
                elif role in ASSISTANT_ROLES:
                    if pending_user is None:
                        dropped_other += 1  # 孤儿 assistant
                        continue
                    pairs.append((pending_user, text))
                    pending_user = None
                else:
                    dropped_other += 1
            if not pairs:
                continue
            if turn_policy == "first":
                records.append({
                    "prompt": pairs[0][0], "response": pairs[0][1],
                })
            else:  # all
                for u, a in pairs:
                    records.append({"prompt": u, "response": a})
        return ImportResult(
            records=records, template=template,
            dropped_system=dropped_system, dropped_other=dropped_other,
            qa_degradation_hint=False,
            detection=detection, turn_policy=turn_policy,
        )

    if detection.schema == "bare_qa":
        template = "qa"
        for item in items:
            keys = _find_bare_qa_keys(item)
            if keys is None:
                dropped_other += 1
                continue
            prompt = (item.get(keys[0]) or "").strip()
            response = (item.get(keys[1]) or "").strip()
            if not prompt or not response:
                dropped_other += 1
                continue
            records.append({"prompt": prompt, "response": response})
        return ImportResult(
            records=records, template=template,
            dropped_system=0, dropped_other=dropped_other,
            qa_degradation_hint=False,
            detection=detection, turn_policy=turn_policy,
        )

    # unknown:不转换,抛错让调用方(UI/CLI)走手动映射
    raise ValueError(
        f"探测结果 unknown,无法自动转换(请手动指定 prompt/response 字段)。"
        f" confidence={detection.confidence:.2f}, sample keys="
        f"{list(detection.sample[0].keys()) if detection.sample else []}"
    )


# ── 落盘(§4.3 产物 + sidecar)───────────────────────────────

def _strip_sample_from_result_dict(result_dict: dict) -> dict:
    """从 ImportResult.to_dict() 的结果里剥掉 detection.sample。

    to_dict 为了 serve detect_import 指令序列化了 sample(前3条原文,供 UI 展示),
    但落盘 sidecar 不该带样本原文(来源文件已有,sidecar 只做元数据追溯,避免膨胀)。
    """
    out = dict(result_dict)
    if "detection" in out and isinstance(out["detection"], dict):
        det = dict(out["detection"])
        det.pop("sample", None)
        out["detection"] = det
    return out

def write_import(
    source_path: Path,
    result: ImportResult,
    out_path: Path,
) -> ImportArtifact:
    """写标准 jsonl + sidecar import.json,返回 artifact(含源文件 hash)。

    jsonl 每行一个标准记录(qa 两字段 / instruction 三字段)。
    sidecar 记录来源 hash、探测结果、策略、丢弃计数(§4.3 末段,可追溯)。
    """
    source_path = Path(source_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for record in result.records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    source_sha = file_sha256(source_path)
    sidecar_path = out_path.with_suffix(out_path.suffix + ".import.json")
    sidecar_payload = {
        "format_version": 1,
        "source": {
            "path": str(source_path),
            "sha256": source_sha,
        },
        # to_dict 现含 detection.sample(serve detect_import 需要),但 sidecar
        # 落盘不该带样本原文(可能很大,且来源文件已有,sidecar 只做元数据追溯)。
        # 这里显式剥掉 sample。
        "result": _strip_sample_from_result_dict(result.to_dict()),
    }
    tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(sidecar_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(sidecar_path)

    return ImportArtifact(
        jsonl_path=out_path,
        sidecar_path=sidecar_path,
        sha256=source_sha,
        record_count=len(result.records),
    )


# ── 预览(§4.4)──────────────────────────────────────────────

def preview_records(
    records: List[dict],
    *,
    template: Literal["qa", "instruction"],
    tokenizer,
    n: int = 3,
) -> List[RenderedSample]:
    """渲染前 n 条样本套模板后的最终喂入文本 + token 级 mask 边界(§4.4)。

    这是"RWKV 对格式敏感"的产品化回应:train/inference 同构在代码层已保证,
    预览把保证可视化。颜色分界用 prefix_len(token 级,与训练 mask 边界一致)。

    复用 templates.QA/INSTRUCTION 渲染,不手写 \\n 拼接(仓库铁律)。
    """
    from .templates import INSTRUCTION, QA

    if template not in ("qa", "instruction"):
        raise ValueError(f"预览模板只支持 qa / instruction, 收到 {template!r}")
    tmpl = QA if template == "qa" else INSTRUCTION

    rendered: List[RenderedSample] = []
    for record in records[:n]:
        if template == "qa":
            prompt_text = (record.get("prompt") or "").strip()
            response_text = (record.get("response") or "").strip()
            prefix = tmpl.format_prefix(q=prompt_text)
        else:
            prompt_text = (record.get("instruction") or "").strip()
            inp = record.get("input") or ""
            response_text = (record.get("response") or "").strip()
            prefix = tmpl.format_prefix(instruction=prompt_text, input=inp)
        target = tmpl.format_target(a=response_text)
        full_text = prefix + target
        prefix_len = len(tokenizer.encode(prefix))
        rendered.append(RenderedSample(
            full_text=full_text,
            prefix_len=prefix_len,
            prompt_text=prompt_text,
            response_text=response_text,
            truncated=False,
        ))
    return rendered


# ── 一站式入口(CLI/UI 共用)─────────────────────────────────

def import_dataset(
    source_path: Path,
    out_path: Path,
    *,
    turn_policy: TurnPolicy = "first",
) -> Tuple[ImportArtifact, ImportResult]:
    """探测 → 转换 → 落盘 一站式(CLI import 命令直接调)。"""
    source_path = Path(source_path)
    items = read_records(source_path)
    detection = detect_schema(items)
    result = convert(items, detection, turn_policy=turn_policy)
    artifact = write_import(source_path, result, out_path)
    return artifact, result
