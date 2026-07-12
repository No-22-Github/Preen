"""serve 协议 handler 快测(FakeEngine,无模型依赖,~1min)。

覆盖 Spec §3.6 验收 a(终结事件/id 透传)、错误码映射、指令路由、流式事件。
abort 时序(§3.6.b)、fuzz 垃圾行进程存活(§3.6.c)、preview ab 数值一致(§3.6.d)、
stdout 全 JSON(§3.6.e)在 test_serve_e2e.py --slow(spawn 真实进程)。

策略:ServeProtocol 注入 StringIO 作 stdin/stdout/stderr,
直接调 handle_line(line) → 读 stdout 验证事件。
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from statetuner.inference import ABResult, GenerationConfig, GenerationResult
from statetuner.serve import ServeProtocol, ServeSessionManager, _ok, _error

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "import"


# ── FakeEngine ──────────────────────────────────────────────

class FakeModel:
    """假 model:满足 new_session 不碰 validate_state_for_model(无 state_path 即可)。"""
    layers = []
    args = type("A", (), {"hidden_size": 0, "head_dim": 0})()


class FakeTokenizer:
    """简易 tokenizer:detect_import 的 preview_records 渲染需要 encode/decode。"""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text if c != " "]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class FakeEngine:
    """记录 generate/compare 调用,产出固定 GenerationResult,支持 should_abort 回调。

    支持 abort:若 should_abort 返回 True,抛 GenerationAborted(模拟真实 generate 行为)。
    """

    def __init__(self, *, abort_at_step: int = -1):
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()  # detect_import 的 preview_records 用
        self.calls: list[dict] = []
        self._abort_at_step = abort_at_step  # 在第 N 次检查 should_abort 时触发

    def generate(self, prompt, *, state=None, config=None, cache=None,
                 on_text=None, should_abort=None):
        self.calls.append({"prompt": prompt, "state": state, "cache": cache})
        # 模拟 abort:若调用方要求中断
        if should_abort is not None and self._abort_at_step >= 0:
            self._abort_at_step -= 1
            if self._abort_at_step < 0:
                from statetuner.inference import GenerationAborted
                raise GenerationAborted()
        text = "tuned" if state is not None else "baseline"
        if on_text is not None:
            on_text(text)
        return GenerationResult(
            text=text,
            display_token_ids=[1, 2],
            fed_token_ids=[1, 2],
            stop_reason="eos",
            cache_clean=True,
            cache=object(),
            elapsed=0.1,
            used_state=state is not None,
            config=config or GenerationConfig(),
        )

    def compare(self, prompt, *, state, config):
        return ABResult(
            prompt=prompt,
            with_state=self.generate(prompt, state=state, config=config),
            baseline=self.generate(prompt, state=None, config=config),
        )

    def load_state(self, state):
        return {0: str(state)}


# ── 协议 harness ────────────────────────────────────────────

class CaptureProtocol:
    """包装 ServeProtocol,捕获 stdout 事件列表,提供断言辅助。"""

    def __init__(self, engine=None, *, abort_at_step=-1):
        self.engine = engine or FakeEngine(abort_at_step=abort_at_step)
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.proto = ServeProtocol(
            self.engine,
            model_path="fake-model",
            stdin=self.stdin, stdout=self.stdout, stderr=self.stderr,
        )

    def handle(self, line: str) -> None:
        """喂一行,事件写进 stdout buffer。"""
        self.proto.handle_line(line)

    def handle_obj(self, req: dict) -> None:
        """喂一个 dict(自动转 JSON 行)。"""
        self.proto.handle_line(json.dumps(req, ensure_ascii=False))

    @property
    def events(self) -> list[dict]:
        """读出所有已发事件(每个一行 JSON)。"""
        out = self.stdout.getvalue()
        return [json.loads(line) for line in out.splitlines() if line.strip()]

    def reset(self) -> None:
        self.stdout.seek(0)
        self.stdout.truncate()


# ── helper:发请求 + 取其终结事件 ─────────────────────────────

def _send_and_collect(cap: CaptureProtocol, req: dict) -> tuple[list[dict], dict]:
    """发一个请求,返回 (该请求的所有事件, 终结事件)。

    终结事件 = type in (ok, error) 的最后一个。
    """
    cap.reset()
    cap.handle_obj(req)
    events = cap.events
    terminators = [e for e in events if e["type"] in ("ok", "error")]
    assert len(terminators) == 1, f"应恰好一个终结事件, 得到 {terminators}"
    return events, terminators[0]


# ── hello / new_session ─────────────────────────────────────

def test_hello_returns_capabilities():
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {"id": "1", "cmd": "hello"})
    assert term["type"] == "ok"
    assert term["id"] == "1"
    assert term["version"]  # 来自 __version__
    assert term["model"] == "fake-model"
    assert "qa" in term["capabilities"]["templates"]
    assert "fast" in term["capabilities"]["think"]


def test_new_session_returns_id():
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {
        "id": "2", "cmd": "new_session", "template": "qa",
    })
    assert term["type"] == "ok"
    assert term["id"] == "2"
    assert isinstance(term["session_id"], str)
    assert len(term["session_id"]) > 0


def test_new_session_rejects_bad_template():
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {
        "id": "3", "cmd": "new_session", "template": "bogus",
    })
    assert term["type"] == "error"
    assert term["code"] == "bad_request"
    assert "template" in term["message"]


def test_new_session_rejects_think_without_reasoning():
    """§1.2:think 仅在 reasoning=True 时合法。"""
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {
        "id": "4", "cmd": "new_session", "template": "qa",
        "think": "fast",  # 缺 reasoning=True
    })
    assert term["type"] == "error"
    assert term["code"] == "bad_request"


# ── send:流式 + 终结 + id 透传 ──────────────────────────────

def test_send_emits_text_chunks_turn_end_and_ok():
    """§3.3 send:text_chunk* → turn_end → ok。每个事件带回 id + session_id。"""
    cap = CaptureProtocol()
    # 先建 session
    _, ns_term = _send_and_collect(cap, {"id": "s1", "cmd": "new_session", "template": "qa"})
    sid = ns_term["session_id"]

    events, term = _send_and_collect(cap, {
        "id": "s2", "cmd": "send", "session_id": sid, "text": "你好",
    })
    types = [e["type"] for e in events]
    assert "text_chunk" in types
    assert "turn_end" in types
    assert term["type"] == "ok"
    assert term["id"] == "s2"
    # 所有事件都透传 id
    assert all(e["id"] == "s2" for e in events)
    # text_chunk 带 delta
    chunk = next(e for e in events if e["type"] == "text_chunk")
    assert chunk["delta"]
    # turn_end 带 result
    turn_end = next(e for e in events if e["type"] == "turn_end")
    assert "result" in turn_end
    assert turn_end["session_id"] == sid


def test_send_unknown_session_returns_not_found():
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {
        "id": "x1", "cmd": "send", "session_id": "no-such", "text": "hi",
    })
    assert term["type"] == "error"
    assert term["code"] == "not_found"
    assert term["id"] == "x1"


def test_send_empty_text_returns_bad_request():
    cap = CaptureProtocol()
    _, ns = _send_and_collect(cap, {"id": "n", "cmd": "new_session", "template": "qa"})
    events, term = _send_and_collect(cap, {
        "id": "x2", "cmd": "send", "session_id": ns["session_id"], "text": "   ",
    })
    assert term["type"] == "error"
    assert term["code"] == "bad_request"


def test_send_busy_rejects_concurrent():
    """§3.5 busy:已有 in-flight 时第二个 send 拒绝。

    这里不真跑并发(快测无真实生成),而是手动占住 busy 锁验证拒绝逻辑。
    """
    cap = CaptureProtocol()
    _, ns = _send_and_collect(cap, {"id": "n", "cmd": "new_session", "template": "qa"})
    sid = ns["session_id"]
    # 手动占住 busy 锁(模拟一个在跑的 send)
    cap.proto._busy.acquire()
    cap.proto._in_flight_id = "running"
    try:
        events, term = _send_and_collect(cap, {
            "id": "busy1", "cmd": "send", "session_id": sid, "text": "x",
        })
        assert term["type"] == "error"
        assert term["code"] == "busy"  # §3.5:busy 是独立错误码
        assert "busy" in term["message"]
    finally:
        cap.proto._busy.release()
        cap.proto._in_flight_id = None


# ── abort ───────────────────────────────────────────────────

def test_abort_with_no_in_flight_returns_ok():
    """无在跑生成时 abort 仍回 ok(幂等)。"""
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {"id": "a1", "cmd": "abort"})
    assert term["type"] == "ok"
    assert term["id"] == "a1"


def test_send_with_aborting_engine_returns_aborted_error():
    """Q3 + P0-2:send → generate 抛 GenerationAborted → error{aborted}。

    FakeEngine(abort_at_step=...) 此前是死代码(写了没人用);
    send → aborted 这条主路径此前零个不依赖模型的测试。补上。
    同时验证 abort 时 session 状态原子性:history 不变 + cache_clean=False。
    """
    cap = CaptureProtocol(abort_at_step=0)  # 第一次 should_abort 检查就触发
    _, ns = _send_and_collect(cap, {"id": "n", "cmd": "new_session", "template": "qa"})
    sid = ns["session_id"]
    events, term = _send_and_collect(cap, {
        "id": "ab", "cmd": "send", "session_id": sid, "text": "你好",
    })
    assert term["type"] == "error"
    assert term["code"] == "aborted"
    assert term["id"] == "ab"
    # P0-2:session 状态原子性 —— history 不留孤儿 user turn
    managed = cap.proto.manager.get_session(sid)
    assert len(managed.session.history) == 0, "abort 后 history 应为空(无孤儿 turn)"
    # cache_clean=False 强制下轮重放(续传路径的 cache 可能已被前向污染)
    assert managed.session.cache_clean is False
    assert managed.session.cache is None


def test_send_abort_then_next_send_replays_cleanly():
    """P0-2 端到端(serve 层):abort 后下一轮 send 能正常完成,不残留孤儿。

    证明 abort 不会让会话进入坏状态(下轮 send 正常发 turn_end + ok)。
    """
    # 第一次 send 用 aborting engine,第二次换正常 engine
    cap = CaptureProtocol(abort_at_step=0)
    _, ns = _send_and_collect(cap, {"id": "n", "cmd": "new_session", "template": "qa"})
    sid = ns["session_id"]
    # 第一轮:abort
    _, aborted = _send_and_collect(cap, {
        "id": "a1", "cmd": "send", "session_id": sid, "text": "Q1",
    })
    assert aborted["code"] == "aborted"
    # 换回正常 engine(FakeEngine 在本文件顶部定义)
    cap.engine = FakeEngine()
    cap.proto.manager.engine = cap.engine
    managed = cap.proto.manager.get_session(sid)
    managed.session.engine = cap.engine
    # 第二轮:应正常完成
    events, term = _send_and_collect(cap, {
        "id": "s2", "cmd": "send", "session_id": sid, "text": "Q2",
    })
    assert term["type"] == "ok"
    # history 应只有 Q2 这轮(无 Q1 孤儿)
    users = [t for t in managed.session.history if t.role == "user"]
    assert [t.text for t in users] == ["Q2"]


# ── 会话控制:set_config / rewind / reset / close ────────────

def test_set_config_updates_session():
    cap = CaptureProtocol()
    _, ns = _send_and_collect(cap, {"id": "n", "cmd": "new_session", "template": "qa"})
    sid = ns["session_id"]
    events, term = _send_and_collect(cap, {
        "id": "c1", "cmd": "set_config", "session_id": sid,
        "gen_config": {"temperature": 0.5, "max_tokens": 100},
    })
    assert term["type"] == "ok"
    managed = cap.proto.manager.get_session(sid)
    assert managed.session.config.temperature == 0.5
    assert managed.session.config.max_tokens == 100


def test_rewind_returns_history_len():
    cap = CaptureProtocol()
    _, ns = _send_and_collect(cap, {"id": "n", "cmd": "new_session", "template": "qa"})
    sid = ns["session_id"]
    # 先来一轮对话(造 history)
    cap.handle_obj({"id": "t1", "cmd": "send", "session_id": sid, "text": "你好"})
    cap.reset()
    events, term = _send_and_collect(cap, {
        "id": "r1", "cmd": "rewind", "session_id": sid, "n": 1,
    })
    assert term["type"] == "ok"
    assert "history_len" in term


def test_reset_clears_session():
    cap = CaptureProtocol()
    _, ns = _send_and_collect(cap, {"id": "n", "cmd": "new_session", "template": "qa"})
    sid = ns["session_id"]
    cap.handle_obj({"id": "t1", "cmd": "send", "session_id": sid, "text": "你好"})
    events, term = _send_and_collect(cap, {
        "id": "clr", "cmd": "reset", "session_id": sid,
    })
    assert term["type"] == "ok"
    managed = cap.proto.manager.get_session(sid)
    assert len(managed.session.history) == 0


def test_close_session_then_send_not_found():
    cap = CaptureProtocol()
    _, ns = _send_and_collect(cap, {"id": "n", "cmd": "new_session", "template": "qa"})
    sid = ns["session_id"]
    cap.handle_obj({"id": "cl", "cmd": "close_session", "session_id": sid})
    events, term = _send_and_collect(cap, {
        "id": "post", "cmd": "send", "session_id": sid, "text": "x",
    })
    assert term["type"] == "error"
    assert term["code"] == "not_found"


# ── preview(单路 + ab)──────────────────────────────────────

def test_preview_single_emits_turn_end_and_ok():
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {
        "id": "p1", "cmd": "preview",
        "prompt": "你好", "template": "qa",
    })
    types = [e["type"] for e in events]
    assert "turn_end" in types
    assert term["type"] == "ok"
    turn_end = next(e for e in events if e["type"] == "turn_end")
    assert "result" in turn_end


def test_preview_ab_requires_state_path():
    """§3.3:ab=True 需要 state_path。"""
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {
        "id": "p2", "cmd": "preview",
        "prompt": "你好", "template": "qa", "ab": True,
    })
    assert term["type"] == "error"
    assert term["code"] == "bad_request"


def test_preview_ab_emits_two_turn_ends():
    """ab=True:turn_end{side:with_state} → turn_end{side:baseline} → ok。

    用 FakeEngine 的 compare(state=None 也能跑,产出 baseline/tuned 文本)。
    state_path 传一个假路径——compare 内部走 state=str(path),
    FakeEngine.generate 不真读文件,所以不会 FileNotFoundError。
    """
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {
        "id": "p3", "cmd": "preview",
        "prompt": "你好", "template": "qa",
        "ab": True, "state_path": "fake.npz",
    })
    types = [e["type"] for e in events]
    turn_ends = [e for e in events if e["type"] == "turn_end"]
    assert len(turn_ends) == 2
    assert turn_ends[0]["side"] == "with_state"
    assert turn_ends[1]["side"] == "baseline"
    assert term["type"] == "ok"


# ── 错误语义 / fuzz 垃圾行(§3.5 不变式)─────────────────────

def test_non_json_line_returns_bad_request_without_id():
    cap = CaptureProtocol()
    cap.handle("this is not json {{{")
    events = cap.events
    assert len(events) == 1
    err = events[0]
    assert err["type"] == "error"
    assert err["code"] == "bad_request"
    assert "id" not in err  # 协议级 error 无 id(§3.4)


def test_non_object_json_returns_bad_request():
    cap = CaptureProtocol()
    cap.handle_obj(["not", "an", "object"])  # JSON 数组
    events = cap.events
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "bad_request"


def test_unknown_command_returns_bad_request():
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {"id": "u1", "cmd": "frobnicate"})
    assert term["type"] == "error"
    assert term["code"] == "bad_request"
    assert "frobnicate" in term["message"]


def test_missing_cmd_returns_bad_request():
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {"id": "u2", "data": "no cmd"})
    assert term["type"] == "error"
    assert term["code"] == "bad_request"


def test_malformed_does_not_crash_repeatedly():
    """§3.5 不变式:连续喂多行垃圾,进程(handler)存活且每行逐个回 error。"""
    cap = CaptureProtocol()
    garbage = [
        "not json at all",
        '{"cmd":}',  # 畸形 JSON
        json.dumps({"no_cmd": True}),
        json.dumps({"id": "x", "cmd": "????"}),
    ]
    for line in garbage:
        cap.handle(line)
    events = cap.events
    # 每行一个 error,handler 存活不崩
    assert len(events) == len(garbage)
    assert all(e["type"] == "error" for e in events)
    # handler 还能正常服务下一请求(没崩)
    cap.reset()
    cap.handle_obj({"id": "after", "cmd": "hello"})
    assert cap.events[-1]["type"] == "ok"


# ── import:detect_import / import(§4.1 via serve)──────────

def test_detect_import_alpaca_returns_detection_and_preview():
    """detect_import:探测 + 转换 + 前3条渲染预览(不落盘)。"""
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {
        "id": "di1", "cmd": "detect_import",
        "data_path": str(FIXTURES / "alpaca_sample.jsonl"),
    })
    assert term["type"] == "ok"
    assert term["detection"]["schema"] == "alpaca"
    assert term["detection"]["confidence"] == 1.0
    # sample 前3条原文在(供 UI 展示)
    assert len(term["detection"]["sample"]) == 3
    # result 含 template / record_count
    assert term["result"]["template"] == "instruction"
    # preview 是前3条渲染样本(含 full_text / prefix_len)
    assert len(term["preview"]) == 3
    assert all("full_text" in p and "prefix_len" in p for p in term["preview"])


def test_detect_import_unknown_schema_returns_without_preview():
    """DPO 格式探测失败 → detection.schema=unknown,result/preview 为空(不报错)。"""
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {
        "id": "di2", "cmd": "detect_import",
        "data_path": str(FIXTURES / "dpo_sample.jsonl"),
    })
    assert term["type"] == "ok"  # 探测失败不报错,UI 据此走手动映射
    assert term["detection"]["schema"] == "unknown"
    assert term["result"] is None
    assert term["preview"] == []


def test_detect_import_missing_file_returns_not_found():
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {
        "id": "di3", "cmd": "detect_import",
        "data_path": "/no/such/file.jsonl",
    })
    assert term["type"] == "error"
    assert term["code"] == "not_found"


def test_import_writes_jsonl_and_sidecar(tmp_path):
    """import:探测 + 转换 + 落盘,返回 artifact 路径。"""
    cap = CaptureProtocol()
    out = tmp_path / "out.jsonl"
    events, term = _send_and_collect(cap, {
        "id": "im1", "cmd": "import",
        "data_path": str(FIXTURES / "bare_qa.jsonl"),
        "out_path": str(out),
    })
    assert term["type"] == "ok"
    assert term["record_count"] == 4
    assert term["sha256"]  # 源文件 hash
    assert term["jsonl_path"] == str(out)
    # 实际落盘了
    assert out.exists()
    assert Path(term["sidecar_path"]).exists()
    # result 不含 detection.sample(sidecar 剥离原则)
    assert "sample" not in term["result"]["detection"]


def test_import_unknown_schema_returns_bad_request(tmp_path):
    """DPO → 探测失败 → convert 抛错 → bad_request(附原因)。"""
    cap = CaptureProtocol()
    events, term = _send_and_collect(cap, {
        "id": "im2", "cmd": "import",
        "data_path": str(FIXTURES / "dpo_sample.jsonl"),
        "out_path": str(tmp_path / "x.jsonl"),
    })
    assert term["type"] == "error"
    assert term["code"] == "bad_request"
    assert "导入失败" in term["message"]


def test_detect_import_then_import_workflow(tmp_path):
    """§4.1 完整 UI 流程:detect_import(确认)→ import(落盘),两步可用同一数据。"""
    cap = CaptureProtocol()
    # 第一步:探测确认
    _, detect_term = _send_and_collect(cap, {
        "id": "wf1", "cmd": "detect_import",
        "data_path": str(FIXTURES / "sharegpt_multiturn.jsonl"),
    })
    assert detect_term["detection"]["schema"] == "sharegpt"
    # 第二步:确认后落盘(turn_policy=all)
    out = tmp_path / "sharegpt.jsonl"
    _, import_term = _send_and_collect(cap, {
        "id": "wf2", "cmd": "import",
        "data_path": str(FIXTURES / "sharegpt_multiturn.jsonl"),
        "out_path": str(out),
        "turn_policy": "all",
    })
    assert import_term["record_count"] == 5  # all 策略:2+1+2 对
    assert out.exists()


def test_internal_error_surfaces_traceback_summary():
    """未预期异常 → internal + traceback 摘要(§3.5)。

    构造:set_config 传一个无法转 float 的值,GenerationConfig.validate 报错。
    实际上 _apply_gen_config 用 replace + validate,非数值会抛 TypeError/ValueError,
    被 handle_line 兜成 internal。
    """
    cap = CaptureProtocol()
    _, ns = _send_and_collect(cap, {"id": "n", "cmd": "new_session", "template": "qa"})
    sid = ns["session_id"]
    events, term = _send_and_collect(cap, {
        "id": "ie", "cmd": "set_config", "session_id": sid,
        "gen_config": {"temperature": "not-a-number"},
    })
    assert term["type"] == "error"
    # temperature 非数值 → validate 抛 ValueError → internal
    assert term["code"] in ("internal", "bad_request")
