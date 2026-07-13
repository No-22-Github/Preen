"""statetuner serve — 常驻推理进程,stdin/stdout JSON lines 协议(Phase 3 Spec §3)。

进程模型(§3.1):单进程单模型,启动加载一个模型常驻。换模型 = UI 重启 serve。
同一时刻至多一个 in-flight 生成,取消用协议指令 abort(不是信号)。

帧格式(§3.2):
  请求:一行一个 JSON 对象 {"id", "cmd", ...params}
  响应:一行一个 JSON 事件。由请求触发的事件带回 "id" 原样透传。
  每个请求必然以一个终结事件收尾:{"id","type":"ok",...} 或
  {"id","type":"error","code","message"}。中间可有任意流式事件。
  stdout 只有 JSON 行,人类可读日志走 stderr。

指令集(§3.3):hello / new_session / send / abort / set_state / set_config /
  rewind / reset / close_session / preview / shutdown。

错误语义(§3.5):bad_request / not_found / busy / aborted / internal。
  协议不变式:任何输入行都不能让 serve 进程崩溃。

abort 机制:单独读线程持续 readline 入队;abort 指令 → threading.Event.set();
  generate 每步检查 Event(should_abort 回调)→ 抛 GenerationAborted。
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import __version__
from .chat import ChatSession
from .inference import (
    GenerationAborted,
    GenerationConfig,
    InferenceEngine,
    render_prompt,
    with_template_stops,
)

# 协议级能力声明(§3.3 hello)
CAPABILITIES = {
    "templates": ["qa", "instruction", "raw"],
    "think": ["off", "fast", "on"],
    "reasoning": True,
}

# 协议版本(T1,协议冻结点)。
# 主版本号:turn_end / text_chunk / hello 的字段结构有破坏性变更时 +1。
# 当前 1:首版含 thinking/answer/phase 字段(本次 commit 引入)。
# 此后改协议字段结构必须先 bump 此号,UI 据此拒绝不兼容的 serve。
PROTOCOL_VERSION = 1


# ── 事件构造助手(§3.4)──────────────────────────────────────

def _event(
    type_: str,
    *,
    id_: Optional[str] = None,
    session_id: Optional[str] = None,
    **fields,
) -> dict:
    """构造一个事件 dict,id/session_id 为 None 时不写入字段。"""
    evt: dict = {"type": type_}
    if id_ is not None:
        evt["id"] = id_
    if session_id is not None:
        evt["session_id"] = session_id
    evt.update(fields)
    return evt


def _ok(id_: Optional[str] = None, **payload) -> dict:
    """成功终结事件。"""
    return _event("ok", id_=id_, **payload)


def _error(
    code: str,
    message: str,
    *,
    id_: Optional[str] = None,
) -> dict:
    """失败终结事件。无 id 的 error 表示协议级错误(§3.4)。"""
    return _event("error", id_=id_, code=code, message=message)


class ProtocolError(Exception):
    """协议级错误(转 error 事件,不让进程崩)。

    code 对齐 Spec §3.5 的 5 个错误码之一:
    bad_request / not_found / busy / aborted / internal。
    默认 bad_request(参数校验);not_found / busy 由调用方显式指定。
    废弃旧实现基于 message 字符串匹配的错误码推断(附录 E.4 裁决)。
    """

    def __init__(self, message: str, *, code: str = "bad_request"):
        super().__init__(message)
        self.code = code


# ── 会话管理器 ──────────────────────────────────────────────

@dataclass
class ManagedSession:
    """单个被管理的 ChatSession + 元数据。"""

    session: ChatSession
    session_id: str
    template: str
    reasoning: bool
    think: str
    state_path: Optional[str]


class ServeSessionManager:
    """管理多个 ChatSession(按 session_id 索引),复用单个常驻 engine。

    多轮续传/重放逻辑由 ChatSession 负责(§2 已完成),本类只做 session 生命周期
    管理 + 指令到 ChatSession.handle 的桥接。
    """

    def __init__(self, engine: InferenceEngine, *, model_path: str = ""):
        self.engine = engine
        self.model_path = model_path
        self._sessions: Dict[str, ManagedSession] = {}
        self._lock = threading.Lock()

    def hello(self) -> dict:
        """进程级能力声明(§3.3 hello 指令 / §3.4 ready 事件共用)。

        T1:protocol_version 是协议冻结点(本次引入)。UI 据此判断兼容性。
        model 字段当前是字符串(单进程单模型);design.md §10.2 模型广场预留:
        将来多模型可能变成 models: [...] + active_model,现在留字符串口子
        近乎零成本,事后加是破坏性变更(protocol_version 会 bump)。
        """
        return {
            "protocol_version": PROTOCOL_VERSION,
            "version": __version__,
            "model": self.model_path,
            "capabilities": CAPABILITIES,
        }

    def new_session(self, params: dict) -> str:
        """构造 ChatSession,返回 session_id(§3.3 new_session)。

        params 字段:template / reasoning / think / state_path / gen_config(全可选)。
        """
        template = params.get("template", "qa")
        reasoning = bool(params.get("reasoning", False))
        think = params.get("think", "off")
        state_path = params.get("state_path")
        gen_config_params = params.get("gen_config") or {}

        self._validate_session_params(template, reasoning, think, gen_config_params)

        # gen_config:缺省走 ChatSession 的高创造力档,允许 serve 调用方覆盖部分字段。
        # T4:int/float 转换可能抛 ValueError(如 temperature="hot"),统一转
        # ProtocolError(bad_request),否则会被 handle_line 兜成 internal。
        try:
            base_config = GenerationConfig(
                max_tokens=int(gen_config_params.get("max_tokens", 300)),
                temperature=float(gen_config_params.get("temperature", 1.2)),
                top_p=float(gen_config_params.get("top_p", 0.5)),
                seed=int(gen_config_params.get("seed", 42)),
                presence_penalty=float(gen_config_params.get("presence_penalty", 0.4)),
                frequency_penalty=float(gen_config_params.get("frequency_penalty", 0.4)),
                penalty_decay=float(gen_config_params.get("penalty_decay", 0.996)),
            )
        except (ValueError, TypeError) as exc:
            raise ProtocolError(f"gen_config 字段类型错误: {exc}") from exc

        # state 加载(若提供路径)
        state = None
        state_label = None
        if state_path:
            from .inspection import validate_state_for_model
            # validate 需要 model 对象;engine 持有 model,这里取出来
            state = validate_state_for_model(Path(state_path), self.engine.model)
            state_label = state_path

        session = ChatSession(
            self.engine,
            config=base_config,
            template=template,
            reasoning=reasoning,
            think=think,
            state=state,
            state_label=state_label,
            ab=False,  # serve 不走 ChatSession.ab,多轮 A/B 推迟 v1.5
        )
        session_id = uuid.uuid4().hex[:16]
        managed = ManagedSession(
            session=session,
            session_id=session_id,
            template=template,
            reasoning=reasoning,
            think=think,
            state_path=state_path,
        )
        with self._lock:
            self._sessions[session_id] = managed
        return session_id

    @staticmethod
    def _validate_session_params(
        template: str, reasoning: bool, think: str, gen_config: dict
    ) -> None:
        if template not in ("qa", "instruction", "raw"):
            raise ProtocolError(f"template 只支持 qa / instruction / raw, 收到 {template!r}")
        if think not in ("off", "fast", "on"):
            raise ProtocolError(f"think 只支持 off / fast / on, 收到 {think!r}")
        if think != "off" and not reasoning:
            raise ProtocolError("think 仅在 reasoning=True 时合法")
        # gen_config 数值校验(复用 GenerationConfig.validate 的子集)
        mt = gen_config.get("max_tokens", 300)
        if not isinstance(mt, int) or mt <= 0:
            raise ProtocolError("gen_config.max_tokens 必须 > 0")

    def get_session(self, session_id: str) -> ManagedSession:
        with self._lock:
            managed = self._sessions.get(session_id)
        if managed is None:
            raise ProtocolError(f"session 不存在: {session_id}", code="not_found")
        return managed

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())

    def preview(self, params: dict) -> Tuple[List[dict], bool]:
        """一次性预览(§3.3 preview),不建 session。

        ab=False → 返回 (流式事件列表[可空], is_ab=False),result 在 turn_end 里。
        ab=True  → 返回 (两个 turn_end 事件, is_ab=True)。

        返回的事件列表不含 ok 终结(ServeProtocol 统一加)。
        """
        prompt = params.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ProtocolError("preview 需要 prompt 字符串")
        template = params.get("template", "raw")
        reasoning = bool(params.get("reasoning", False))
        think = params.get("think", "off")
        state_path = params.get("state_path")
        ab = bool(params.get("ab", False))
        gen_config_params = params.get("gen_config") or {}

        self._validate_session_params(template, reasoning, think, gen_config_params)

        if ab and not state_path:
            raise ProtocolError("ab=True 需要 state_path")

        # T4:gen_config 类型转换错误 → bad_request(而非 internal)
        try:
            cfg = with_template_stops(
                GenerationConfig(
                    max_tokens=int(gen_config_params.get("max_tokens", 80)),
                    temperature=float(gen_config_params.get("temperature", 0.0)),
                    top_p=float(gen_config_params.get("top_p", 0.9)),
                    seed=int(gen_config_params.get("seed", 42)),
                ),
                template,
            )
        except (ValueError, TypeError) as exc:
            raise ProtocolError(f"gen_config 字段类型错误: {exc}") from exc
        wrapped = render_prompt(prompt, template, reasoning=reasoning, think=think)
        state = str(state_path) if state_path else None

        # T1:preview 的 turn_end 也带 thinking/answer(think=on 时)。
        track_think = reasoning and think == "on" and template != "raw"

        def _turn_end_fields(result_dict: dict) -> dict:
            fields = {"result": result_dict}
            if track_think:
                from .thinking import split_thinking
                thinking, answer = split_thinking(result_dict.get("text", ""))
                fields["thinking"] = thinking
                fields["answer"] = answer.lstrip("\n")
            return fields

        events: List[dict] = []
        if ab:
            result = self.engine.compare(wrapped, state=state, config=cfg)
            events.append(_event("turn_end", side="with_state", **_turn_end_fields(result.with_state.to_dict())))
            events.append(_event("turn_end", side="baseline", **_turn_end_fields(result.baseline.to_dict())))
            return events, True
        # 单路:无 on_text 流式(preview 是一次性,不建 session);事件只含 turn_end
        result = self.engine.generate(wrapped, state=state, config=cfg)
        events.append(_event("turn_end", **_turn_end_fields(result.to_dict())))
        return events, False

    def detect_import(self, params: dict) -> dict:
        """探测数据格式 + 转换 + 渲染预览(不落盘,§4.1/§4.4 UI 确认步)。

        复用 engine 的 tokenizer 做 preview_records 渲染(token 级 mask 边界)。
        返回 detection + 前3条渲染样本,供 UI 在落盘前展示确认。
        探测失败(schema=unknown)不报错——detection 原样返回,UI 据此走手动映射。

        §4.1 流程的"探测/预览"步;落盘走 import_dataset 方法。
        """
        from .importer import (
            convert, detect_schema, detection_for_fields, preview_records, read_records,
        )

        data_path = params.get("data_path")
        if not isinstance(data_path, str) or not data_path.strip():
            raise ProtocolError("detect_import 需要 data_path 字符串")
        path = Path(data_path).expanduser()
        if not path.is_file():
            raise ProtocolError(f"数据文件不存在: {path}", code="not_found")

        items = read_records(path)
        detection = detect_schema(items)
        prompt_key = params.get("prompt_key")
        response_key = params.get("response_key")
        if (prompt_key is None) != (response_key is None):
            raise ProtocolError("prompt_key 与 response_key 必须同时提供")
        if prompt_key is not None:
            if not isinstance(prompt_key, str) or not isinstance(response_key, str):
                raise ProtocolError("prompt_key/response_key 必须是字符串")
            detection = detection_for_fields(items, prompt_key, response_key)
        turn_policy = params.get("turn_policy", "first")
        if turn_policy not in ("first", "all"):
            raise ProtocolError("turn_policy 只支持 first / all")
        ctx_len = params.get("ctx_len")
        if ctx_len is not None and (not isinstance(ctx_len, int) or ctx_len <= 0):
            raise ProtocolError("ctx_len 必须是正整数")

        # 探测失败时仍返回 detection(含 sample 原文),UI 走手动映射;
        # 转换只在非 unknown 时做(unknown convert 会抛错)。
        converted: List[dict] = []
        rendered: List[dict] = []
        result_dict: Optional[dict] = None
        if detection.schema != "unknown":
            result = convert(items, detection, turn_policy=turn_policy)
            converted = result.records
            result_dict = result.to_dict()
            # 渲染预览需要 tokenizer(engine 持有)
            rendered = [
                rs.to_dict()
                for rs in preview_records(
                    converted[:3], template=result.template,
                    tokenizer=self.engine.tokenizer, n=3, ctx_len=ctx_len,
                )
            ]
        return {
            "detection": detection.to_dict(),
            "result": result_dict,
            "preview": rendered,
        }

    def import_dataset(self, params: dict) -> dict:
        """探测 + 转换 + 落盘(§4.1 流程的落盘步,产物可直接喂 train)。

        与 detect_import 的区别:本方法落盘 jsonl + sidecar,返回 artifact 路径。
        UI 流程:detect_import(确认)→ import_dataset(落盘)。
        也可跳过确认直接 import(UI 一键导入场景)。
        """
        from .importer import _strip_sample_from_result_dict as _strip_sample
        from .importer import import_dataset as _do_import

        data_path = params.get("data_path")
        out_path = params.get("out_path")
        turn_policy = params.get("turn_policy", "first")
        if not isinstance(data_path, str) or not data_path.strip():
            raise ProtocolError("import 需要 data_path 字符串")
        if not isinstance(out_path, str) or not out_path.strip():
            raise ProtocolError("import 需要 out_path 字符串")
        if turn_policy not in ("first", "all"):
            raise ProtocolError(f"turn_policy 只支持 first / all, 收到 {turn_policy!r}")

        src = Path(data_path).expanduser()
        if not src.is_file():
            raise ProtocolError(f"数据文件不存在: {src}", code="not_found")

        try:
            prompt_key = params.get("prompt_key")
            response_key = params.get("response_key")
            if (prompt_key is None) != (response_key is None):
                raise ValueError("prompt_key 与 response_key 必须同时提供")
            artifact, result = _do_import(
                src, Path(out_path), turn_policy=turn_policy,
                prompt_key=prompt_key, response_key=response_key,
            )
        except ValueError as exc:
            # 探测失败(unknown)或转换错误 → bad_request(附原因供 UI 展示)
            raise ProtocolError(f"导入失败: {exc}") from exc
        return {
            "jsonl_path": str(artifact.jsonl_path),
            "sidecar_path": str(artifact.sidecar_path),
            "sha256": artifact.sha256,
            "record_count": artifact.record_count,
            "result": _strip_sample(result.to_dict()),
        }


# ── 协议主循环(读线程 + abort Event)─────────────────────────

class ServeProtocol:
    """stdin/stdout JSON lines 协议主循环。

    线程模型:
      - 读线程:持续 readline → Queue(主线程处理期间读线程不停)。
        读到 abort 指令 → set _abort_event(通知正在跑的 generate 中断)。
      - 主线程:从 Queue 取指令,handle_line 处理,emit 事件到 stdout。
      - busy 锁:同时只允许一个 in-flight send/preview。

    不变式:任何输入行都不能让进程崩溃(§3.5)。handle_line 全包 try/except。
    """

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        model_path: str = "",
        manager: Optional[ServeSessionManager] = None,
        stdin=None,
        stdout=None,
        stderr=None,
    ):
        self.manager = manager or ServeSessionManager(engine, model_path=model_path)
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._abort_event = threading.Event()
        self._busy = threading.Lock()
        self._in_flight_id: Optional[str] = None  # 当前在跑的请求 id(busy 检查用)
        self._shutdown = False
        # X1:_emit 跨线程无锁会撕裂 JSON 行。读线程(abort 的 ok)和主线程同时写
        # stdout 靠 GIL 撞运气,违反「stdout 只有完整 JSON 行」不变式。显式加锁。
        self._emit_lock = threading.Lock()

    # ── 输出 ────────────────────────────────────────────────────

    def _emit(self, event: dict) -> None:
        """写一个事件到 stdout(原子写一行 + flush)。

        X1:加锁保证一行完整 —— 读线程(abort ok)与主线程(turn_end/text_chunk)
        可能并发 emit,裸 write + flush 会撕裂行。
        """
        line = json.dumps(event, ensure_ascii=False)
        with self._emit_lock:
            self._stdout.write(line + "\n")
            self._stdout.flush()

    def _log(self, msg: str) -> None:
        """人类可读日志走 stderr(§3.2:stdout 只有 JSON)。"""
        self._stderr.write(msg + "\n")
        self._stderr.flush()

    # ── 主循环 ──────────────────────────────────────────────

    def run(self) -> None:
        """主循环:启动读线程,发 ready,循环处理指令直到 shutdown / EOF。"""
        reader = threading.Thread(target=self._read_loop, daemon=True)
        reader.start()

        # §3.4 ready:启动完成、模型加载完毕(无 id,进程级)
        self._emit(_event("ready", **self.manager.hello()))

        while not self._shutdown:
            try:
                line = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:  # 读线程 EOF 信号
                break
            self.handle_line(line)
            if self._shutdown:
                break

    def _read_loop(self) -> None:
        """读线程:持续 readline → Queue。EOF → None 信号。"""
        try:
            for line in self._stdin:
                # abort 特判:不进队列,直接 set event,立即生效(§3.3 abort)。
                # 其余指令进队列让主线程顺序处理。
                stripped = line.strip()
                if self._maybe_handle_abort_inline(stripped):
                    continue
                self._queue.put(stripped)
        except Exception as exc:  # 读线程不能崩
            self._log(f"# 读线程异常: {exc}")
        finally:
            self._queue.put(None)  # EOF 信号

    def _maybe_handle_abort_inline(self, line: str) -> bool:
        """abort 指令在读线程内联处理(不进队列),立即 set event。

        也要回 ok/error 终结,所以仍需 emit。返回 True 表示已处理(不进队列)。
        """
        if not line:
            return False
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            return False  # 非 JSON 不在此处理,让主线程报 bad_request
        if not isinstance(req, dict) or req.get("cmd") != "abort":
            return False
        # abort 指令:通知在跑的 generate 中断
        id_ = req.get("id")
        if self._abort_event.is_set():
            # 已有 abort 在飞,重复 abort 直接回 ok(幂等)
            self._emit(_ok(id_))
            return True
        if self._in_flight_id is None:
            # 没有在跑的生成,abort 无的放矢,仍回 ok
            self._emit(_ok(id_))
            return True
        self._abort_event.set()
        # 被中断请求的 error{aborted} 由主线程在 handle_send 捕获时发出;
        # 这里只给 abort 自己回 ok。
        self._emit(_ok(id_))
        return True

    # ── 单行处理(主线程)────────────────────────────────────

    def handle_line(self, line: str) -> None:
        """处理一行输入 → emit 事件(含终结)。

        任何异常都转成 error 终结事件,进程不崩(§3.5 不变式)。
        """
        # JSON 解析
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            self._emit(_error("bad_request", f"JSON 解析失败: {exc.msg}"))
            return
        if not isinstance(req, dict):
            self._emit(_error("bad_request", "请求必须是 JSON 对象"))
            return

        cmd = req.get("cmd")
        id_ = req.get("id")
        params = {k: v for k, v in req.items() if k not in ("cmd", "id")}

        if not isinstance(cmd, str) or not cmd:
            self._emit(_error("bad_request", "缺少 cmd 字段", id_=id_))
            return

        try:
            self._dispatch(cmd, id_, params)
        except ProtocolError as exc:
            # 直接用 exc.code(Spec §3.5 五错误码之一,由 raise 处显式指定)。
            # 废弃旧实现的 message 字符串匹配(_classify_protocol_error,附录 E.4 裁决)。
            self._emit(_error(exc.code, str(exc), id_=id_))
        except GenerationAborted:
            # send/preview 被中断:已被 handle_send 处理,不会到这里
            self._emit(_error("aborted", "生成已中断", id_=id_))
        except Exception as exc:  # internal:未预期异常,附 traceback 摘要(§3.5)
            tb = traceback.format_exc()
            summary = tb[-500:] if len(tb) > 500 else tb
            self._emit(_error("internal", f"{exc}\n{summary}", id_=id_))
            self._log(f"# internal error on cmd={cmd}: {exc}\n{tb}")

    def _dispatch(self, cmd: str, id_: Optional[str], params: dict) -> None:
        """指令路由(§3.3)。每个分支保证发一个终结事件。"""
        if cmd == "hello":
            self._emit(_ok(id_, **self.manager.hello()))
            return
        if cmd == "new_session":
            session_id = self.manager.new_session(params)
            self._emit(_ok(id_, session_id=session_id))
            return
        if cmd == "close_session":
            sid = params.get("session_id")
            if not isinstance(sid, str):
                raise ProtocolError("close_session 需要 session_id")
            self.manager.close_session(sid)
            self._emit(_ok(id_))
            return
        if cmd == "shutdown":
            self._emit(_ok(id_))
            self._shutdown = True
            return
        if cmd == "abort":
            # abort 正常在读线程内联处理(_maybe_handle_abort_inline,立即生效)。
            # 这里是同步兜底路径(测试直接调 handle_line / 无在跑生成时的幂等回 ok)。
            # 注意(P3/X2):走到这里不能断言"无在跑" —— 读线程可能在主线程把 send
            # 取出队列之前就内联处理了 abort,此时 _in_flight_id 可能已设、event 未 set。
            # 当前实现幂等回 ok,真正的竞态修复见 X2(本 commit 仅订正注释)。
            self._emit(_ok(id_))
            return
        if cmd == "detect_import":
            # §4.1 探测/预览步(不落盘):CPU/IO 操作,不走 busy 锁(无生成)。
            payload = self.manager.detect_import(params)
            self._emit(_ok(id_, **payload))
            return
        if cmd == "import":
            # §4.1 落盘步:产物 jsonl + sidecar,可直接喂 train。不走 busy 锁。
            payload = self.manager.import_dataset(params)
            self._emit(_ok(id_, **payload))
            return
        if cmd == "preview":
            self._handle_preview(id_, params)
            return
        if cmd == "send":
            self._handle_send(id_, params)
            return
        if cmd in ("set_state", "set_config", "rewind", "reset"):
            self._handle_session_cmd(cmd, id_, params)
            return
        raise ProtocolError(f"未知指令: {cmd!r}")

    # ── send(流式 + abort)─────────────────────────────────

    def _handle_send(self, id_: Optional[str], params: dict) -> None:
        """send {session_id, text} → text_chunk* → turn_end → ok(§3.3)。

        busy 锁:同时只允许一个 in-flight send。
        abort:_abort_event 传给 generate 的 should_abort,中断则发 error{aborted}。

        T1 协议字段:
          - text_chunk 带 phase: "think"|"answer"(仅 reasoning+think=on 有意义;
            其他档位恒 "answer")。UI 据此 dim/正常渲染增量,不用自己再写拆分。
          - turn_end 带 thinking/answer(从 result.text 用 thinking.py 拆出,
            单一事实源)。非 think=on 时两者都为 None,text 仍是原始全文。
        """
        sid = params.get("session_id")
        text = params.get("text")
        if not isinstance(sid, str):
            raise ProtocolError("send 需要 session_id")
        if not isinstance(text, str) or not text.strip():
            raise ProtocolError("send 需要非空 text")

        managed = self.manager.get_session(sid)  # 可能 raise not_found

        if not self._busy.acquire(blocking=False):
            raise ProtocolError("busy: 已有生成进行中", code="busy")
        self._in_flight_id = id_
        self._abort_event.clear()
        try:
            session = managed.session
            # §3.3 abort:把 _abort_event.is_set 绑定到 session,generate 每步检查。
            # 读线程读 abort 指令 → set event → generate 下一步抛 GenerationAborted。
            session.abort_checker = self._abort_event.is_set

            # think=on 才需要 phase 跟踪(其他档位恒 answer)。
            track_phase = (
                session.reasoning and session.think == "on"
                and session.template != "raw"
            )
            # 累积文本用于 phase 分类(thinking.py.classify_phase);
            # 非流式时 on_text=None,phase 跟踪仍可用于 turn_end 拆分(基于 result.text)。
            accum = [""]

            def on_text(delta: str) -> None:
                if track_phase:
                    accum[0] += delta
                    from .thinking import classify_phase
                    phase = classify_phase(accum[0])
                else:
                    phase = "answer"
                self._emit(_event(
                    "text_chunk", id_=id_, session_id=sid,
                    delta=delta, phase=phase,
                ))

            try:
                reply = session.handle(text, on_text=on_text)
            except GenerationAborted:
                # §3.5 aborted:被中断,发 error 终结。
                # P0-2:ChatSession._handle_single 已在 GenerationAborted 时
                # 置 cache=None + cache_clean=False(续传路径的 self.cache 可能
                # 已被前向就地污染),下轮强制重放。旧注释"cache 自动重建无需修补"
                # 是错的(重放只在 cache_clean=False 时触发,abort 路径原先没置 False)。
                self._emit(_error("aborted", "生成已中断", id_=id_))
                return

            # turn_end:result = GenerationResult.to_dict()(§3.4)
            result = reply.payload or {}
            turn_end_fields: dict = {"result": result}
            # T1:think=on 拆出 thinking/answer 顶层字段(供 UI 直接消费,
            # 不用再从 result.text 自己拆)。非 think=on 不加(保持 None/缺省)。
            if track_phase:
                from .thinking import split_thinking
                full_text = result.get("text", "")
                thinking, answer = split_thinking(full_text)
                turn_end_fields["thinking"] = thinking
                turn_end_fields["answer"] = answer.lstrip("\n")
            self._emit(_event(
                "turn_end", id_=id_, session_id=sid, **turn_end_fields,
            ))
            self._emit(_ok(id_))
        finally:
            self._in_flight_id = None
            self._abort_event.clear()
            self._busy.release()

    def _handle_preview(self, id_: Optional[str], params: dict) -> None:
        """preview(§3.3):一次性,不建 session,内部即建即弃 cache。

        ab=False → turn_end → ok
        ab=True  → turn_end{side:with_state} → turn_end{side:baseline} → ok
        """
        if not self._busy.acquire(blocking=False):
            raise ProtocolError("busy: 已有生成进行中", code="busy")
        self._in_flight_id = id_
        self._abort_event.clear()
        try:
            events, _is_ab = self.manager.preview(params)
            for evt in events:
                evt["id"] = id_  # preview 事件带回 id(§3.2)
                self._emit(evt)
            self._emit(_ok(id_))
        except GenerationAborted:
            self._emit(_error("aborted", "生成已中断", id_=id_))
        finally:
            self._in_flight_id = None
            self._abort_event.clear()
            self._busy.release()

    # ── 会话控制指令(set_state / set_config / rewind / reset)──

    def _handle_session_cmd(self, cmd: str, id_: Optional[str], params: dict) -> None:
        sid = params.get("session_id")
        if not isinstance(sid, str):
            raise ProtocolError(f"{cmd} 需要 session_id")
        managed = self.manager.get_session(sid)
        session = managed.session

        if cmd == "set_state":
            # T3:走 ChatSession.set_state public API,协议 payload 取结构化字段,
            # 不再从 reply.lines[0] 捞中文人话(改 CLI 文案协议会跟着变)。
            state_path = params.get("state_path")  # None = 关闭 state
            if state_path is not None and not isinstance(state_path, str):
                raise ProtocolError("set_state 的 state_path 必须是字符串或 null")
            result = session.set_state(state_path)
            if not result.ok:
                # 加载失败 → bad_request(附 message 供 UI 展示)
                raise ProtocolError(result.message)
            managed.state_path = result.state_label
            self._emit(_ok(
                id_,
                state_label=result.state_label,
                history_cleared=result.history_cleared,
                message=result.message,
            ))
            return

        if cmd == "set_config":
            gen_config = params.get("gen_config")
            if not isinstance(gen_config, dict):
                raise ProtocolError("set_config 需要 gen_config 对象")
            self._apply_gen_config(session, gen_config)
            self._emit(_ok(id_))
            return

        if cmd == "rewind":
            n = params.get("n", 1)
            if not isinstance(n, int) or n < 1:
                raise ProtocolError("rewind 的 n 必须是 >= 1 的整数")
            result = session.rewind(n)
            self._emit(_ok(
                id_,
                rounds_removed=result.rounds_removed,
                history_len=result.history_len,
                message=result.message,
            ))
            return

        if cmd == "reset":
            session.reset()
            self._emit(_ok(id_))
            return

    @staticmethod
    def _apply_gen_config(session: ChatSession, gen_config: dict) -> None:
        """把 gen_config 的字段应用到 session.config(下一轮生效,§3.3 set_config)。

        支持字段:max_tokens / temperature / top_p / seed /
                  presence_penalty / frequency_penalty / penalty_decay。
        用 dataclasses.replace 重建 frozen config。

        T4:类型/取值错误 → ProtocolError(bad_request),而非 internal。
        UI 对两者处理完全不同(internal=报 bug;bad_request=输入框画红边)。
        """
        from dataclasses import replace
        allowed = {
            "max_tokens", "temperature", "top_p", "seed",
            "presence_penalty", "frequency_penalty", "penalty_decay",
        }
        updates = {k: v for k, v in gen_config.items() if k in allowed}
        if not updates:
            return
        try:
            new_config = replace(session.config, **updates)
            new_config.validate()
        except (ValueError, TypeError) as exc:
            raise ProtocolError(f"gen_config 参数非法: {exc}") from exc
        session.config = new_config


# ── 进程入口辅助 ────────────────────────────────────────────

def run_serve(model_path: str, *, cache_limit_spec: Optional[str] = None) -> None:
    """CLI serve 命令的进程入口。

    cache_limit_spec 必须在 load_model 前生效(时序铁律,见 AGENTS.md 内存事实)。
    T2:不再从 cli 导 _apply_cache_limit(会把 typer 拖进 sidecar 进程);
    改用 runtime.apply_cache_limit,错误走 ProtocolError/ValueError。
    """
    from .core import load_model
    from .runtime import apply_cache_limit

    try:
        apply_cache_limit(cache_limit_spec)
    except ValueError as exc:
        # serve 进程启动期:错误发到 stderr 后退出(协议尚未就绪,无法发 error 事件)
        sys.stderr.write(f"# cache-limit 错误: {exc}\n")
        sys.exit(2)
    model, tok = load_model(model_path, patch=False)
    engine = InferenceEngine(model, tok)
    ServeProtocol(engine, model_path=model_path).run()
