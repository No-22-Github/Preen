//
//  ServeClient.swift
//  Preen
//
//  serve 常驻推理进程客户端:spawn `statetuner serve`,stdin 写指令 / stdout 读事件。
//
//  核心机制(已核对 serve.py):
//   1. **ready 握手**:启动后 stdout 首行必是 ready(无 id,进程级);
//      收到前任何 send 都 throw。UI 等到 ready 才解锁交互。
//   2. **id → CheckedContinuation 表**:每个请求一个 id,对应一个 continuation;
//      ok/error 终结事件 resume 对应 continuation;流式事件(text_chunk/turn_end)
//      通过独立 AsyncStream<ServeEvent> 广播,Store 订阅。
//   3. **abort 双通道(绝不耦合)**:
//      - sendAbort() 发自己的 id,自己的 continuation → 期望 ok(读线程内联,几乎立即)
//      - 被 abort 的原 send(不同 id)的 continuation 收到 error{aborted} → throw ServeError.aborted
//      - 两者的 id/continuation 完全独立,绝不共享状态。
//   4. **busy 是请求级互斥**:send/preview 同时只允许一个 in-flight;
//      并发第二个立即收 error{busy} → throw ServeError.busy(不隐式排队,UI 据此禁用)。
//
//  并发风格(语言模式 5):
//   - NSLock 保护 id→continuation 表、ready 状态、id 自增。
//   - stdout 读循环用 Task;不跨 isolation 边界共享 mutable Process/FileHandle。
//   - Continuation 是 Sendable-checked(ServeResponse 是值类型)。
//

import Foundation

/// serve 请求的返回(对应终结事件 ok / error)。
enum ServeResponse: Sendable {
    case ok(OkPayload)
    case error(code: ServeErrorCode, message: String)
}

/// ServeClient 抛出的错误。
enum ServeError: Error, Sendable {
    /// 进程未就绪(还没收到 ready)就发指令。
    case notReady
    /// busy:已有 in-flight 生成(serve.py 请求级互斥)。
    case busy
    /// 请求被 abort 中断(原 send 的视角)。
    case aborted
    /// serve 返回的 bad_request / not_found / internal 等。
    case serveError(code: ServeErrorCode, message: String)
    /// 进程意外退出(管道关闭)。
    case processExited
    /// 编码/写 stdin 失败。
    case ioError(String)
}

/// serve 客户端。
final class ServeClient {

    private let process = Process()
    private var stdinHandle: FileHandle?
    private var readTask: Task<Void, Never>?

    /// 状态锁:保护 ready / nextId / continuations / eventsContinuation。
    private let lock = NSLock()
    private var _isReady = false
    private var _nextId: UInt64 = 1
    /// id → continuation。请求-响应配对的核心。
    private var continuations: [String: CheckedContinuation<ServeResponse, Never>] = [:]
    /// 流式事件广播(ready / text_chunk / turn_end / 任意 error 也广播一份给 UI 展示)。
    private var eventsContinuation: AsyncStream<ServeEvent>.Continuation?

    /// stderr 累积(诊断用)。
    private let stderrLock = NSLock()
    private var _stderrLog = ""

    /// stderr 增量回调:每收到一段后端输出就触发(供 UI 实时展示启动日志)。
    /// 在后台线程(readabilityHandler 队列)触发,调用方自行切线程。
    var onStderr: ((String) -> Void)?

    /// stdout 增量回调:每读到一行 JSON 事件就触发原始行(诊断启动期 ready 是否到达)。
    /// 在读循环 Task 线程触发。调用方通常只在启动期(未 ready)关心它。
    var onStdout: ((String) -> Void)?

    init() {}

    // MARK: - 生命周期

    /// 启动 serve 进程。
    /// - Parameters:
    ///   - model: HF 模型目录。
    ///   - onReady: 收到 ready 事件时回调(主线程)。
    /// - Returns: 流式事件流(ready / text_chunk / turn_end;error 也会广播)。
    func start(model: URL, cacheLimitGb: String = "auto") -> AsyncStream<ServeEvent> {
        let argv = [
            "-m", "statetuner.cli", "serve",
            "--model", model.path,
            "--cache-limit-gb", cacheLimitGb,
        ]
        process.executableURL = PythonResolver.executable
        process.arguments = argv
        process.environment = PythonResolver.childEnvironment

        // 启动诊断:把 app 实际用的 python / PYTHONPATH / argv 推到启动日志窗口。
        // 最高频根因是 PREEN_SIDECAR_PYTHON 没进子进程环境 → fallback 到
        // /usr/bin/python3(系统自带,无 mlx 依赖,import 即崩)。这里一眼能看出来。
        let pythonPath = PythonResolver.repoRoot?
            .appendingPathComponent("src").path ?? "(无 — 环境变量缺失,可能用错解释器)"
        let diag = """
        # [Preen] sidecar python: \(PythonResolver.executable.path)
        # [Preen] PYTHONPATH: \(pythonPath)
        # [Preen] HF_HOME: \(PythonResolver.hfCache.path)
        # [Preen] argv: \(argv.joined(separator: " "))

        """
        onStderr?(diag)

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        let stdinPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        process.standardInput = stdinPipe
        stdinHandle = stdinPipe.fileHandleForWriting

        readStderrInBackground(handle: stderrPipe.fileHandleForReading)

        let (stream, continuation) = AsyncStream<ServeEvent>.makeStream()
        lock.lock()
        eventsContinuation = continuation
        lock.unlock()

        do {
            try process.run()
        } catch {
            // 启动失败:用合成 error 事件通知 UI。
            continuation.yield(.error(id: nil, code: .internal,
                                      message: "无法启动 serve 进程:\(error.localizedDescription)"))
            continuation.finish()
            return stream
        }

        // 后台读 stdout → 解码事件 → 分发。
        readTask = Task { [weak self] in
            await self?.readStdoutLoop(handle: stdoutPipe.fileHandleForReading, continuation: continuation)
        }

        return stream
    }

    /// 优雅停机(发 shutdown 指令,进程自然退出)。
    func shutdown() async {
        guard process.isRunning else { return }
        _ = try? await send(.shutdown(id: newId()))
    }

    /// 强制终止(SIGTERM,用于崩溃恢复或 app 退出)。
    func terminate() {
        if process.isRunning {
            process.terminate()
        }
    }

    /// 进程是否仍在跑。
    var isRunning: Bool {
        process.isRunning
    }

    /// 是否已收到 ready(可发指令)。
    var isReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _isReady
    }

    // MARK: - 指令 API(async,等终结事件)

    /// 发送任意请求,等终结事件返回。
    func send(_ request: ServeRequest) async throws -> ServeResponse {
        // ready 前不许发(hello 例外 —— 但本客户端不主动发 hello,靠 ready 事件拿 payload)。
        if request.cmd != "hello" && !isReady {
            throw ServeError.notReady
        }
        return try await withCheckedContinuation { (cont: CheckedContinuation<ServeResponse, Never>) in
            registerContinuation(id: request.id, cont: cont)
        do {
            let line = try request.encodeToLine()
            try writeStdin(line)
        } catch {
            // 写失败:立刻 resume(不让 awaiter 永远挂)。
            _ = unregisterContinuation(id: request.id)
            cont.resume(returning: .error(code: .internal, message: "写 stdin 失败:\(error.localizedDescription)"))
        }
        }.serving()  // 把 .error 转成 throw(详见 ServeResponse.serving)
    }

    /// 便捷:new_session。
    func newSession(template: String? = "qa",
                    reasoning: Bool? = nil,
                    think: String? = nil,
                    statePath: String? = nil,
                    genConfig: GenConfigDTO? = nil) async throws -> String {
        let resp = try await send(.newSession(id: newId(), template: template, reasoning: reasoning,
                                              think: think, statePath: statePath, genConfig: genConfig))
        guard case .ok(let payload) = resp, let sid = payload.sessionId else {
            throw ServeError.serveError(code: .internal, message: "new_session 未返回 session_id")
        }
        return sid
    }

    /// 便捷:send(文本消息)。终结事件由流式事件消费方处理,本方法只等 ok/error。
    func send(sessionId: String, text: String) async throws {
        let resp = try await send(.send(id: newId(), sessionId: sessionId, text: text))
        if case .error(let code, let msg) = resp {
            throw ServeError.serveError(code: code, message: msg)
        }
    }

    /// 便捷:abort。**独立通道**:用自己的 id,不耦合被中断的 send。
    /// abort 的语义:发 → 立即收 ok(读线程内联);被中断的原 send(另一个 await)收 error{aborted}。
    func abort() async throws {
        let resp = try await send(.abort(id: newId()))
        if case .error(let code, let msg) = resp {
            throw ServeError.serveError(code: code, message: msg)
        }
    }

    // MARK: - 内部

    /// 后台 stdout 读循环:解码 → 分发。
    private func readStdoutLoop(handle: FileHandle,
                                continuation: AsyncStream<ServeEvent>.Continuation) async {
        do {
            for try await line in handle.bytes.lines {
                guard !line.isEmpty else { continue }
                guard let data = line.data(using: .utf8) else { continue }
                // 诊断镜像:启动期把 stdout 原始行也推给 UI,看 ready 是否到达 Swift 端。
                if let raw = String(data: data, encoding: .utf8) {
                    onStdout?(raw)
                }
                let event: ServeEvent
                do {
                    event = try JSONDecoder().decode(ServeEvent.self, from: data)
                } catch {
                    #if DEBUG
                    print("[ServeClient] 解码失败 line=\(line.prefix(200)) error=\(error)")
                    #endif
                    continue  // 丢一行坏数据(serve 不变量:不让坏行影响后续)
                }
                handleEvent(event, broadcast: continuation)
            }
        } catch {
            #if DEBUG
            print("[ServeClient] stdout 读异常:\(error)")
            #endif
        }

        // stdout 关闭 = 进程退出。所有挂起的 continuation resume processExited。
        drainPendingContinuations()
        continuation.finish()
    }

    /// 处理一个事件:广播 + 终结事件 resume continuation。
    private func handleEvent(_ event: ServeEvent, broadcast: AsyncStream<ServeEvent>.Continuation) {
        // ready 特殊:进程级,不对应任何 continuation。
        if case .ready(let payload) = event {
            lock.lock()
            _isReady = true
            lock.unlock()
            // 把 ready 也广播出去(UI 监听 events 流来切「可交互」状态)。
            broadcast.yield(event)
            // ready 的 protocolVersion 校验(后续可扩展为版本协商)。
            #if DEBUG
            print("[ServeClient] ready: protocol=\(payload.protocolVersion) model=\(payload.model)")
            #endif
            return
        }

        // text_chunk / turn_end:流式,只广播(resume 由后续终结事件负责)。
        switch event {
        case .textChunk, .turnEnd:
            broadcast.yield(event)
            return
        default:
            break
        }

        // ok / error:终结事件。先广播(让 UI 同步看到状态),再 resume continuation。
        broadcast.yield(event)

        guard let id = event.requestId else { return }
        guard let cont = unregisterContinuation(id: id) else {
            #if DEBUG
            print("[ServeClient] 收到终结事件但无挂起 continuation: id=\(id)")
            #endif
            return
        }

        switch event {
        case .ok(_, let payload):
            cont.resume(returning: .ok(payload))
        case .error(_, let code, let message):
            cont.resume(returning: .error(code: code, message: message))
        default:
            break
        }
    }

    // MARK: - continuation 表管理(锁内)

    private func newId() -> String {
        lock.lock()
        defer { lock.unlock() }
        let id = _nextId
        _nextId += 1
        return "c\(id)"
    }

    private func registerContinuation(id: String, cont: CheckedContinuation<ServeResponse, Never>) {
        lock.lock()
        continuations[id] = cont
        lock.unlock()
    }

    private func unregisterContinuation(id: String) -> CheckedContinuation<ServeResponse, Never>? {
        lock.lock()
        defer { lock.unlock() }
        return continuations.removeValue(forKey: id)
    }

    /// 进程退出时:所有挂起 continuation 收到 processExited。
    private func drainPendingContinuations() {
        lock.lock()
        let pending = continuations
        continuations.removeAll()
        lock.unlock()
        for (_, cont) in pending {
            cont.resume(returning: .error(code: .internal, message: "serve 进程已退出"))
        }
    }

    // MARK: - stdin / stderr

    private func writeStdin(_ line: String) throws {
        guard let handle = stdinHandle else { throw ServeError.ioError("stdin 未就绪") }
        guard let data = line.data(using: .utf8) else { throw ServeError.ioError("编码失败") }
        try handle.write(contentsOf: data)
    }

    private func readStderrInBackground(handle: FileHandle) {
        handle.readabilityHandler = { [weak self] fh in
            let data = fh.availableData
            guard !data.isEmpty else { return }
            guard let str = String(data: data, encoding: .utf8) else { return }
            self?.appendStderr(str)
        }
    }

    private func appendStderr(_ s: String) {
        stderrLock.lock()
        _stderrLog += s
        if _stderrLog.count > 32 * 1024 {
            _stderrLog = String(_stderrLog.suffix(32 * 1024))
        }
        stderrLock.unlock()
        // 实时推给 UI(启动日志窗口)。
        onStderr?(s)
    }

    /// 取 stderr 累积(诊断失败时用)。
    var stderrLog: String {
        stderrLock.lock()
        defer { stderrLock.unlock() }
        return _stderrLog
    }
}

// MARK: - ServeResponse 便利

extension ServeResponse {
    /// 把 .error 转成 throw(让 awaiter 用 do/catch 处理);
    /// .ok 原样返回 payload。
    func serving() throws -> ServeResponse {
        switch self {
        case .ok: return self
        case .error(let code, let message):
            switch code {
            case .busy: throw ServeError.busy
            case .aborted: throw ServeError.aborted
            default: throw ServeError.serveError(code: code, message: message)
            }
        }
    }
}
