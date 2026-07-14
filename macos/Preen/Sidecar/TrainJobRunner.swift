//
//  TrainJobRunner.swift
//  Preen
//
//  训练子进程管理:`Process` spawn `statetuner train`,stdout JSON lines
//  → JSONDecoder → TrainEvent,逐行 async 推到 AsyncStream。
//
//  契约(已核对 cli.py + events.py):
//   - train 的 stdout = 结构化事件流(JSON lines);stderr = 人类日志(# 开头)。
//   - 取消 = SIGINT(process.interrupt())→ 期望收到 cancelled 事件后进程退出。
//   - events-file 与 stdout 内容相同(同一事件双写);Swift 只读 stdout。
//
//  并发风格(语言模式 5):
//   - 没有用 actor。后台读循环用 Task;UI Store 在 MainActor 上 consume 事件。
//   - Process / FileHandle 不跨 isolation 边界共享 mutable 状态(读循环独占 pipe 读端)。
//

import Foundation

/// 训练子进程的封装。一次性 —— 跑完即弃。
///
/// 用法:
/// ```
/// let runner = TrainJobRunner(model: ..., data: ..., config: ...)
/// let stream = runner.start()  // 返回 AsyncStream<TrainEvent>
/// for await event in stream { store.consume(event: event) }
/// // 取消:
/// runner.cancel()
/// ```
final class TrainJobRunner {

    private let process = Process()
    private var readTask: Task<Void, Never>?
    private var continuation: AsyncStream<TrainEvent>.Continuation?

    /// stderr 累积的人类日志(诊断用)。原子访问。
    private let stderrLock = NSLock()
    private var _stderrLog = ""
    private let stateLock = NSLock()
    private var _pid: Int32?
    private var _exitInfo: ProcessExitInfo?
    private var stderrFileHandle: FileHandle?

    var onStderr: ((String) -> Void)?
    var onExit: ((ProcessExitInfo) -> Void)?

    init() {}

    /// 启动训练进程,返回事件流。
    /// - Parameters 在 Models/TrainingConfig 里组装;此处接收 argv + cwd。
    func start(argv: [String], currentDirectory: URL?, stderrFile: URL? = nil) -> AsyncStream<TrainEvent> {
        process.executableURL = PythonResolver.executable
        process.arguments = argv
        process.environment = PythonResolver.childEnvironment
        if let cwd = currentDirectory {
            process.currentDirectoryURL = cwd
        }

        // stdout / stderr pipe。
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        if let stderrFile {
            FileManager.default.createFile(atPath: stderrFile.path, contents: nil)
            stderrFileHandle = try? FileHandle(forWritingTo: stderrFile)
        }

        // stderr 后台读 → 累积到 _stderrLog。
        readStderrInBackground(handle: stderrPipe.fileHandleForReading)

        // AsyncStream:从 stdout 逐行解码事件。
        let (stream, continuation) = AsyncStream<TrainEvent>.makeStream()
        self.continuation = continuation

        do {
            try process.run()
            stateLock.lock()
            _pid = process.processIdentifier
            stateLock.unlock()
        } catch {
            // 启动失败:推一个合成 failed 事件,让 UI 能感知。
            let message = "无法启动训练进程：\(error.localizedDescription)"
            appendStderr(message + "\n")
            continuation.yield(.failed(message: message, path: nil, timestamp: Date().timeIntervalSince1970))
            continuation.finish()
            return stream
        }

        // 后台读 stdout → 解码 → yield。
        readTask = Task { [weak self] in
            await self?.readStdoutLines(handle: stdoutPipe.fileHandleForReading, continuation: continuation)
        }

        return stream
    }

    /// SIGINT 取消。不阻塞 —— 进程退出由读循环感知并 finish stream。
    func cancel() {
        guard process.isRunning else { return }
        process.interrupt()  // SIGINT
    }

    /// 不阻塞 MainActor 地等待训练进程完成 SIGINT 收尾。
    func waitUntilExit() async {
        while process.isRunning {
            try? await Task.sleep(for: .milliseconds(50))
        }
    }

    /// 进程是否仍在跑。
    var isRunning: Bool {
        process.isRunning
    }

    var pid: Int32? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return _pid
    }

    var exitInfo: ProcessExitInfo? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return _exitInfo
    }

    /// 取 stderr 累积(诊断失败时用)。
    var stderrLog: String {
        stderrLock.lock()
        defer { stderrLock.unlock() }
        return _stderrLog
    }

    // MARK: - 内部

    private func readStdoutLines(handle: FileHandle, continuation: AsyncStream<TrainEvent>.Continuation) async {
        // FileHandle.bytes.lines:逐行 async,自动处理跨包边界。
        // 进程关闭 stdout 时迭代结束。
        do {
            // bytes 是 AsyncBytes;lines 是 AsyncLineSequence。
            for try await line in handle.bytes.lines {
                guard !line.isEmpty else { continue }
                guard let data = line.data(using: .utf8) else { continue }
                do {
                    let event = try JSONDecoder().decode(TrainEvent.self, from: data)
                    continuation.yield(event)
                } catch {
                    // 解码失败:发一个合成 unknown 事件,UI 不会因一行坏数据卡死。
                    // (events.py 不变量:任何输入行不让进程崩;我们镜像此精神到客户端。)
                    #if DEBUG
                    print("[TrainJobRunner] 解码失败 line=\(line.prefix(200)) error=\(error)")
                    #endif
                    continuation.yield(.unknown(type: "(decode-error)", timestamp: Date().timeIntervalSince1970,
                                                payload: ["raw": String(line.prefix(2000))]))
                }
            }
        } catch {
            // IO 错误(极少);进程多半已死,让 stream finish。
            #if DEBUG
            print("[TrainJobRunner] stdout 读异常:\(error)")
            #endif
        }

        // 进程已退出 + stdout drain 完。等进程完全收尾,然后 finish stream。
        process.waitUntilExit()
        let info = ProcessExitInfo(
            status: process.terminationStatus,
            reason: process.terminationReason == .exit ? .exit : .uncaughtSignal
        )
        stateLock.withLock { _exitInfo = info }
        onExit?(info)
        try? stderrFileHandle?.close()
        stderrFileHandle = nil
        continuation.finish()
    }

    private func readStderrInBackground(handle: FileHandle) {
        // 用 readabilityHandler 累积 stderr(不走 async,避免和 stdout 读循环耦合)。
        handle.readabilityHandler = { [weak self] fh in
            let data = fh.availableData
            guard !data.isEmpty else { return }
            guard let str = String(data: data, encoding: .utf8) else { return }
            self?.appendStderr(str)
            // 进程关闭后 readabilityHandler 会拿到空 data,我们借机清空 handler。
            // (此处简化:不主动清,readStdoutLines 的 waitUntilExit 会让进程归零。)
        }
    }

    private func appendStderr(_ s: String) {
        stderrLock.lock()
        _stderrLog += s
        // 防止无限增长(诊断用,留最后 32KB)。
        if _stderrLog.count > 32 * 1024 {
            _stderrLog = String(_stderrLog.suffix(32 * 1024))
        }
        if let data = s.data(using: .utf8) {
            try? stderrFileHandle?.write(contentsOf: data)
        }
        stderrLock.unlock()
        onStderr?(s)
    }
}
