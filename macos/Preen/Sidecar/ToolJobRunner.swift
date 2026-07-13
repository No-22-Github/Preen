import Foundation

/// 一次性离线工具子进程。stdout 只接受 ToolEvent JSON Lines，stderr 留作诊断。
final class ToolJobRunner {
    private let process = Process()
    private var readTask: Task<Void, Never>?
    private let logLock = NSLock()
    private var _stderrLog = ""

    var stderrLog: String {
        logLock.withLock { _stderrLog }
    }

    var isRunning: Bool { process.isRunning }

    deinit {
        if process.isRunning { process.terminate() }
    }

    func start(argv: [String], currentDirectory: URL?) -> AsyncStream<ToolEvent> {
        process.executableURL = PythonResolver.executable
        process.arguments = argv
        process.environment = PythonResolver.childEnvironment
        process.currentDirectoryURL = currentDirectory

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        readStderr(handle: stderr.fileHandleForReading)

        let (stream, continuation) = AsyncStream<ToolEvent>.makeStream()
        do {
            try process.run()
        } catch {
            continuation.yield(ToolEvent(
                type: .failed, tool: "process", timestamp: Date().timeIntervalSince1970,
                phase: nil, message: "无法启动工具进程：\(error.localizedDescription)",
                current: nil, total: nil, progress: nil, path: nil, result: nil
            ))
            continuation.finish()
            return stream
        }

        readTask = Task { [weak self] in
            guard let self else { return }
            var sawTerminal = false
            do {
                for try await line in stdout.fileHandleForReading.bytes.lines {
                    guard let data = line.data(using: .utf8), !line.isEmpty else { continue }
                    do {
                        let event = try JSONDecoder().decode(ToolEvent.self, from: data)
                        if [.completed, .failed, .cancelled].contains(event.type) {
                            sawTerminal = true
                        }
                        continuation.yield(event)
                    } catch {
                        self.appendLog("无法解析工具事件：\(line.prefix(500))\n")
                    }
                }
            } catch {
                self.appendLog("读取工具输出失败：\(error.localizedDescription)\n")
            }
            self.process.waitUntilExit()
            if !sawTerminal {
                let message = self.stderrLog.isEmpty
                    ? "工具进程异常退出（code \(self.process.terminationStatus)）"
                    : self.stderrLog
                continuation.yield(ToolEvent(
                    type: self.process.terminationReason == .uncaughtSignal ? .cancelled : .failed,
                    tool: "process", timestamp: Date().timeIntervalSince1970,
                    phase: nil, message: message, current: nil, total: nil,
                    progress: nil, path: nil, result: nil
                ))
            }
            continuation.finish()
        }
        return stream
    }

    func cancel() {
        guard process.isRunning else { return }
        process.interrupt()
    }

    private func readStderr(handle: FileHandle) {
        handle.readabilityHandler = { [weak self] file in
            let data = file.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            self?.appendLog(text)
        }
    }

    private func appendLog(_ text: String) {
        logLock.withLock {
            _stderrLog += text
            if _stderrLog.count > 32 * 1024 {
                _stderrLog = String(_stderrLog.suffix(32 * 1024))
            }
        }
    }
}
