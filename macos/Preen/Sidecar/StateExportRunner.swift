import Foundation

struct StateExportResult {
    let output: URL
    let log: String
}

enum StateExportError: LocalizedError {
    case failed(String)

    var errorDescription: String? {
        if case .failed(let message) = self { return message }
        return nil
    }
}

final class StateExportRunner {
    func export(state: URL, output: URL) async throws -> StateExportResult {
        try await Task.detached(priority: .userInitiated) {
            let process = Process()
            let stdout = Pipe()
            let stderr = Pipe()
            process.executableURL = PythonResolver.executable
            process.arguments = [
                "-m", "statetuner.cli", "export",
                "--state", state.path, "--out", output.path,
            ]
            process.environment = PythonResolver.childEnvironment
            process.currentDirectoryURL = PythonResolver.repoRoot
            process.standardOutput = stdout
            process.standardError = stderr
            try process.run()
            let outputData = stdout.fileHandleForReading.readDataToEndOfFile()
            let errorData = stderr.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            let log = [outputData, errorData]
                .compactMap { String(data: $0, encoding: .utf8) }
                .filter { !$0.isEmpty }
                .joined(separator: "\n")
            guard process.terminationStatus == 0 else {
                throw StateExportError.failed(log.isEmpty ? "导出 .pth 失败" : log)
            }
            return StateExportResult(output: output, log: log)
        }.value
    }
}
