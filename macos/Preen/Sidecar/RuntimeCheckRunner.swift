import Foundation

struct RuntimeCheckResult: Equatable {
    let report: DoctorReport?
    let log: String
    let exit: ProcessExitInfo?

    var errorMessage: String? {
        if let report, report.isUsable { return nil }
        if let report, !report.mlx.ok { return report.mlx.error ?? L10n.string("MLX 不可用") }
        if let report, !report.mlxLM.ok { return report.mlxLM.error ?? L10n.string("mlx-lm 不可用") }
        if let report, !report.metalAvailable { return report.metalError ?? L10n.string("Metal 不可用") }
        return log.isEmpty ? L10n.string("Python 运行时检查失败") : log
    }

    static func decode(output: Data, stderr: Data, exit: ProcessExitInfo?) -> RuntimeCheckResult {
        let report = try? JSONDecoder().decode(DoctorReport.self, from: output)
        let stderrText = String(data: stderr, encoding: .utf8) ?? ""
        let outputText = report == nil ? (String(data: output, encoding: .utf8) ?? "") : ""
        return RuntimeCheckResult(
            report: report,
            log: [stderrText, outputText].filter { !$0.isEmpty }.joined(separator: "\n"),
            exit: exit
        )
    }
}

final class RuntimeCheckRunner {
    func check() async -> RuntimeCheckResult {
        await Task.detached(priority: .userInitiated) {
            let process = Process()
            let stdout = Pipe()
            let stderr = Pipe()
            process.executableURL = PythonResolver.executable
            process.arguments = ["-m", "statetuner.cli", "doctor", "--json"]
            process.environment = PythonResolver.childEnvironment
            process.currentDirectoryURL = PythonResolver.repoRoot
            process.standardOutput = stdout
            process.standardError = stderr

            do {
                try process.run()
            } catch {
                return RuntimeCheckResult(report: nil, log: error.localizedDescription, exit: nil)
            }

            let output = stdout.fileHandleForReading.readDataToEndOfFile()
            let errorData = stderr.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            let reason: ProcessExitInfo.Reason = process.terminationReason == .exit ? .exit : .uncaughtSignal
            return RuntimeCheckResult.decode(
                output: output,
                stderr: errorData,
                exit: ProcessExitInfo(status: process.terminationStatus, reason: reason)
            )
        }.value
    }
}
