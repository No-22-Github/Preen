import Foundation

/// `statetuner state-info --json` 的只读结构预检结果。
struct StateInspectionResult: Decodable, Equatable {
    let format: String
    let layers: Int
    let continuousLayers: Bool
    let rwkv7Compatible: Bool

    enum CodingKeys: String, CodingKey {
        case format, layers
        case continuousLayers = "continuous_layers"
        case rwkv7Compatible = "rwkv7_compatible"
    }
}

enum StateInspectionOutcome: Equatable {
    case success(StateInspectionResult)
    case failure(String)
}

final class StateInspectionRunner {
    func inspect(stateURL: URL) async -> StateInspectionOutcome {
        await Task.detached(priority: .userInitiated) {
            let process = Process()
            let stdout = Pipe()
            let stderr = Pipe()
            process.executableURL = PythonResolver.executable
            process.arguments = [
                "-m", "statetuner.cli", "state-info",
                "--state", stateURL.path,
                "--json",
            ]
            process.environment = PythonResolver.childEnvironment
            process.currentDirectoryURL = PythonResolver.repoRoot
            process.standardOutput = stdout
            process.standardError = stderr
            do {
                try process.run()
            } catch {
                return .failure(error.localizedDescription)
            }
            let outData = stdout.fileHandleForReading.readDataToEndOfFile()
            let errData = stderr.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            if let result = try? JSONDecoder().decode(StateInspectionResult.self, from: outData),
               result.rwkv7Compatible, result.continuousLayers, result.layers > 0 {
                return .success(result)
            }
            if let result = try? JSONDecoder().decode(StateInspectionResult.self, from: outData) {
                return .failure(L10n.format("State 结构不兼容 RWKV-7：%d 层，格式 %@", result.layers, result.format))
            }
            let stderrText = String(data: errData, encoding: .utf8) ?? ""
            let lastLine = stderrText.split(separator: "\n").last.map(String.init) ?? ""
            return .failure(lastLine.isEmpty
                ? L10n.format("State 检查失败（退出码 %d）", process.terminationStatus)
                : L10n.backendMessage(lastLine, fallback: "State 检查失败"))
        }.value
    }
}
