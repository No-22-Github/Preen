import Foundation

/// `statetuner data-info --json` 的结构化结果(对齐 inspection.DataInspection.to_dict)。
/// 用真实 tokenizer 统计有效样本 / 截断 / token 长度,供配参界面训练前 sanity check。
struct DataInspectionResult: Decodable, Equatable {
    let total: Int
    let valid: Int
    let skippedEmptyQuestion: Int
    let skippedEmptyAnswer: Int
    let truncated: Int              // 部分截断(截头保尾,target 还在,能练)
    let targetFullyTruncated: Int   // 完全截断(target 预测位被切掉,不能练)
    let minTokens: Int
    let meanTokens: Double
    let p95Tokens: Double
    let maxTokens: Int
    let ctxLen: Int
    let template: String

    enum CodingKeys: String, CodingKey {
        case total, valid, truncated, template
        case skippedEmptyQuestion = "skipped_empty_question"
        case skippedEmptyAnswer = "skipped_empty_answer"
        case targetFullyTruncated = "target_fully_truncated"
        case minTokens = "min_tokens"
        case meanTokens = "mean_tokens"
        case p95Tokens = "p95_tokens"
        case maxTokens = "max_tokens"
        case ctxLen = "ctx_len"
    }
}

/// data-info 的一次性执行结果:成功带 inspection,失败带错误文案。
enum DataInspectionOutcome: Equatable {
    case success(DataInspectionResult)
    case failure(String)
}

final class DataInspectionRunner {
    /// 跑一次 tokenizer 检查。model/data 为空或进程失败时返回 .failure。
    func inspect(modelPath: String, dataPath: String, ctxLen: Int) async -> DataInspectionOutcome {
        guard !modelPath.isEmpty, !dataPath.isEmpty else {
            return .failure(L10n.string("需要同时指定模型与数据"))
        }
        return await Task.detached(priority: .userInitiated) {
            let process = Process()
            let stdout = Pipe()
            let stderr = Pipe()
            process.executableURL = PythonResolver.executable
            process.arguments = [
                "-m", "statetuner.cli", "data-info",
                "--model", modelPath,
                "--data", dataPath,
                "--ctx-len", String(ctxLen),
                "--template", "auto",
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

            if let result = try? JSONDecoder().decode(DataInspectionResult.self, from: outData) {
                return .success(result)
            }
            // 失败:优先用 stderr 末行(data-info 的 _bad_input 走 stderr)。
            let errText = String(data: errData, encoding: .utf8) ?? ""
            let lastLine = errText.split(separator: "\n").last.map(String.init) ?? ""
            return .failure(lastLine.isEmpty
                ? L10n.format("数据检查失败(退出码 %d)", process.terminationStatus)
                : L10n.backendMessage(lastLine, fallback: "数据检查失败"))
        }.value
    }
}
