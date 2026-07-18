import Foundation

enum TrainingDataPreflightOutcome: Equatable {
    case success(DatasetPreviewResult, cached: Bool)
    case failure(String)
}

/// Runs the same detector, template renderer, tokenizer inspection, and full-record
/// pass as Toolbox dataset preview, but follows the exact loader route used by train.
final class TrainingDataPreflightRunner {
    func inspect(
        modelPath: String,
        dataPath: String,
        ctxLen: Int,
        template: String
    ) async -> TrainingDataPreflightOutcome {
        guard !modelPath.isEmpty, !dataPath.isEmpty else {
            return .failure(L10n.string("需要同时指定模型与数据"))
        }
        let key = DatasetPreflightCache.makeKey(
            modelPath: modelPath,
            dataPath: dataPath,
            ctxLen: ctxLen,
            template: template,
            trainingDataRoute: true
        )
        if let cached = DatasetPreflightCache.load(key) {
            return .success(cached, cached: true)
        }

        let runner = ToolJobRunner()
        let argv = [
            "-m", "statetuner.cli", "dataset-preview",
            "--model", modelPath,
            "--data", dataPath,
            "--ctx-len", String(ctxLen),
            "--template", template,
            "--training-data-route",
            "--cache-out", key.previewURL.path,
            "--page-size", "3",
        ]
        let stream = runner.start(argv: argv, currentDirectory: PythonResolver.repoRoot)
        for await event in stream {
            if Task.isCancelled {
                runner.cancel()
                return .failure(L10n.string("数据检查已取消"))
            }
            switch event.type {
            case .completed:
                guard event.tool == "dataset_preview", let value = event.result,
                      let result = try? value.decode(DatasetPreviewResult.self)
                else { return .failure(L10n.string("无法解析数据检查结果")) }
                DatasetPreflightCache.save(result, for: key)
                return .success(result, cached: false)
            case .failed:
                return .failure(L10n.backendMessage(
                    event.message ?? "",
                    fallback: "数据检查失败"
                ))
            case .cancelled:
                return .failure(L10n.string("数据检查已取消"))
            case .started, .progress, .warning:
                continue
            }
        }
        return .failure(L10n.string("数据检查进程异常退出"))
    }
}
