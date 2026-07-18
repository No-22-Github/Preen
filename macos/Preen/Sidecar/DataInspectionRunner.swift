import Foundation

enum TrainingDataPreflightOutcome: Equatable {
    case success(DatasetPreviewResult, cached: Bool)
    case failure(String)
}

/// Runs the same detector, template renderer, tokenizer inspection, and full-record
/// pass as Toolbox dataset preview, but follows the exact loader route used by train.
///
/// `runner` 必须是实例属性:ToolJobRunner 的 readTask 用 [weak self] 持有 runner,
/// 若用局部变量,函数返回后 runner 被 ARC 释放 → deinit 调 process.terminate() 杀子进程 →
/// readTask 的 guard let self 提前退出且永远不调 continuation.finish() →
/// 调用方 for await event 死等 → 界面卡在"正在按最终模板检查全部样本…"。
/// 用实例属性 + defer 释放,保证生命周期覆盖整个 for await 循环。
final class TrainingDataPreflightRunner {
    private var runner: ToolJobRunner?

    func cancel() {
        runner?.cancel()
    }

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

        let activeRunner = ToolJobRunner()
        self.runner = activeRunner
        defer { self.runner = nil }

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
        let stream = activeRunner.start(argv: argv, currentDirectory: PythonResolver.repoRoot)
        for await event in stream {
            if Task.isCancelled {
                activeRunner.cancel()
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
