import Foundation
import Observation

@Observable
@MainActor
final class BackendStore {
    private(set) var runtime = RuntimeStatus()
    private(set) var inference = WorkerStatus.inferenceIdle
    private(set) var training = WorkerStatus.trainingIdle
    private(set) var runtimeLog = ""
    private(set) var inferenceLog = ""
    private(set) var trainingLog = ""

    private let runtimeRunner: RuntimeCheckRunner

    init(runtimeRunner: RuntimeCheckRunner = RuntimeCheckRunner()) {
        self.runtimeRunner = runtimeRunner
    }

    func checkRuntime() async {
        runtime = RuntimeStatus()
        let result = await runtimeRunner.check()
        runtimeLog = result.log
        if let report = result.report, report.isUsable {
            runtime = RuntimeStatus(
                phase: .ready,
                report: report,
                message: "Python \(report.python) · MLX \(report.mlx.version ?? "就绪")",
                checkedAt: Date()
            )
        } else {
            runtime = RuntimeStatus(
                phase: .unavailable,
                report: result.report,
                message: result.errorMessage ?? "运行时不可用",
                checkedAt: Date()
            )
        }
    }

    func updateInference(phase: WorkerPhase, pid: Int32? = nil, message: String) {
        inference = WorkerStatus(phase: phase, pid: pid, message: message)
    }

    func updateTraining(phase: WorkerPhase, pid: Int32? = nil, message: String) {
        training = WorkerStatus(phase: phase, pid: pid, message: message)
    }

    func appendInferenceLog(_ text: String) {
        inferenceLog = Self.appending(text, to: inferenceLog)
    }

    func appendTrainingLog(_ text: String) {
        trainingLog = Self.appending(text, to: trainingLog)
    }

    private static func appending(_ text: String, to existing: String) -> String {
        let combined = existing + text
        return combined.count > 128 * 1024 ? String(combined.suffix(128 * 1024)) : combined
    }
}
