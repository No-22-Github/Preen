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
    private(set) var processMetrics: [ProcessMetric] = []
    private(set) var latestProcessMetric: ProcessMetric?
    private(set) var currentTrainingStep = 0
    private(set) var currentSecondsPerStep: Double?

    private let runtimeRunner: RuntimeCheckRunner
    private let metricsSampler: ProcessMetricsSampler
    private var metricsTask: Task<Void, Never>?

    init(
        runtimeRunner: RuntimeCheckRunner = RuntimeCheckRunner(),
        metricsSampler: ProcessMetricsSampler = ProcessMetricsSampler()
    ) {
        self.runtimeRunner = runtimeRunner
        self.metricsSampler = metricsSampler
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
        if phase == .running, let pid {
            startMetrics(pid: pid)
        } else if phase == .idle || phase == .failed {
            metricsTask?.cancel()
            metricsTask = nil
        }
    }

    func resetTrainingMetrics() {
        processMetrics.removeAll()
        latestProcessMetric = nil
        currentTrainingStep = 0
        currentSecondsPerStep = nil
    }

    func updateTrainingProgress(step: Int, secondsPerStep: Double?) {
        currentTrainingStep = step
        currentSecondsPerStep = secondsPerStep
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

    private func startMetrics(pid: Int32) {
        metricsTask?.cancel()
        metricsTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                let step = currentTrainingStep
                let seconds = currentSecondsPerStep
                let sampler = metricsSampler
                if let metric = await Task.detached(priority: .utility, operation: {
                    sampler.sample(pid: pid, step: step, secondsPerStep: seconds)
                }).value {
                    latestProcessMetric = metric
                    if let last = processMetrics.last, last.step == metric.step {
                        processMetrics[processMetrics.count - 1] = last.mergingPeak(with: metric)
                    } else {
                        processMetrics.append(metric)
                    }
                }
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }
}
