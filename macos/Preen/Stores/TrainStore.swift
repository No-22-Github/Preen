//
//  TrainStore.swift
//  Preen
//
//  训练状态机 + 事件累积。@Observable @MainActor(UI 线程)。
//
//  状态机(六态,design.md §4.2):
//   idle → running → finishing → completed
//                       ↘ failed
//                       ↘ cancelled
//
//  ⚠️ 铁律(design.md §4.2 验收 c):
//   - final ≠ 完成。final 事件 = Trainer 跑完,**产物可能尚未落盘** → 切「finishing」(收尾中)。
//   - 只有 completed 事件才切「completed」(唯一可信完成信号)。
//   - 构造只发 final 不发 completed 的场景测 UI,必须停在 finishing。
//

import Foundation
import Observation

/// 训练状态机。
enum TrainState: Equatable {
    case idle
    case preparing
    case running
    case finishing  // final 收到,completed 没到
    case completed  // 唯一可信完成
    case failed
    case cancelled
}

/// epoch 边界(Swift Charts 画 RuleMark)。
struct EpochBoundary: Identifiable, Equatable {
    let id = UUID()
    let epoch: Int
    let step: Int  // 该 epoch 起始步(用首个 step 的 step 号近似)
}

/// held-out loss 一个点(虚线)。
struct HeldOutPoint: Identifiable, Equatable {
    var id: Int { epoch }
    let epoch: Int
    let step: Int
    let loss: Double
}

@Observable
@MainActor
final class TrainStore {

    // === 状态机 ===
    private(set) var state: TrainState = .idle

    // === 损耗曲线 ===
    private(set) var lossPoints: [TrainingMetric] = []
    private(set) var heldOutPoints: [HeldOutPoint] = []
    private(set) var epochLossPoints: [EpochLossPoint] = []
    private(set) var epochBoundaries: [EpochBoundary] = []

    // === 进度(3 秒判据:不点不滚能读到)===
    private(set) var currentEpoch: Int = 0
    private(set) var currentStep: Int = 0
    private(set) var totalSteps: Int = 0
    private(set) var currentLoss: Double = 0
    private(set) var currentLr: Double = 0
    private(set) var startedAt: Date?
    private(set) var estimatedRemainingSeconds: Double?
    private(set) var currentSecondsPerStep: Double?
    private var stepTimingSamples: [(step: Int, timestamp: Double)] = []

    // === 配置(start 事件填,UI 顶部摘要回显)===
    private(set) var configSnapshot: TrainConfigSnapshot?

    // === 产物 ===
    private(set) var outputPath: String?      // completed 的 path
    private(set) var finalBest: Double?       // final 的 best(held-out 最佳)
    private(set) var elapsed: Double?         // completed 的 elapsed
    private(set) var checkpoints: [(epoch: Int, path: String)] = []
    private(set) var earlyStopInfo: (epoch: Int, best: Double, heldOutLoss: Double)?

    // === 失败/取消 ===
    private(set) var errorMessage: String?
    private(set) var cancelledMessage: String?

    // === 诊断 ===
    private(set) var unknownEventCount: Int = 0  // 演进兜底命中次数

    // === runner 持有(取消用)===
    private var runner: TrainJobRunner?
    private let repository: RunRepository
    private let backendStore: BackendStore
    private let dockProgress: any DockProgressControlling
    private let notifications = TrainingNotificationController()
    private var preparationTask: Task<Void, Never>?
    private(set) var currentRun: TrainingRun?
    private(set) var currentRunDirectory: URL?

    init(
        repository: RunRepository,
        backendStore: BackendStore,
        dockProgress: (any DockProgressControlling)? = nil
    ) {
        self.repository = repository
        self.backendStore = backendStore
        self.dockProgress = dockProgress ?? DockProgressController()
    }

    convenience init() {
        self.init(repository: RunRepository(), backendStore: BackendStore())
    }

    // MARK: - 生命周期

    /// 配置 + runner 注入,开始训练。UI 调用。
    func start(config: TrainingConfig) {
        reset()
        backendStore.resetTrainingMetrics()
        notifications.prepare()
        dockProgress.update(progress: 0)
        state = .preparing
        let run = TrainingRun(config: config.persisted)
        currentRun = run
        backendStore.updateTraining(phase: .starting, message: "正在准备训练记录")

        preparationTask = Task { [weak self] in
            guard let self else { return }
            do {
                let directory = try await repository.create(run)
                guard !Task.isCancelled else {
                    markPreparingRunCancelled()
                    return
                }
                currentRunDirectory = directory
                var resolvedConfig = config
                resolvedConfig.eventsFilePath = directory
                    .appendingPathComponent(RunRepository.eventsFilename).path
                let stderrURL = directory.appendingPathComponent(RunRepository.stderrFilename)
                launch(config: resolvedConfig, stderrURL: stderrURL)
            } catch {
                let message = "无法创建训练记录：\(error.localizedDescription)"
                errorMessage = message
                state = .failed
                backendStore.updateTraining(phase: .failed, message: message)
                finishBackgroundFeedback(title: "训练启动失败", body: message)
            }
        }
    }

    /// SIGINT 取消。UI 的「取消」按钮调用。
    func cancel() {
        if state == .preparing {
            preparationTask?.cancel()
            markPreparingRunCancelled()
            return
        }
        runner?.cancel()
    }

    /// 请求取消并等待训练子进程真正退出，供训练/推理互斥切换使用。
    func cancelAndWait() async {
        if state == .preparing {
            cancel()
            return
        }
        guard let runner, runner.isRunning else { return }
        backendStore.updateTraining(
            phase: .stopping,
            pid: runner.pid,
            message: "正在停止训练以启动推理"
        )
        runner.cancel()
        await runner.waitUntilExit()
    }

    var hasActiveProcess: Bool {
        switch state {
        case .preparing, .running, .finishing:
            return true
        case .idle, .completed, .failed, .cancelled:
            return runner?.isRunning == true
        }
    }

    /// 重置到 idle(清空所有状态,准备再训一个)。
    func reset() {
        preparationTask?.cancel()
        preparationTask = nil
        state = .idle
        lossPoints.removeAll()
        heldOutPoints.removeAll()
        epochLossPoints.removeAll()
        epochBoundaries.removeAll()
        currentEpoch = 0
        currentStep = 0
        totalSteps = 0
        currentLoss = 0
        currentLr = 0
        startedAt = nil
        estimatedRemainingSeconds = nil
        currentSecondsPerStep = nil
        stepTimingSamples.removeAll()
        configSnapshot = nil
        outputPath = nil
        finalBest = nil
        elapsed = nil
        checkpoints.removeAll()
        earlyStopInfo = nil
        errorMessage = nil
        cancelledMessage = nil
        unknownEventCount = 0
        runner = nil
        currentRun = nil
        currentRunDirectory = nil
        backendStore.resetTrainingMetrics()
        dockProgress.clear()
    }

    // MARK: - 事件消费(穷举,不许 default — design.md §4.2)

    func consume(event: TrainEvent) {
        switch event {
        case .start(let config, _):
            configSnapshot = config
            totalSteps = config.nSamples * config.epochs  // total_steps = epochs * n_samples
        case .resume(let epoch, _, _):
            currentEpoch = epoch
        case .epochStart(let epoch, _):
            currentEpoch = epoch
        case .step(let step, let total, let loss, let lr, let epoch, let timestamp):
            currentStep = step
            currentLoss = loss
            currentLr = lr
            totalSteps = total
            if let ep = epoch { currentEpoch = ep }
            lossPoints.append(TrainingMetric(
                step: step, loss: loss, learningRate: lr, epoch: currentEpoch, timestamp: timestamp
            ))
            updateEta(step: step, timestamp: timestamp)
            backendStore.updateTrainingProgress(step: step, secondsPerStep: currentSecondsPerStep)
            dockProgress.update(progress: progress)
        case .epochEnd(let epoch, let loss, let stateStd, _, let heldOut, let best, _, _):
            currentEpoch = epoch
            let epochEndStep = actualEpochEndStep(epoch: epoch)
            epochBoundaries.append(EpochBoundary(epoch: epoch, step: epochEndStep))
            epochLossPoints.append(EpochLossPoint(
                epoch: epoch, step: epochEndStep, trainLoss: loss,
                heldOutLoss: heldOut, stateStd: stateStd
            ))
            if let h = heldOut {
                heldOutPoints.append(HeldOutPoint(epoch: epoch, step: epochEndStep, loss: h))
            }
            _ = best  // best 暂存到 final 处理;epoch_end 的 best 不一定是最终
            _ = loss  // epoch 平均 loss,暂不用(本期只画 step loss)
        case .stdWarning:
            // 产品默认不启用(AGENTS.md:state std 健康区间未标定,只记录不报警)。
            // 收到即忽略(design.md §4.2 事件映射表)。
            break
        case .checkpoint(let epoch, let path, _):
            checkpoints.append((epoch: epoch, path: path))
        case .earlyStop(let epoch, let best, let heldOutLoss, _, _):
            earlyStopInfo = (epoch: epoch, best: best, heldOutLoss: heldOutLoss)
        case .final(_, let elapsed, let best, _):
            // ⚠️ final ≠ 完成(design.md §4.2 验收 c)。切「收尾中」,等 completed。
            self.elapsed = elapsed
            self.finalBest = best
            if state == .running {
                state = .finishing
            }
        case .completed(let path, let elapsed, _, _):
            // ✅ 唯一可信完成信号。
            let shouldNotify = state == .running || state == .finishing
            self.outputPath = path
            self.elapsed = elapsed
            if totalSteps > 0 { currentStep = totalSteps - 1 }
            estimatedRemainingSeconds = 0
            state = .completed
            backendStore.updateTraining(phase: .idle, message: "训练已完成")
            if shouldNotify {
                finishBackgroundFeedback(
                    title: "训练已完成",
                    body: "State 已写入 \(URL(fileURLWithPath: path).lastPathComponent)"
                )
            }
        case .failed(let message, _, _):
            let shouldNotify = state == .running || state == .finishing
            self.errorMessage = message
            state = .failed
            backendStore.updateTraining(phase: .failed, message: message)
            if shouldNotify { finishBackgroundFeedback(title: "训练失败", body: message) }
        case .cancelled(let message, _):
            let shouldNotify = state == .running || state == .finishing
            self.cancelledMessage = message
            state = .cancelled
            backendStore.updateTraining(phase: .idle, message: "训练已取消")
            if shouldNotify { finishBackgroundFeedback(title: "训练已取消", body: message) }
        case .unknown:
            // 演进兜底:Python 加新事件类型不让旧 app 崩。
            // 不切状态,只计数(UI 可提示「收到未知事件 N 次,建议升级」)。
            unknownEventCount += 1
            #if DEBUG
            print("[TrainStore] unknown event: \(event)")
            #endif
        }
        updatePersistedRun(with: event)
    }

    // MARK: - 内部

    private func updateEta(step: Int, timestamp: Double) {
        if let previous = stepTimingSamples.last, step > previous.step, timestamp >= previous.timestamp {
            stepTimingSamples.append((step: step, timestamp: timestamp))
        } else if stepTimingSamples.isEmpty {
            stepTimingSamples.append((step: step, timestamp: timestamp))
        }
        if stepTimingSamples.count > 21 {
            stepTimingSamples.removeFirst(stepTimingSamples.count - 21)
        }
        guard stepTimingSamples.count >= 2, totalSteps > 0 else {
            estimatedRemainingSeconds = nil
            currentSecondsPerStep = nil
            return
        }
        let pairs = zip(stepTimingSamples, stepTimingSamples.dropFirst())
        let perStep = pairs.compactMap { older, newer -> Double? in
            let stepDelta = newer.step - older.step
            guard stepDelta > 0 else { return nil }
            return (newer.timestamp - older.timestamp) / Double(stepDelta)
        }
        guard !perStep.isEmpty else {
            estimatedRemainingSeconds = nil
            currentSecondsPerStep = nil
            return
        }
        let average = perStep.reduce(0, +) / Double(perStep.count)
        currentSecondsPerStep = average
        estimatedRemainingSeconds = average * Double(max(0, totalSteps - step - 1))
    }

    private func actualEpochEndStep(epoch: Int) -> Int {
        guard let configSnapshot else { return currentStep }
        return min(max(0, totalSteps - 1), max(0, (epoch + 1) * configSnapshot.nSamples - 1))
    }

    /// 事件流结束时,若仍在 running/finishing,说明进程异常退出。
    private func handleStreamEnd() {
        switch state {
        case .running, .finishing:
            // 没等到终结事件 → 标失败(让 UI 能恢复,不卡死在 running/finishing)。
            if errorMessage == nil {
                errorMessage = "训练进程异常退出（未发出 completed/failed/cancelled 事件）"
            }
            state = .failed
            updatePersistedRun(with: .failed(
                message: errorMessage ?? "训练进程异常退出",
                path: outputPath,
                timestamp: Date().timeIntervalSince1970
            ))
            backendStore.updateTraining(phase: .failed, message: errorMessage ?? "训练进程异常退出")
            finishBackgroundFeedback(title: "训练异常退出", body: errorMessage ?? "训练进程异常退出")
        case .idle, .preparing, .completed, .failed, .cancelled:
            break
        }
    }

    private func launch(config: TrainingConfig, stderrURL: URL) {
        state = .running
        startedAt = Date()
        let runner = TrainJobRunner()
        self.runner = runner
        runner.onStderr = { [weak self] chunk in
            Task { @MainActor [weak self] in self?.backendStore.appendTrainingLog(chunk) }
        }
        runner.onExit = { [weak self] info in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if info.status == 0, self.state == .completed {
                    self.backendStore.updateTraining(phase: .idle, message: "训练已完成")
                }
            }
        }
        let stream = runner.start(
            argv: config.commandLineArguments(),
            currentDirectory: PythonResolver.repoRoot,
            stderrFile: stderrURL
        )
        backendStore.updateTraining(phase: .running, pid: runner.pid, message: "训练中")
        Task { [weak self] in
            for await event in stream {
                self?.consume(event: event)
            }
            self?.handleStreamEnd()
        }
    }

    private func markPreparingRunCancelled() {
        guard state == .preparing else { return }
        cancelledMessage = "训练在启动前已取消"
        state = .cancelled
        if var run = currentRun {
            let now = Date()
            run.status = .cancelled
            run.updatedAt = now
            run.finishedAt = now
            run.failureMessage = cancelledMessage
            currentRun = run
            Task { try? await repository.save(run) }
        }
        backendStore.updateTraining(phase: .idle, message: "训练已取消")
        finishBackgroundFeedback(title: "训练已取消", body: cancelledMessage ?? "训练在启动前已取消")
    }

    private func finishBackgroundFeedback(title: String, body: String) {
        dockProgress.clear()
        notifications.send(title: title, body: body)
    }

    private func updatePersistedRun(with event: TrainEvent) {
        guard var run = currentRun else { return }
        run.apply(event: event)
        if case .completed(let path, _, _, _) = event {
            associateArtifacts(statePath: path, with: &run)
        }
        currentRun = run

        let shouldSave: Bool
        switch event {
        case .start, .epochEnd, .checkpoint, .final, .completed, .failed, .cancelled:
            shouldSave = true
        case .resume, .epochStart, .step, .stdWarning, .earlyStop, .unknown:
            shouldSave = false
        }
        if shouldSave {
            Task { try? await repository.save(run) }
        }
    }

    private func associateArtifacts(statePath: String, with run: inout TrainingRun) {
        let stateURL = URL(fileURLWithPath: statePath)
        let baseURL = stateURL.deletingPathExtension()
        let metadataURL = baseURL.appendingPathExtension("meta.json")
        if FileManager.default.fileExists(atPath: metadataURL.path) {
            run.artifacts.metadataPath = metadataURL.path
            if let metadata = try? StateMetadata.load(from: metadataURL) {
                run.summary.actualEpochs = metadata.result.epochsRun
                run.summary.finalLoss = metadata.result.finalLoss
                run.summary.heldOutLoss = metadata.result.bestHeldOutLoss
                run.summary.stateStd = metadata.result.finalStateStd
                run.summary.elapsedSeconds = metadata.result.elapsed
                run.summary.dataHash = metadata.dataSHA256
            }
        }
        if let config = run.config {
            let pthURL = config.pthOutputPath.map(URL.init(fileURLWithPath:))
                ?? URL(fileURLWithPath: config.outputPath).deletingPathExtension().appendingPathExtension("pth")
            if config.exportPth, FileManager.default.fileExists(atPath: pthURL.path) {
                run.artifacts.pthPath = pthURL.path
            }
        }
    }

    // MARK: - 派生(UI 便利)

    /// 已运行秒数(从 start 到现在)。
    var elapsedSeconds: Double? {
        guard let start = startedAt else { return nil }
        if state == .completed || state == .failed || state == .cancelled {
            return elapsed
        }
        return Date().timeIntervalSince(start)
    }

    /// 预计剩余秒数。
    var remainingSeconds: Double? {
        estimatedRemainingSeconds
    }

    /// 进度百分比(0~1)。
    var progress: Double {
        guard totalSteps > 0 else { return 0 }
        let completed = lossPoints.isEmpty ? 0 : currentStep + 1
        return min(1, Double(completed) / Double(totalSteps))
    }

    var displayedCurrentStep: Int {
        lossPoints.isEmpty ? 0 : min(totalSteps, currentStep + 1)
    }

    /// 当前 loss 显示文案。
    var lossDisplay: String {
        if lossPoints.isEmpty { return "—" }
        return String(format: "%.3f", currentLoss)
    }

    /// 当前 lr 显示文案。
    var lrDisplay: String {
        if currentLr == 0 { return "—" }
        return String(format: "%.4f", currentLr)
    }

    var processMetrics: [ProcessMetric] { backendStore.processMetrics }

    var latestProcessMetric: ProcessMetric? { backendStore.latestProcessMetric }

    /// 训练运行时图表统一使用十进制 GB；doctor 尚未完成时直接读取本机物理内存兜底。
    var memoryCapacityGB: Double {
        if let reported = backendStore.runtime.report?.memorySizeGB, reported > 0 {
            return reported
        }
        return Double(ProcessInfo.processInfo.physicalMemory) / 1e9
    }

    func memoryPressure(for metric: ProcessMetric) -> MemoryPressureLevel {
        MemoryMetricMath.pressureLevel(
            footprintGB: metric.physicalFootprintGB,
            physicalMemoryGB: memoryCapacityGB,
            systemPressure: metric.pressure
        )
    }

    /// 把秒数格式化成 `Mm Ss` / `Hh Mm`。
    nonisolated static func formatDuration(_ seconds: Double) -> String {
        let s = Int(seconds)
        if s < 60 { return "\(s)s" }
        if s < 3600 { return "\(s / 60)m \(s % 60)s" }
        return "\(s / 3600)h \(s / 60 % 60)m"
    }
}
