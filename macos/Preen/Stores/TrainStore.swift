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
    case running
    case finishing  // final 收到,completed 没到
    case completed  // 唯一可信完成
    case failed
    case cancelled
}

/// loss 折线一个点。
struct LossPoint: Identifiable, Equatable {
    let id = UUID()
    let step: Int
    let loss: Double
    let epoch: Int
}

/// epoch 边界(Swift Charts 画 RuleMark)。
struct EpochBoundary: Identifiable, Equatable {
    let id = UUID()
    let epoch: Int
    let step: Int  // 该 epoch 起始步(用首个 step 的 step 号近似)
}

/// held-out loss 一个点(虚线)。
struct HeldOutPoint: Identifiable, Equatable {
    let id = UUID()
    let epoch: Int
    let loss: Double
}

@Observable
@MainActor
final class TrainStore {

    // === 状态机 ===
    private(set) var state: TrainState = .idle

    // === 损耗曲线 ===
    private(set) var lossPoints: [LossPoint] = []
    private(set) var heldOutPoints: [HeldOutPoint] = []
    private(set) var epochBoundaries: [EpochBoundary] = []

    // === 进度(3 秒判据:不点不滚能读到)===
    private(set) var currentEpoch: Int = 0
    private(set) var currentStep: Int = 0
    private(set) var totalSteps: Int = 0
    private(set) var currentLoss: Double = 0
    private(set) var currentLr: Double = 0
    private(set) var startedAt: Date?
    private(set) var estimatedTotalSeconds: Double?

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

    // MARK: - 生命周期

    /// 配置 + runner 注入,开始训练。UI 调用。
    func start(config: TrainingConfig) {
        reset()
        state = .running
        startedAt = Date()
        let runner = TrainJobRunner()
        self.runner = runner
        let cwd = PythonResolver.repoRoot  // 让相对路径可用
        let stream = runner.start(argv: config.commandLineArguments(), currentDirectory: cwd)
        // 后台消费事件流(每条都 hop 回 MainActor)。
        Task { [weak self] in
            for await event in stream {
                self?.consume(event: event)
            }
            // 流结束(stream finish = 进程退出 + 事件 drain 完)。
            // 如果此时仍是 running/finishing,说明进程异常退出没发终结事件,
            // 退回失败态让 UI 能恢复。
            self?.handleStreamEnd()
        }
    }

    /// SIGINT 取消。UI 的「取消」按钮调用。
    func cancel() {
        runner?.cancel()
    }

    /// 重置到 idle(清空所有状态,准备再训一个)。
    func reset() {
        state = .idle
        lossPoints.removeAll()
        heldOutPoints.removeAll()
        epochBoundaries.removeAll()
        currentEpoch = 0
        currentStep = 0
        totalSteps = 0
        currentLoss = 0
        currentLr = 0
        startedAt = nil
        estimatedTotalSeconds = nil
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
        case .step(let step, let total, let loss, let lr, let epoch, _):
            currentStep = step
            currentLoss = loss
            currentLr = lr
            totalSteps = total
            if let ep = epoch { currentEpoch = ep }
            lossPoints.append(LossPoint(step: step, loss: loss, epoch: currentEpoch))
            updateEta()
        case .epochEnd(let epoch, let loss, _, _, let heldOut, let best, _, _):
            currentEpoch = epoch
            // 画 epoch 边界(用当前 step 号作为 RuleMark 位置)。
            epochBoundaries.append(EpochBoundary(epoch: epoch, step: currentStep))
            if let h = heldOut {
                heldOutPoints.append(HeldOutPoint(epoch: epoch, loss: h))
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
            self.outputPath = path
            self.elapsed = elapsed
            state = .completed
        case .failed(let message, _, _):
            self.errorMessage = message
            state = .failed
        case .cancelled(let message, _):
            self.cancelledMessage = message
            state = .cancelled
        case .unknown:
            // 演进兜底:Python 加新事件类型不让旧 app 崩。
            // 不切状态,只计数(UI 可提示「收到未知事件 N 次,建议升级」)。
            unknownEventCount += 1
            #if DEBUG
            print("[TrainStore] unknown event: \(event)")
            #endif
        }
    }

    // MARK: - 内部

    private func updateEta() {
        guard let start = startedAt, totalSteps > 0, currentStep > 0 else {
            estimatedTotalSeconds = nil
            return
        }
        let elapsed = Date().timeIntervalSince(start)
        let perStep = elapsed / Double(currentStep)
        estimatedTotalSeconds = perStep * Double(totalSteps)
    }

    /// 事件流结束时,若仍在 running/finishing,说明进程异常退出。
    private func handleStreamEnd() {
        switch state {
        case .running, .finishing:
            // 没等到终结事件 → 标失败(让 UI 能恢复,不卡死在 running/finishing)。
            if errorMessage == nil {
                errorMessage = "训练进程异常退出(未发出 completed/failed/cancelled 事件)"
            }
            state = .failed
        case .idle, .completed, .failed, .cancelled:
            break
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
        guard let total = estimatedTotalSeconds, let done = elapsedSeconds else { return nil }
        return max(0, total - done)
    }

    /// 进度百分比(0~1)。
    var progress: Double {
        guard totalSteps > 0 else { return 0 }
        return min(1, Double(currentStep) / Double(totalSteps))
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

    /// 把秒数格式化成 `Mm Ss` / `Hh Mm`。
    static func formatDuration(_ seconds: Double) -> String {
        let s = Int(seconds)
        if s < 60 { return "\(s)s" }
        if s < 3600 { return "\(s / 60)m \(s % 60)s" }
        return "\(s / 3600)h \(s / 60 % 60)m"
    }
}
