import Foundation

enum TrainingTermination: Equatable {
    case completed
    case earlyStopped(patience: Int?)
    case cancelled
    case failed
    case interrupted
    case inProgress
}

/// Objective, threshold-free facts used by both the live terminal page and history.
/// Every value comes from training events, the persisted run, or adjacent metadata.
struct TrainingResultExplanation: Equatable {
    let termination: TrainingTermination
    let configuredEpochs: Int?
    let actualEpochs: Int
    let firstEpochLoss: Double?
    let finalEpochLoss: Double?
    let relativeTrainLossChangePercent: Double?
    let firstHeldOutLoss: Double?
    let finalHeldOutLoss: Double?
    let bestHeldOutLoss: Double?
    let bestHeldOutEpoch: Int?
    let stateStd: Double?
    let elapsedSeconds: Double?
    let trainSamples: Int?
    let heldOutSamples: Int?
    let truncatedSamples: Int?
    let droppedSamples: Int?
    let template: String?
    let contextLength: Int?
    let modelName: String?
    let precision: String?
    let dataHash: String?
    let failureMessage: String?
    let lastStartedEpoch: Int?
    let completedSteps: Int?

    var completedAtLeastOneEpoch: Bool { actualEpochs > 0 }

    static func relativeChangePercent(first: Double?, final: Double?) -> Double? {
        guard let first, let final, first.isFinite, final.isFinite, first != 0 else { return nil }
        return (final - first) / first * 100
    }

    @MainActor
    init(store: TrainStore) {
        let run = store.currentRun
        let points = store.epochLossPoints.sorted { $0.epoch < $1.epoch }
        let first = points.first?.trainLoss ?? run?.summary.firstEpochLoss
        let final = points.last?.trainLoss ?? run?.summary.finalLoss
        let held = points.compactMap { point in
            point.heldOutLoss.map { (epoch: point.epoch + 1, loss: $0) }
        }
        let bestHeld = held.min { $0.loss < $1.loss }
        let earlyStopped = store.earlyStopInfo != nil || run?.summary.earlyStopped == true
        termination = Self.termination(
            status: run?.status ?? Self.status(for: store.state),
            earlyStopped: earlyStopped,
            patience: run?.config?.earlyStopPatience ?? store.configSnapshot?.earlyStopPatience
        )
        configuredEpochs = run?.config?.epochs ?? store.configSnapshot?.epochs
        actualEpochs = points.last.map { $0.epoch + 1 } ?? run?.summary.actualEpochs ?? 0
        firstEpochLoss = first
        finalEpochLoss = final
        relativeTrainLossChangePercent = Self.relativeChangePercent(first: first, final: final)
        firstHeldOutLoss = held.first?.loss
        finalHeldOutLoss = held.last?.loss
        bestHeldOutLoss = bestHeld?.loss ?? run?.summary.heldOutLoss
        bestHeldOutEpoch = bestHeld?.epoch ?? run?.summary.bestHeldOutEpoch
        stateStd = points.last?.stateStd ?? run?.summary.stateStd
        elapsedSeconds = store.elapsedSeconds ?? run?.summary.elapsedSeconds
        trainSamples = store.trainSampleCount ?? run?.summary.trainSamples
        heldOutSamples = store.heldOutSampleCount ?? run?.summary.heldOutSamples
        truncatedSamples = store.truncatedSampleCount ?? run?.summary.truncatedSamples
        droppedSamples = store.droppedSampleCount ?? run?.summary.droppedSamples
        template = run?.config?.template
        contextLength = run?.config?.contextLength ?? store.configSnapshot?.ctxLen
        modelName = run?.config.map { URL(fileURLWithPath: $0.modelPath).lastPathComponent }
        precision = run?.config.map { ModelConfigProbe.precisionBadge(for: $0.modelPath).uppercased() }
        dataHash = run?.summary.dataHash ?? run?.config?.datasetSHA256
        failureMessage = run?.failureMessage ?? store.errorMessage ?? store.cancelledMessage
        lastStartedEpoch = store.lastStartedEpoch
        completedSteps = store.displayedCurrentStep
    }

    init(run: TrainingRun, events: [TrainEvent], metadata: StateMetadata?) {
        var epochFacts: [(epoch: Int, loss: Double, held: Double?, stateStd: Double)] = []
        var eventTrainSamples: Int?
        var eventHeldOutSamples: Int?
        var eventTruncated: Int?
        var eventDropped: Int?
        var earlyStopped = run.summary.earlyStopped == true
        var latestStartedEpoch: Int?
        var latestStep: Int?
        for event in events {
            switch event {
            case .epochStart(let epoch, _):
                latestStartedEpoch = epoch + 1
            case .step(let step, _, _, _, let epoch, _):
                latestStep = step + 1
                if let epoch { latestStartedEpoch = epoch + 1 }
            case .epochEnd(let epoch, let loss, let stateStd, _, let held, _, _, _):
                epochFacts.append((epoch + 1, loss, held, stateStd))
            case .dataSummary(_, _, let train, let heldOut, let truncated, let dropped, _, _):
                eventTrainSamples = train
                eventHeldOutSamples = heldOut
                eventTruncated = truncated
                eventDropped = dropped
            case .earlyStop:
                earlyStopped = true
            case .start, .resume, .stdWarning, .checkpoint, .final,
                 .completed, .failed, .cancelled, .unknown:
                break
            }
        }
        epochFacts.sort { $0.epoch < $1.epoch }
        let held = epochFacts.compactMap { item in
            item.held.map { (epoch: item.epoch, loss: $0) }
        }
        let bestHeld = held.min { $0.loss < $1.loss }
        let first = epochFacts.first?.loss ?? run.summary.firstEpochLoss
        let final = epochFacts.last?.loss ?? run.summary.finalLoss ?? metadata?.result?.finalLoss

        termination = Self.termination(
            status: run.status,
            earlyStopped: earlyStopped,
            patience: run.config?.earlyStopPatience ?? metadata?.config?.earlyStopPatience
        )
        configuredEpochs = run.config?.epochs ?? metadata?.config?.epochs
        actualEpochs = epochFacts.last?.epoch
            ?? run.summary.actualEpochs
            ?? metadata?.result?.epochsRun
            ?? 0
        firstEpochLoss = first
        finalEpochLoss = final
        relativeTrainLossChangePercent = Self.relativeChangePercent(first: first, final: final)
        firstHeldOutLoss = held.first?.loss
        finalHeldOutLoss = held.last?.loss
        bestHeldOutLoss = bestHeld?.loss
            ?? run.summary.heldOutLoss
            ?? metadata?.result?.bestHeldOutLoss
        bestHeldOutEpoch = bestHeld?.epoch
            ?? run.summary.bestHeldOutEpoch
            ?? metadata?.result?.bestHeldOutEpoch
        stateStd = epochFacts.last?.stateStd
            ?? run.summary.stateStd
            ?? metadata?.result?.finalStateStd
        elapsedSeconds = run.summary.elapsedSeconds
            ?? metadata?.result?.elapsed
            ?? Self.wallClockElapsed(run)
        trainSamples = eventTrainSamples
            ?? run.summary.trainSamples
            ?? metadata?.dataStats?.trainSamples
        heldOutSamples = eventHeldOutSamples
            ?? run.summary.heldOutSamples
            ?? metadata?.dataStats?.heldOutSamples
        truncatedSamples = eventTruncated
            ?? run.summary.truncatedSamples
            ?? metadata?.dataStats?.truncated
        droppedSamples = eventDropped
            ?? run.summary.droppedSamples
            ?? metadata?.dataStats?.droppedSamples
        template = run.config?.template ?? metadata?.template
        contextLength = run.config?.contextLength ?? metadata?.config?.ctxLen
        let modelPath = run.config?.modelPath ?? metadata?.modelPath
        modelName = modelPath.map { URL(fileURLWithPath: $0).lastPathComponent }
        precision = metadata?.precision?.weights?.uppercased()
            ?? modelPath.map { ModelConfigProbe.precisionBadge(for: $0).uppercased() }
        dataHash = run.summary.dataHash
            ?? metadata.flatMap { $0.dataSHA256.isEmpty ? nil : $0.dataSHA256 }
            ?? run.config?.datasetSHA256
        failureMessage = run.failureMessage
        lastStartedEpoch = latestStartedEpoch
        completedSteps = latestStep
    }

    var abbreviatedDataHash: String? {
        guard let dataHash else { return nil }
        guard dataHash.count > 16 else { return dataHash }
        return "\(dataHash.prefix(12))…\(dataHash.suffix(4))"
    }

    var diagnosticText: String {
        var lines = ["Preen training result"]
        lines.append("termination: \(termination.diagnosticValue)")
        lines.append("epochs: \(actualEpochs)/\(configuredEpochs.map(String.init) ?? "unknown")")
        if let firstEpochLoss { lines.append("first_epoch_loss: \(firstEpochLoss)") }
        if let finalEpochLoss { lines.append("final_epoch_loss: \(finalEpochLoss)") }
        if let bestHeldOutLoss { lines.append("best_held_out_loss: \(bestHeldOutLoss)") }
        if let bestHeldOutEpoch { lines.append("best_held_out_epoch: \(bestHeldOutEpoch)") }
        if let stateStd { lines.append("state_std: \(stateStd)") }
        if let trainSamples { lines.append("train_samples: \(trainSamples)") }
        if let heldOutSamples { lines.append("held_out_samples: \(heldOutSamples)") }
        if let truncatedSamples { lines.append("truncated_samples: \(truncatedSamples)") }
        if let droppedSamples { lines.append("dropped_samples: \(droppedSamples)") }
        if let template { lines.append("template: \(template)") }
        if let contextLength { lines.append("ctx_len: \(contextLength)") }
        if let modelName { lines.append("model: \(modelName)") }
        if let precision { lines.append("precision: \(precision)") }
        if let dataHash { lines.append("data_sha256: \(dataHash)") }
        if let elapsedSeconds { lines.append("elapsed_seconds: \(elapsedSeconds)") }
        if let failureMessage { lines.append("message: \(failureMessage)") }
        return lines.joined(separator: "\n")
    }

    private static func termination(
        status: TrainingRunStatus,
        earlyStopped: Bool,
        patience: Int?
    ) -> TrainingTermination {
        if status == .completed, earlyStopped { return .earlyStopped(patience: patience) }
        switch status {
        case .completed: return .completed
        case .cancelled: return .cancelled
        case .failed: return .failed
        case .interrupted: return .interrupted
        case .preparing, .running, .finishing: return .inProgress
        }
    }

    private static func status(for state: TrainState) -> TrainingRunStatus {
        switch state {
        case .idle, .preparing: return .preparing
        case .running: return .running
        case .finishing: return .finishing
        case .completed: return .completed
        case .failed: return .failed
        case .cancelled: return .cancelled
        }
    }

    private static func wallClockElapsed(_ run: TrainingRun) -> Double? {
        guard let start = run.startedAt, let finish = run.finishedAt else { return nil }
        return max(0, finish.timeIntervalSince(start))
    }
}

private extension TrainingTermination {
    var diagnosticValue: String {
        switch self {
        case .completed: return "completed"
        case .earlyStopped(let patience): return "early_stop(patience=\(patience.map(String.init) ?? "unknown"))"
        case .cancelled: return "cancelled"
        case .failed: return "failed"
        case .interrupted: return "interrupted"
        case .inProgress: return "in_progress"
        }
    }
}
