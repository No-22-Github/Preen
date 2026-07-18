import Foundation

enum TrainingRunKind: String, Codable, CaseIterable {
    case training
    case imported
}

enum TrainingRunStatus: String, Codable, CaseIterable {
    case preparing
    case running
    case finishing
    case completed
    case failed
    case cancelled
    case interrupted

    var isTerminal: Bool {
        switch self {
        case .completed, .failed, .cancelled, .interrupted:
            return true
        case .preparing, .running, .finishing:
            return false
        }
    }
}

struct PersistedTrainingConfig: Codable, Equatable {
    var modelPath: String
    var dataPath: String
    var outputPath: String
    var template: String
    var learningRate: Double
    var learningRateFloor: Double
    var warmup: Int
    var contextLength: Int
    var epochs: Int
    var gradientClip: Double
    var earlyStop: Bool
    var earlyStopPatience: Int
    var testRatio: Double
    var seed: Int
    var cacheLimitGB: String
    var checkpointDirectory: String? = nil
    var exportPth: Bool = false
    var pthOutputPath: String? = nil
    var datasetSource: String? = nil
    var datasetVersion: String? = nil
    var datasetSHA256: String? = nil
}

struct TrainingRunArtifacts: Codable, Equatable {
    var statePath: String?
    var metadataPath: String?
    var pthPath: String?
    var checkpoints: [String]

    static let empty = TrainingRunArtifacts(checkpoints: [])
}

struct TrainingRunSummary: Codable, Equatable {
    var actualEpochs: Int?
    var finalLoss: Double?
    var heldOutLoss: Double?
    var stateStd: Double?
    var elapsedSeconds: Double?
    var dataHash: String?

    static let empty = TrainingRunSummary()
}

struct ComparisonMetrics: Codable, Equatable {
    var stopReason: String?
    var tokenCount: Int?
    var generationTPS: Double?

    init(result: GenerationResult?) {
        stopReason = result?.stopReason
        tokenCount = result?.tokenCount
        generationTPS = result?.generationTps
    }
}

struct SavedComparison: Codable, Identifiable, Equatable {
    let id: UUID
    let prompt: String
    let baselineText: String
    let stateText: String
    let template: String
    let reasoning: Bool
    let think: String
    let genConfig: GenConfig
    let baseline: ComparisonMetrics
    let withState: ComparisonMetrics
    let createdAt: Date

    init(
        id: UUID = UUID(),
        prompt: String,
        baselineText: String,
        stateText: String,
        template: String,
        reasoning: Bool,
        think: String,
        genConfig: GenConfig,
        baseline: ComparisonMetrics,
        withState: ComparisonMetrics,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.prompt = prompt
        self.baselineText = baselineText
        self.stateText = stateText
        self.template = template
        self.reasoning = reasoning
        self.think = think
        self.genConfig = genConfig
        self.baseline = baseline
        self.withState = withState
        self.createdAt = createdAt
    }
}

struct TrainingRun: Codable, Identifiable, Equatable {
    static let schemaVersion = 1

    var schema: Int = schemaVersion
    let id: UUID
    var kind: TrainingRunKind
    var status: TrainingRunStatus
    let createdAt: Date
    var updatedAt: Date
    var startedAt: Date?
    var finishedAt: Date?
    var config: PersistedTrainingConfig?
    var artifacts: TrainingRunArtifacts
    var summary: TrainingRunSummary
    var failureMessage: String?

    init(
        id: UUID = UUID(),
        kind: TrainingRunKind = .training,
        status: TrainingRunStatus = .preparing,
        createdAt: Date = Date(),
        config: PersistedTrainingConfig? = nil
    ) {
        self.id = id
        self.kind = kind
        self.status = status
        self.createdAt = createdAt
        self.updatedAt = createdAt
        self.config = config
        self.artifacts = .empty
        self.summary = .empty
    }
}

extension TrainingRun {
    mutating func apply(event: TrainEvent) {
        let date = Date(timeIntervalSince1970: event.timestamp)
        updatedAt = date
        switch event {
        case .start:
            status = .running
            startedAt = date
        case .resume, .epochStart, .step, .stdWarning, .earlyStop, .unknown:
            break
        case .epochEnd(let epoch, let loss, let stateStd, _, let heldOutLoss, _, _, _):
            summary.actualEpochs = epoch + 1
            summary.finalLoss = loss
            summary.heldOutLoss = heldOutLoss
            summary.stateStd = stateStd
        case .checkpoint(_, let path, _):
            if !artifacts.checkpoints.contains(path) {
                artifacts.checkpoints.append(path)
            }
        case .final(let path, let elapsed, let best, _):
            status = .finishing
            artifacts.statePath = path
            summary.elapsedSeconds = elapsed
            if let best { summary.heldOutLoss = best }
        case .completed(let path, let elapsed, _, _):
            status = .completed
            artifacts.statePath = path
            summary.elapsedSeconds = elapsed
            finishedAt = date
        case .failed(let message, _, _):
            status = .failed
            failureMessage = message
            finishedAt = date
        case .cancelled(let message, _):
            status = .cancelled
            failureMessage = message
            finishedAt = date
        }
    }
}
