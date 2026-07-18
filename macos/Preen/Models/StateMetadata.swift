import Foundation

struct StateMetadata: Decodable, Equatable {
    let formatVersion: Int
    let createdAt: Double
    let modelName: String
    let modelPath: String
    let data: String
    let dataSHA256: String
    let template: String?
    let stateFormat: String?
    let stateDtype: String?
    let precision: Precision?
    let config: Config?
    let dataStats: DataStats?
    let result: Result?
    let artifacts: Artifacts?

    /// v1 UI compatibility: historical metadata called this path `model`.
    var model: String { modelPath }

    struct Config: Decodable, Equatable {
        let lr: Double
        let lrFloor: Double
        let warmup: Int
        let ctxLen: Int
        let epochs: Int
        let gradClip: Double
        let logEvery: Int
        let earlyStop: Bool
        let earlyStopPatience: Int
        let checkpointDir: String?
        let checkpointEvery: Int
        let resume: String?
        let seed: Int

        enum CodingKeys: String, CodingKey {
            case lr, warmup, epochs, resume, seed
            case lrFloor = "lr_floor"
            case ctxLen = "ctx_len"
            case gradClip = "grad_clip"
            case logEvery = "log_every"
            case earlyStop = "early_stop"
            case earlyStopPatience = "early_stop_patience"
            case checkpointDir = "checkpoint_dir"
            case checkpointEvery = "checkpoint_every"
        }
    }

    struct Result: Decodable, Equatable {
        let epochsRun: Int
        let finalLoss: Double
        let finalStateStd: Double
        let bestHeldOutLoss: Double?
        let bestHeldOutEpoch: Int?
        let elapsed: Double

        enum CodingKeys: String, CodingKey {
            case elapsed
            case epochsRun = "epochs_run"
            case finalLoss = "final_loss"
            case finalStateStd = "final_state_std"
            case bestHeldOutLoss = "best_held_out_loss"
            case bestHeldOutEpoch = "best_held_out_epoch"
        }
    }

    struct DataStats: Decodable, Equatable {
        let total: Int?
        let valid: Int?
        let truncated: Int?
        let targetFullyTruncated: Int?
        let trainSamples: Int?
        let heldOutSamples: Int?
        let droppedSamples: Int?

        enum CodingKeys: String, CodingKey {
            case total, valid, truncated
            case targetFullyTruncated = "target_fully_truncated"
            case trainSamples = "train_samples"
            case heldOutSamples = "held_out_samples"
            case droppedSamples = "dropped_samples"
        }
    }

    struct Precision: Decodable, Equatable {
        let weights: String?
        let trainState: String?
        let export: String?

        enum CodingKeys: String, CodingKey {
            case weights, export
            case trainState = "train_state"
        }
    }

    struct Artifacts: Decodable, Equatable {
        let stateNPZ: String
        let statePTH: String?

        enum CodingKeys: String, CodingKey {
            case stateNPZ = "state_npz"
            case statePTH = "state_pth"
        }
    }

    enum CodingKeys: String, CodingKey {
        case model, data, template, precision, config, result, artifacts
        case dataStats = "data_stats"
        case modelName = "model_name"
        case modelPath = "model_path"
        case stateFormat = "state_format"
        case stateDtype = "state_dtype"
        case formatVersion = "format_version"
        case createdAt = "created_at"
        case dataSHA256 = "data_sha256"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        formatVersion = try c.decodeIfPresent(Int.self, forKey: .formatVersion) ?? 1
        createdAt = try c.decodeIfPresent(Double.self, forKey: .createdAt) ?? 0
        let legacyModel = try c.decodeIfPresent(String.self, forKey: .model) ?? ""
        modelPath = try c.decodeIfPresent(String.self, forKey: .modelPath) ?? legacyModel
        modelName = try c.decodeIfPresent(String.self, forKey: .modelName)
            ?? URL(fileURLWithPath: modelPath).lastPathComponent
        data = try c.decodeIfPresent(String.self, forKey: .data) ?? ""
        dataSHA256 = try c.decodeIfPresent(String.self, forKey: .dataSHA256) ?? ""
        template = try c.decodeIfPresent(String.self, forKey: .template)
        stateFormat = try c.decodeIfPresent(String.self, forKey: .stateFormat)
        stateDtype = try c.decodeIfPresent(String.self, forKey: .stateDtype)
        precision = try c.decodeIfPresent(Precision.self, forKey: .precision)
        config = try c.decodeIfPresent(Config.self, forKey: .config)
        dataStats = try c.decodeIfPresent(DataStats.self, forKey: .dataStats)
        result = try c.decodeIfPresent(Result.self, forKey: .result)
        artifacts = try c.decodeIfPresent(Artifacts.self, forKey: .artifacts)
    }

    static func load(from url: URL) throws -> StateMetadata {
        try JSONDecoder().decode(StateMetadata.self, from: Data(contentsOf: url))
    }

    static func adjacentURL(for stateURL: URL) -> URL {
        stateURL.deletingPathExtension().appendingPathExtension("meta.json")
    }

    static func loadAdjacent(to stateURL: URL) -> StateMetadata? {
        try? load(from: adjacentURL(for: stateURL))
    }
}
