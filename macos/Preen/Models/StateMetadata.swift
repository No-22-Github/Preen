import Foundation

struct StateMetadata: Decodable, Equatable {
    let formatVersion: Int
    let createdAt: Double
    let model: String
    let data: String
    let dataSHA256: String
    let template: String
    let config: Config
    let result: Result
    let artifacts: Artifacts

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
        let elapsed: Double

        enum CodingKeys: String, CodingKey {
            case elapsed
            case epochsRun = "epochs_run"
            case finalLoss = "final_loss"
            case finalStateStd = "final_state_std"
            case bestHeldOutLoss = "best_held_out_loss"
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
        case model, data, template, config, result, artifacts
        case formatVersion = "format_version"
        case createdAt = "created_at"
        case dataSHA256 = "data_sha256"
    }

    static func load(from url: URL) throws -> StateMetadata {
        try JSONDecoder().decode(StateMetadata.self, from: Data(contentsOf: url))
    }
}
