import Foundation

enum RuntimePhase: String, Codable, Equatable {
    case checking
    case ready
    case unavailable
}

enum WorkerPhase: String, Codable, Equatable {
    case idle
    case starting
    case ready
    case running
    case stopping
    case failed
}

struct DoctorModule: Codable, Equatable {
    let ok: Bool
    let version: String?
    let error: String?
}

struct DoctorReport: Codable, Equatable {
    let python: String
    let platform: String
    let machine: String
    let appleSilicon: Bool
    let numpy: DoctorModule
    let mlDtypes: DoctorModule
    let mlx: DoctorModule
    let mlxLM: DoctorModule
    let metalAvailable: Bool
    let metalError: String?
    let memorySizeGB: Double?
    let workingSetGB: Double?

    enum CodingKeys: String, CodingKey {
        case python, platform, machine, numpy, mlx
        case appleSilicon = "apple_silicon"
        case mlDtypes = "ml_dtypes"
        case mlxLM = "mlx_lm"
        case metalAvailable = "metal_available"
        case metalError = "metal_error"
        case memorySizeGB = "memory_size_gb"
        case workingSetGB = "working_set_gb"
    }

    var isUsable: Bool {
        appleSilicon && mlx.ok && mlxLM.ok && metalAvailable
    }
}

struct RuntimeStatus: Equatable {
    var phase: RuntimePhase = .checking
    var report: DoctorReport?
    var message = "正在检查 Python 与 MLX"
    var checkedAt: Date?
}

struct WorkerStatus: Equatable {
    var phase: WorkerPhase = .idle
    var pid: Int32?
    var message: String

    static let inferenceIdle = WorkerStatus(phase: .idle, message: "推理未启动")
    static let trainingIdle = WorkerStatus(phase: .idle, message: "没有训练任务")
}

struct ProcessExitInfo: Equatable {
    enum Reason: String, Codable {
        case exit
        case uncaughtSignal
    }

    let status: Int32
    let reason: Reason
}
