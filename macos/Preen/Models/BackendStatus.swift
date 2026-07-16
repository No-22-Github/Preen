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
    let osVersion: String?
    let osBuild: String?
    let chipName: String?
    let hardwareModel: String?
    let numpy: DoctorModule
    let mlDtypes: DoctorModule
    let mlx: DoctorModule
    let mlxLM: DoctorModule
    let metalAvailable: Bool
    let metalError: String?
    let memorySizeGB: Double?
    let memorySizeGiB: Double?
    let workingSetGB: Double?
    let workingSetGiB: Double?

    enum CodingKeys: String, CodingKey {
        case python, platform, machine, numpy, mlx
        case appleSilicon = "apple_silicon"
        case osVersion = "os_version"
        case osBuild = "os_build"
        case chipName = "chip_name"
        case hardwareModel = "hardware_model"
        case mlDtypes = "ml_dtypes"
        case mlxLM = "mlx_lm"
        case metalAvailable = "metal_available"
        case metalError = "metal_error"
        case memorySizeGB = "memory_size_gb"
        case memorySizeGiB = "memory_size_gib"
        case workingSetGB = "working_set_gb"
        case workingSetGiB = "working_set_gib"
    }

    var isUsable: Bool {
        appleSilicon && mlx.ok && mlxLM.ok && metalAvailable
    }

    /// 新后端只提供十进制 GB，App 还原 bytes 后换算为 GiB；旧 `_gib` 仅兼容读取。
    var displayedMemorySizeGiB: Double? {
        if let memorySizeGB { return memorySizeGB * 1e9 / Double(1024 * 1024 * 1024) }
        return memorySizeGiB
    }

    var memorySizeLabel: String? {
        guard let value = displayedMemorySizeGiB else { return nil }
        let rounded = value.rounded()
        return abs(value - rounded) < 0.005
            ? String(format: "%.0f GB", rounded)
            : String(format: "%.2f GB", value)
    }

    /// 新后端只提供十进制 GB；旧 `_gib` 字段仅作解码兼容。
    var displayedWorkingSetGiB: Double? {
        if let workingSetGB { return workingSetGB * 1e9 / Double(1024 * 1024 * 1024) }
        return workingSetGiB
    }

    var workingSetLabel: String? {
        guard let value = displayedWorkingSetGiB else { return nil }
        return String(format: "%.2f GB", value)
    }

    var operatingSystemLabel: String {
        guard let osVersion, !osVersion.isEmpty else { return platform }
        if let osBuild, !osBuild.isEmpty {
            return "macOS \(osVersion) (\(osBuild))"
        }
        return "macOS \(osVersion)"
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

extension RuntimePhase {
    var diagnosticLabel: String {
        switch self {
        case .checking: return "检查中"
        case .ready: return "可用"
        case .unavailable: return "不可用"
        }
    }
}

extension WorkerPhase {
    var diagnosticLabel: String {
        switch self {
        case .idle: return "空闲"
        case .starting: return "启动中"
        case .ready: return "就绪"
        case .running: return "运行中"
        case .stopping: return "停止中"
        case .failed: return "失败"
        }
    }
}

/// 生成可直接粘贴到 Issue 的脱敏 Markdown；只使用白名单字段，不带日志和任意错误文本。
enum BackendDiagnostics {
    static func markdown(
        runtime: RuntimeStatus,
        inference: WorkerStatus,
        training: WorkerStatus,
        appVersion: String,
        appBuild: String,
        systemVersionFallback: String,
        generatedAt: Date = Date()
    ) -> String {
        let report = runtime.report
        var lines = [
            "### Preen 环境信息",
            "",
            "- Preen: \(appVersion) (\(appBuild))",
            "- 系统: \(report?.operatingSystemLabel ?? systemVersionFallback)",
        ]
        if let chipName = report?.chipName {
            lines.append("- 芯片: \(chipName)")
        }
        if let hardwareModel = report?.hardwareModel {
            lines.append("- 硬件标识: \(hardwareModel)")
        }
        if let machine = report?.machine {
            lines.append("- 架构: \(machine)")
        }
        if let memory = report?.memorySizeLabel {
            lines.append("- 统一内存: \(memory)")
        }
        if let workingSet = report?.workingSetLabel {
            lines.append("- MLX 建议工作集上限: \(workingSet)")
        }
        if let report {
            lines.append(contentsOf: [
                "- Python: \(report.python)",
                "- MLX: \(moduleSummary(report.mlx))",
                "- MLX-LM: \(moduleSummary(report.mlxLM))",
                "- NumPy: \(moduleSummary(report.numpy))",
                "- ml-dtypes: \(moduleSummary(report.mlDtypes))",
                "- Metal: \(report.metalAvailable ? "可用" : "不可用")",
            ])
        }
        lines.append(contentsOf: [
            "- 运行时检查: \(runtime.phase.diagnosticLabel)",
            "- 推理服务: \(inference.phase.diagnosticLabel)",
            "- 训练任务: \(training.phase.diagnosticLabel)",
            "- 生成时间: \(timestamp(generatedAt))",
        ])
        return lines.joined(separator: "\n")
    }

    private static func moduleSummary(_ module: DoctorModule) -> String {
        guard module.ok else { return "不可用" }
        return module.version ?? "可用"
    }

    private static func timestamp(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }
}
