import Foundation

enum TrainingOutputPathMode: String, Equatable {
    case automatic
    case manual
}

enum TrainingOutputPathError: LocalizedError, Equatable {
    case invalidModelConfig
    case destinationExists(String)
    case parentNotWritable(String)
    case insufficientSpace(required: Int64, available: Int64)

    var errorDescription: String? {
        switch self {
        case .invalidModelConfig:
            return L10n.string("无法从模型 config.json 估算 State 产物大小")
        case .destinationExists(let path):
            return L10n.format("目标已存在，不会覆盖：%@。请选择其他位置。", path)
        case .parentNotWritable(let path):
            return L10n.format("输出目录不可写：%@。请选择其他位置。", path)
        case .insufficientSpace(let required, let available):
            return L10n.format(
                "空间不足：产物与原子写入预计需要 %@，当前可用 %@。请选择其他磁盘或关闭 PTH 导出。",
                ByteCountFormatter.string(fromByteCount: required, countStyle: .file),
                ByteCountFormatter.string(fromByteCount: available, countStyle: .file)
            )
        }
    }
}

struct TrainingOutputEstimate: Equatable {
    let stateBytes: Int64
    let metadataBytes: Int64
    let pthBytes: Int64

    var requiredFreeBytes: Int64 {
        // NPZ/PTH store fp32 State tensors. Add 10% container overhead and keep
        // metadata's temporary atomic-write copy in the peak requirement.
        let artifacts = stateBytes + pthBytes
        return Int64((Double(artifacts) * 1.10).rounded(.up)) + metadataBytes * 2
    }
}

enum TrainingOutputPath {
    private static let metadataAllowance: Int64 = 1 * 1024 * 1024

    static func automaticURL(
        dataPath: String,
        modelPath: String,
        rootURL: URL = PythonResolver.statesDirectory,
        date: Date = Date(),
        fileManager: FileManager = .default
    ) -> URL {
        let dataName = safeComponent(
            URL(fileURLWithPath: dataPath).deletingPathExtension().lastPathComponent,
            fallback: "data"
        )
        let modelName = safeComponent(
            URL(fileURLWithPath: modelPath).lastPathComponent,
            fallback: "model"
        )
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let base = "\(dataName)-\(modelName)-\(formatter.string(from: date))"

        var suffix = 1
        while true {
            let directoryName = suffix == 1 ? base : "\(base)-\(suffix)"
            let directory = rootURL.appendingPathComponent(directoryName, isDirectory: true)
            let state = directory.appendingPathComponent("state.npz")
            if !fileManager.fileExists(atPath: directory.path),
               !fileManager.fileExists(atPath: state.path) {
                return state
            }
            suffix += 1
        }
    }

    static func estimate(modelPath: String, exportPth: Bool) throws -> TrainingOutputEstimate {
        let configURL = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let layers = integer(json["num_hidden_layers"]), layers > 0,
              let hiddenSize = integer(json["hidden_size"]), hiddenSize > 0
        else { throw TrainingOutputPathError.invalidModelConfig }

        let headDim = integer(json["head_dim"]) ?? 64
        let heads = integer(json["num_heads"]) ?? hiddenSize / headDim
        guard headDim > 0, heads > 0 else { throw TrainingOutputPathError.invalidModelConfig }
        let raw = Int64(layers) * Int64(heads) * Int64(headDim) * Int64(headDim) * 4
        return TrainingOutputEstimate(
            stateBytes: raw,
            metadataBytes: metadataAllowance,
            pthBytes: exportPth ? raw : 0
        )
    }

    static func validate(
        config: TrainingConfig,
        fileManager: FileManager = .default,
        availableCapacity: ((URL) -> Int64?)? = nil
    ) throws -> TrainingOutputEstimate {
        let stateURL = URL(fileURLWithPath: config.outPath).standardizedFileURL
        let metadataURL = stateURL.deletingPathExtension().appendingPathExtension("meta.json")
        let pthURL = config.pthOutPath.isEmpty
            ? stateURL.deletingPathExtension().appendingPathExtension("pth")
            : URL(fileURLWithPath: config.pthOutPath).standardizedFileURL
        var destinations = [stateURL, metadataURL]
        if config.exportPth { destinations.append(pthURL) }
        if let existing = destinations.first(where: { fileManager.fileExists(atPath: $0.path) }) {
            throw TrainingOutputPathError.destinationExists(existing.path)
        }

        let parent = stateURL.deletingLastPathComponent()
        let writableAncestor = nearestExistingDirectory(from: parent, fileManager: fileManager)
        guard let writableAncestor, fileManager.isWritableFile(atPath: writableAncestor.path) else {
            throw TrainingOutputPathError.parentNotWritable(parent.path)
        }
        var pthWritableAncestor: URL?
        if config.exportPth {
            let pthParent = pthURL.deletingLastPathComponent()
            let pthAncestor = nearestExistingDirectory(from: pthParent, fileManager: fileManager)
            guard let pthAncestor, fileManager.isWritableFile(atPath: pthAncestor.path) else {
                throw TrainingOutputPathError.parentNotWritable(pthParent.path)
            }
            pthWritableAncestor = pthAncestor
        }

        let estimate = try estimate(modelPath: config.modelPath, exportPth: config.exportPth)
        let stateRequirement = Int64((Double(estimate.stateBytes) * 1.10).rounded(.up))
            + estimate.metadataBytes * 2
        if let pthWritableAncestor,
           volumeIdentifier(for: writableAncestor) != volumeIdentifier(for: pthWritableAncestor) {
            try requireCapacity(
                stateRequirement,
                at: writableAncestor,
                availableCapacity: availableCapacity
            )
            try requireCapacity(
                Int64((Double(estimate.pthBytes) * 1.10).rounded(.up)),
                at: pthWritableAncestor,
                availableCapacity: availableCapacity
            )
        } else {
            try requireCapacity(
                estimate.requiredFreeBytes,
                at: writableAncestor,
                availableCapacity: availableCapacity
            )
        }
        return estimate
    }

    private static func safeComponent(_ raw: String, fallback: String) -> String {
        let invalid = CharacterSet(charactersIn: "/:").union(.controlCharacters)
        let pieces = raw.components(separatedBy: invalid).filter { !$0.isEmpty }
        let joined = pieces.joined(separator: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: #"\s+"#, with: "-", options: .regularExpression)
        return String((joined.isEmpty ? fallback : joined).prefix(64))
    }

    private static func nearestExistingDirectory(
        from url: URL,
        fileManager: FileManager
    ) -> URL? {
        var candidate = url.standardizedFileURL
        while candidate.path != "/" && !fileManager.fileExists(atPath: candidate.path) {
            candidate.deleteLastPathComponent()
        }
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: candidate.path, isDirectory: &isDirectory),
              isDirectory.boolValue else { return nil }
        return candidate
    }

    private static func integer(_ value: Any?) -> Int? {
        if let value = value as? Int { return value }
        if let value = value as? NSNumber { return value.intValue }
        return nil
    }

    private static func volumeIdentifier(for url: URL) -> AnyHashable? {
        guard let values = try? url.resourceValues(forKeys: [.volumeIdentifierKey]) else {
            return nil
        }
        return values.volumeIdentifier as? AnyHashable
    }

    private static func requireCapacity(
        _ required: Int64,
        at url: URL,
        availableCapacity: ((URL) -> Int64?)?
    ) throws {
        let capacity = availableCapacity?(url)
            ?? (try? url.resourceValues(
                forKeys: [.volumeAvailableCapacityForImportantUsageKey]
            ).volumeAvailableCapacityForImportantUsage)
        if let capacity, capacity < required {
            throw TrainingOutputPathError.insufficientSpace(required: required, available: capacity)
        }
    }
}
