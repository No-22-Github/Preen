import Foundation

enum RunRepositoryError: LocalizedError {
    case unsupportedSchema(Int)
    case runStillActive

    var errorDescription: String? {
        switch self {
        case .unsupportedSchema(let schema):
            return L10n.format("不支持的训练记录版本：%lld", schema)
        case .runStillActive:
            return L10n.string("训练仍在运行，请先取消训练再删除记录")
        }
    }
}

actor RunRepository {
    static let runFilename = "run.json"
    static let eventsFilename = "events.jsonl"
    static let stderrFilename = "stderr.log"
    static let comparisonsFilename = "comparisons.jsonl"

    let rootURL: URL
    private let fileManager: FileManager
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(rootURL: URL = PythonResolver.runsDirectory, fileManager: FileManager = .default) {
        self.rootURL = rootURL
        self.fileManager = fileManager
        self.encoder = JSONEncoder()
        self.decoder = JSONDecoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
    }

    func create(_ run: TrainingRun) throws -> URL {
        let directory = directoryURL(for: run.id)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        fileManager.createFile(
            atPath: directory.appendingPathComponent(Self.eventsFilename).path,
            contents: nil
        )
        fileManager.createFile(
            atPath: directory.appendingPathComponent(Self.stderrFilename).path,
            contents: nil
        )
        try save(run)
        return directory
    }

    func save(_ run: TrainingRun) throws {
        let directory = directoryURL(for: run.id)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = try encoder.encode(run)
        try Self.writeAtomically(data, to: directory.appendingPathComponent(Self.runFilename), fileManager: fileManager)
    }

    func load(id: UUID) throws -> TrainingRun {
        try load(from: directoryURL(for: id).appendingPathComponent(Self.runFilename))
    }

    func loadEvents(id: UUID) -> [TrainEvent] {
        let url = directoryURL(for: id).appendingPathComponent(Self.eventsFilename)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        let decoder = JSONDecoder()
        return text.split(whereSeparator: \.isNewline).compactMap { line in
            try? decoder.decode(TrainEvent.self, from: Data(line.utf8))
        }
    }

    func loadStderr(id: UUID) -> String {
        let url = directoryURL(for: id).appendingPathComponent(Self.stderrFilename)
        return (try? String(contentsOf: url, encoding: .utf8)) ?? ""
    }

    func appendComparison(runID: UUID, record: SavedComparison) throws {
        _ = try load(id: runID)  // 只允许写入真实存在的 run。
        let url = directoryURL(for: runID).appendingPathComponent(Self.comparisonsFilename)
        let lineEncoder = JSONEncoder()
        lineEncoder.dateEncodingStrategy = .iso8601
        lineEncoder.outputFormatting = [.sortedKeys]
        var line = try lineEncoder.encode(record)
        line.append(0x0A)
        if !fileManager.fileExists(atPath: url.path) {
            try line.write(to: url, options: .atomic)
            return
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: line)
    }

    func loadComparisons(runID: UUID) -> [SavedComparison] {
        let url = directoryURL(for: runID).appendingPathComponent(Self.comparisonsFilename)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return text.split(whereSeparator: \.isNewline).compactMap { line in
            try? decoder.decode(SavedComparison.self, from: Data(line.utf8))
        }
    }

    func registerImportedState(stateURL: URL, metadataURL: URL? = nil) throws -> TrainingRun {
        let now = Date()
        var run = TrainingRun(kind: .imported, status: .completed, createdAt: now)
        run.finishedAt = now
        run.artifacts.statePath = stateURL.path
        if let metadataURL, fileManager.fileExists(atPath: metadataURL.path) {
            run.artifacts.metadataPath = metadataURL.path
            if let metadata = try? StateMetadata.load(from: metadataURL) {
                run.summary.actualEpochs = metadata.result?.epochsRun
                run.summary.finalLoss = metadata.result?.finalLoss
                run.summary.heldOutLoss = metadata.result?.bestHeldOutLoss
                run.summary.bestHeldOutEpoch = metadata.result?.bestHeldOutEpoch
                run.summary.stateStd = metadata.result?.finalStateStd
                run.summary.elapsedSeconds = metadata.result?.elapsed
                run.summary.dataHash = metadata.dataSHA256.isEmpty ? nil : metadata.dataSHA256
                run.summary.trainSamples = metadata.dataStats?.trainSamples
                run.summary.heldOutSamples = metadata.dataStats?.heldOutSamples
                run.summary.truncatedSamples = metadata.dataStats?.truncated
                run.summary.droppedSamples = metadata.dataStats?.droppedSamples
                run.summary.targetFullyTruncated = metadata.dataStats?.targetFullyTruncated
            }
        }
        _ = try create(run)
        return run
    }

    func setPthArtifact(runID: UUID, path: String) throws -> TrainingRun {
        var run = try load(id: runID)
        run.artifacts.pthPath = path
        run.updatedAt = Date()
        try save(run)
        return run
    }

    /// 只删除 App 管理的记录目录；State/PTH/checkpoint 等外部产物不在此目录内。
    func delete(id: UUID) throws {
        let run = try load(id: id)
        guard run.status.isTerminal else { throw RunRepositoryError.runStillActive }
        try fileManager.removeItem(at: directoryURL(for: id))
    }

    func scan() -> [TrainingRun] {
        guard let directories = try? fileManager.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        return directories.compactMap { directory in
            try? load(from: directory.appendingPathComponent(Self.runFilename))
        }
        .sorted { $0.createdAt > $1.createdAt }
    }

    func markUnfinishedRunsInterrupted(at date: Date = Date()) throws -> [TrainingRun] {
        var updated: [TrainingRun] = []
        for var run in scan() where !run.status.isTerminal {
            run.status = .interrupted
            run.updatedAt = date
            run.finishedAt = date
            run.failureMessage = L10n.string("App 上次退出时训练尚未结束")
            try save(run)
            updated.append(run)
        }
        return updated
    }

    func directoryURL(for id: UUID) -> URL {
        rootURL.appendingPathComponent(id.uuidString.lowercased(), isDirectory: true)
    }

    private func load(from url: URL) throws -> TrainingRun {
        let run = try decoder.decode(TrainingRun.self, from: Data(contentsOf: url))
        guard run.schema == TrainingRun.schemaVersion else {
            throw RunRepositoryError.unsupportedSchema(run.schema)
        }
        return run
    }

    private static func writeAtomically(_ data: Data, to destination: URL, fileManager: FileManager) throws {
        let temporary = destination.deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).\(UUID().uuidString).tmp")
        try data.write(to: temporary)
        do {
            if fileManager.fileExists(atPath: destination.path) {
                _ = try fileManager.replaceItemAt(destination, withItemAt: temporary)
            } else {
                try fileManager.moveItem(at: temporary, to: destination)
            }
        } catch {
            try? fileManager.removeItem(at: temporary)
            throw error
        }
    }
}
