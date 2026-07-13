import Foundation

enum RunRepositoryError: LocalizedError {
    case unsupportedSchema(Int)

    var errorDescription: String? {
        switch self {
        case .unsupportedSchema(let schema):
            return "不支持的训练记录版本: \(schema)"
        }
    }
}

actor RunRepository {
    static let runFilename = "run.json"
    static let eventsFilename = "events.jsonl"
    static let stderrFilename = "stderr.log"

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
            run.failureMessage = "App 上次退出时训练尚未结束"
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
