import XCTest
@testable import Preen

final class RunRepositoryTests: XCTestCase {
    func testSavedComparisonPersistsInRunDirectory() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = RunRepository(rootURL: root)
        let run = TrainingRun(status: .completed)
        _ = try await repository.create(run)
        let record = SavedComparison(
            prompt: "hello",
            baselineText: "plain",
            stateText: "styled",
            template: "qa",
            reasoning: true,
            think: "fast",
            genConfig: .defaultConfig,
            baseline: ComparisonMetrics(result: nil),
            withState: ComparisonMetrics(result: nil),
            createdAt: Date(timeIntervalSince1970: 1_000)
        )
        try await repository.appendComparison(runID: run.id, record: record)
        let loaded = await repository.loadComparisons(runID: run.id)
        XCTAssertEqual(loaded, [record])
    }
    private var temporaryRoot: URL!

    override func setUpWithError() throws {
        temporaryRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("PreenTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: temporaryRoot)
    }

    func testTrainingRunCodableRoundTrip() throws {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        var run = TrainingRun(id: UUID(), status: .running, createdAt: date)
        run.startedAt = date
        run.config = fixtureConfig
        run.artifacts.statePath = "/tmp/state.npz"
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        XCTAssertEqual(try decoder.decode(TrainingRun.self, from: encoder.encode(run)), run)
    }

    func testAtomicSaveReplacesExistingRecord() async throws {
        let repository = RunRepository(rootURL: temporaryRoot)
        var run = TrainingRun(config: fixtureConfig)
        _ = try await repository.create(run)
        run.status = .completed
        run.updatedAt = run.createdAt.addingTimeInterval(30)
        try await repository.save(run)
        let loaded = try await repository.load(id: run.id)
        XCTAssertEqual(loaded.status, .completed)
        let directory = temporaryRoot.appendingPathComponent(run.id.uuidString.lowercased())
        XCTAssertEqual(
            Set(try FileManager.default.contentsOfDirectory(atPath: directory.path)),
            Set([RunRepository.runFilename, RunRepository.eventsFilename, RunRepository.stderrFilename])
        )
    }

    func testScanSortsRecordsAndSkipsMalformedDirectories() async throws {
        let repository = RunRepository(rootURL: temporaryRoot)
        let older = TrainingRun(createdAt: Date(timeIntervalSince1970: 10))
        let newer = TrainingRun(createdAt: Date(timeIntervalSince1970: 20))
        _ = try await repository.create(older)
        _ = try await repository.create(newer)
        try FileManager.default.createDirectory(at: temporaryRoot.appendingPathComponent("broken"), withIntermediateDirectories: true)
        let runs = await repository.scan()
        XCTAssertEqual(runs.map(\.id), [newer.id, older.id])
    }

    func testDeleteRemovesRecordButKeepsExternalArtifacts() async throws {
        let repository = RunRepository(rootURL: temporaryRoot)
        let stateURL = temporaryRoot.deletingLastPathComponent()
            .appendingPathComponent("\(UUID().uuidString).npz")
        defer { try? FileManager.default.removeItem(at: stateURL) }
        try Data("state".utf8).write(to: stateURL)

        var run = TrainingRun(status: .completed)
        run.artifacts.statePath = stateURL.path
        _ = try await repository.create(run)

        try await repository.delete(id: run.id)

        let deletedDirectory = await repository.directoryURL(for: run.id)
        XCTAssertFalse(FileManager.default.fileExists(atPath: deletedDirectory.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: stateURL.path))
        let remaining = await repository.scan()
        XCTAssertTrue(remaining.isEmpty)
    }

    func testDeleteRejectsActiveRun() async throws {
        let repository = RunRepository(rootURL: temporaryRoot)
        let run = TrainingRun(status: .running)
        _ = try await repository.create(run)

        do {
            try await repository.delete(id: run.id)
            XCTFail("应拒绝删除运行中的记录")
        } catch let error as RunRepositoryError {
            XCTAssertEqual(
                error.errorDescription,
                L10n.string("训练仍在运行，请先取消训练再删除记录")
            )
        }
        let activeDirectory = await repository.directoryURL(for: run.id)
        XCTAssertTrue(FileManager.default.fileExists(atPath: activeDirectory.path))
    }

    private var fixtureConfig: PersistedTrainingConfig {
        PersistedTrainingConfig(
            modelPath: "/models/rwkv", dataPath: "/data/train.jsonl", outputPath: "/states/a.npz",
            template: "qa", learningRate: 0.01, learningRateFloor: 0.0001, warmup: 10,
            contextLength: 512, epochs: 3, gradientClip: 1, earlyStop: true,
            earlyStopPatience: 3, testRatio: 0.1, seed: 42, cacheLimitGB: "auto"
        )
    }
}
