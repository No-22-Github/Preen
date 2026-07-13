import XCTest
@testable import Preen

final class RunRepositoryTests: XCTestCase {
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
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), [RunRepository.runFilename])
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

    private var fixtureConfig: PersistedTrainingConfig {
        PersistedTrainingConfig(
            modelPath: "/models/rwkv", dataPath: "/data/train.jsonl", outputPath: "/states/a.npz",
            template: "qa", learningRate: 0.01, learningRateFloor: 0.0001, warmup: 10,
            contextLength: 512, epochs: 3, gradientClip: 1, earlyStop: true,
            earlyStopPatience: 3, testRatio: 0.1, seed: 42, cacheLimitGB: "auto"
        )
    }
}
