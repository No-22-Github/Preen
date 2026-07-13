import XCTest
@testable import Preen

final class StateMetadataTests: XCTestCase {
    func testExistingNekoQAMetadata() throws {
        let projectRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = projectRoot.appendingPathComponent("output/nekoqa_state.meta.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("本地 output/nekoqa_state.meta.json 不存在")
        }
        let metadata = try StateMetadata.load(from: url)
        XCTAssertEqual(metadata.formatVersion, 1)
        XCTAssertEqual(metadata.template, "qa")
        XCTAssertGreaterThan(metadata.result.epochsRun, 0)
        XCTAssertGreaterThan(metadata.dataSHA256.count, 10)
    }

    func testImportedRecordDoesNotInventTrainingConfig() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let state = root.appendingPathComponent("external.npz")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data([0]).write(to: state)
        let repository = RunRepository(rootURL: root.appendingPathComponent("runs"))
        let run = try await repository.registerImportedState(stateURL: state)
        XCTAssertEqual(run.kind, .imported)
        XCTAssertEqual(run.status, .completed)
        XCTAssertNil(run.config)
        XCTAssertEqual(run.artifacts.statePath, state.path)
    }

    func testSuccessAndFailureFixturesRemainVisible() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = RunRepository(rootURL: root)
        let success = TrainingRun(status: .completed, createdAt: Date(timeIntervalSince1970: 1))
        var failure = TrainingRun(status: .failed, createdAt: Date(timeIntervalSince1970: 2))
        failure.failureMessage = "fixture failure"
        _ = try await repository.create(success)
        _ = try await repository.create(failure)
        let runs = await repository.scan()
        XCTAssertEqual(Set(runs.map(\.status)), Set([.completed, .failed]))
        XCTAssertEqual(runs.first?.failureMessage, "fixture failure")
    }
}
