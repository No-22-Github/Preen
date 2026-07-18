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
        XCTAssertGreaterThan(metadata.result?.epochsRun ?? 0, 0)
        XCTAssertGreaterThan(metadata.dataSHA256.count, 10)
    }

    func testLegacyV1DerivesModelNameAndKeepsOptionalFieldsCompatible() throws {
        let json = """
        {
          "format_version": 1,
          "created_at": 1,
          "model": "/models/rwkv-old",
          "data": "/data/train.json",
          "data_sha256": "abc",
          "template": "instruction"
        }
        """
        let metadata = try JSONDecoder().decode(StateMetadata.self, from: Data(json.utf8))
        XCTAssertEqual(metadata.modelPath, "/models/rwkv-old")
        XCTAssertEqual(metadata.modelName, "rwkv-old")
        XCTAssertEqual(metadata.template, "instruction")
        XCTAssertNil(metadata.stateFormat)
        XCTAssertNil(metadata.result)
    }

    func testMinimalV2ContractDecodesWithoutLegacyTrainingSummary() throws {
        let json = """
        {
          "format_version": 2,
          "created_at": 1784300000,
          "model_name": "rwkv7-g1d-0.4b",
          "model_path": "/models/rwkv7-g1d-0.4b",
          "template": "qa",
          "data_sha256": "0123456789",
          "state_format": "npz",
          "state_dtype": "float32"
        }
        """
        let metadata = try JSONDecoder().decode(StateMetadata.self, from: Data(json.utf8))
        XCTAssertEqual(metadata.formatVersion, 2)
        XCTAssertEqual(metadata.modelName, "rwkv7-g1d-0.4b")
        XCTAssertEqual(metadata.stateFormat, "npz")
        XCTAssertEqual(metadata.stateDtype, "float32")
        XCTAssertNil(metadata.config)
        XCTAssertNil(metadata.result)
    }

    @MainActor
    func testRecordSuggestionAppliesUntilUserExplicitlyOverrides() {
        let store = ChatStore()
        store.prepareSessionReplacement(
            statePath: "/tmp/first.npz",
            suggestedTemplate: .instruction,
            source: .trainingRecord
        )
        XCTAssertEqual(store.sessionConfig.template, .instruction)
        XCTAssertEqual(store.sessionConfigSource, .trainingRecord)

        var userConfig = store.sessionConfig
        userConfig.template = .raw
        store.applySessionConfig(userConfig)
        store.prepareSessionReplacement(
            statePath: "/tmp/second.npz",
            suggestedTemplate: .qa,
            source: .stateMetadata
        )
        XCTAssertEqual(store.sessionConfig.template, .raw)
        XCTAssertEqual(store.sessionConfigSource, .user)
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
