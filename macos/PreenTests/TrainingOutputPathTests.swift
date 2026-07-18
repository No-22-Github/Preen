import XCTest
@testable import Preen

final class TrainingOutputPathTests: XCTestCase {
    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("preen-output-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func writeModel(at root: URL, name: String = "rwkv model") throws -> URL {
        let model = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let config = #"{"num_hidden_layers":24,"hidden_size":1024,"head_dim":64,"num_heads":16}"#
        try config.write(to: model.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        return model
    }

    func testAutomaticPathIsSafePredictableAndResolvesCollision() throws {
        let root = try temporaryDirectory()
        let model = try writeModel(at: root)
        let data = root.appendingPathComponent("my:data.json")
        try "[]".write(to: data, atomically: true, encoding: .utf8)
        let date = Date(timeIntervalSince1970: 1_721_297_412)

        let first = TrainingOutputPath.automaticURL(
            dataPath: data.path, modelPath: model.path, rootURL: root, date: date
        )
        XCTAssertEqual(first.lastPathComponent, "state.npz")
        XCTAssertFalse(first.deletingLastPathComponent().lastPathComponent.contains(":"))
        try FileManager.default.createDirectory(
            at: first.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let second = TrainingOutputPath.automaticURL(
            dataPath: data.path, modelPath: model.path, rootURL: root, date: date
        )
        XCTAssertEqual(
            second.deletingLastPathComponent().lastPathComponent,
            first.deletingLastPathComponent().lastPathComponent + "-2"
        )
    }

    func testAutomaticPathTracksSourcesButManualPathStaysLocked() throws {
        let root = try temporaryDirectory()
        let modelA = try writeModel(at: root, name: "model-a")
        let modelB = try writeModel(at: root, name: "model-b")
        var config = TrainingConfig.defaultConfig
        config.dataPath = root.appendingPathComponent("cats.json").path
        config.modelPath = modelA.path
        config.refreshAutomaticOutputPath(date: Date(timeIntervalSince1970: 100), rootURL: root)
        let first = config.outPath

        config.epochs = 99
        config.refreshAutomaticOutputPath(date: Date(timeIntervalSince1970: 200), rootURL: root)
        XCTAssertEqual(config.outPath, first)
        config.modelPath = modelB.path
        config.refreshAutomaticOutputPath(date: Date(timeIntervalSince1970: 200), rootURL: root)
        XCTAssertNotEqual(config.outPath, first)

        config.markOutputPathManual(root.appendingPathComponent("chosen.npz").path)
        let manual = config.outPath
        config.dataPath = root.appendingPathComponent("dogs.json").path
        config.refreshAutomaticOutputPath(date: Date(timeIntervalSince1970: 300), rootURL: root)
        XCTAssertEqual(config.outPath, manual)
    }

    func testEstimateUsesRWKV7StateShapeAndOptionalPTH() throws {
        let root = try temporaryDirectory()
        let model = try writeModel(at: root)
        let stateOnly = try TrainingOutputPath.estimate(modelPath: model.path, exportPth: false)
        XCTAssertEqual(stateOnly.stateBytes, 24 * 16 * 64 * 64 * 4)
        XCTAssertEqual(stateOnly.pthBytes, 0)
        let withPth = try TrainingOutputPath.estimate(modelPath: model.path, exportPth: true)
        XCTAssertEqual(withPth.pthBytes, withPth.stateBytes)
        XCTAssertGreaterThan(withPth.requiredFreeBytes, withPth.stateBytes * 2)
    }

    func testPreflightRejectsExistingArtifactsAndInsufficientSpace() throws {
        let root = try temporaryDirectory()
        let model = try writeModel(at: root)
        var config = TrainingConfig.defaultConfig
        config.modelPath = model.path
        config.dataPath = root.appendingPathComponent("data.json").path
        config.markOutputPathManual(root.appendingPathComponent("state.npz").path)
        try "existing".write(toFile: config.outPath, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try TrainingOutputPath.validate(config: config)) { error in
            guard case TrainingOutputPathError.destinationExists = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
        try FileManager.default.removeItem(atPath: config.outPath)
        XCTAssertThrowsError(
            try TrainingOutputPath.validate(config: config, availableCapacity: { _ in 1 })
        ) { error in
            guard case TrainingOutputPathError.insufficientSpace = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }
}
