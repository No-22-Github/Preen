import XCTest
@testable import Preen

final class DatasetPreflightCacheTests: XCTestCase {
    private var directory: URL!
    private var modelDirectory: URL!
    private var dataURL: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("PreenPreflightCache-\(UUID().uuidString)", isDirectory: true)
        modelDirectory = directory.appendingPathComponent("model", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        dataURL = directory.appendingPathComponent("data.jsonl")
        try Data("{\"prompt\":\"q\",\"response\":\"a\"}\n".utf8).write(to: dataURL)
        try Data("{}".utf8).write(to: modelDirectory.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(to: modelDirectory.appendingPathComponent("tokenizer.json"))
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func key(
        ctx: Int = 512,
        template: String = "qa",
        turn: String = "first",
        prompt: String = "",
        response: String = "",
        training: Bool = true
    ) -> DatasetPreflightCacheKey {
        DatasetPreflightCache.makeKey(
            modelPath: modelDirectory.path,
            dataPath: dataURL.path,
            ctxLen: ctx,
            template: template,
            turnPolicy: turn,
            promptKey: prompt,
            responseKey: response,
            trainingDataRoute: training
        )
    }

    func testSameInputsProduceStableKey() {
        XCTAssertEqual(key(), key())
    }

    func testEveryRenderingInputInvalidatesKey() {
        let baseline = key().value
        XCTAssertNotEqual(key(ctx: 256).value, baseline)
        XCTAssertNotEqual(key(template: "instruction").value, baseline)
        XCTAssertNotEqual(key(turn: "all").value, baseline)
        XCTAssertNotEqual(key(prompt: "question").value, baseline)
        XCTAssertNotEqual(key(response: "answer").value, baseline)
        XCTAssertNotEqual(key(training: false).value, baseline)
    }

    func testDataModificationTimeInvalidatesKey() throws {
        let baseline = key().value
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 2_000_000_000)],
            ofItemAtPath: dataURL.path
        )
        XCTAssertNotEqual(key().value, baseline)
    }

    func testTokenizerModificationTimeInvalidatesKey() throws {
        let baseline = key().value
        let tokenizer = modelDirectory.appendingPathComponent("tokenizer.json")
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 2_000_000_000)],
            ofItemAtPath: tokenizer.path
        )
        XCTAssertNotEqual(key().value, baseline)
    }

    func testImporterSidecarModificationInvalidatesKey() throws {
        let baseline = key().value
        let sidecar = dataURL.appendingPathExtension("import.json")
        try Data("{\"result\":{\"template\":\"qa\"}}".utf8).write(to: sidecar)
        XCTAssertNotEqual(key().value, baseline)

        let withSidecar = key().value
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 2_000_000_000)],
            ofItemAtPath: sidecar.path
        )
        XCTAssertNotEqual(key().value, withSidecar)
    }

    func testResultRoundTripsThroughSharedCache() {
        let cacheKey = key()
        defer {
            try? FileManager.default.removeItem(at: cacheKey.previewURL)
            try? FileManager.default.removeItem(at: cacheKey.resultURL)
        }
        let result = DatasetPreviewResult(
            detection: DatasetDetectionResult(
                schema: "standard", promptKeys: ["prompt"], responseKeys: ["response"],
                confidence: 1, totalSampled: 1
            ),
            result: nil,
            preview: [],
            inspection: DatasetInspectionResult(
                total: 1, valid: 1, truncated: 0, targetFullyTruncated: 0,
                minTokens: 4, meanTokens: 4, p95Tokens: 4, maxTokens: 4,
                ctxLen: 512, template: "qa"
            ),
            pagination: nil,
            availableKeys: ["prompt", "response"],
            turnPolicy: "first"
        )
        DatasetPreflightCache.save(result, for: cacheKey)
        XCTAssertEqual(DatasetPreflightCache.load(cacheKey), result)
    }
}
