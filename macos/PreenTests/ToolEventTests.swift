import XCTest
@testable import Preen

final class ToolEventTests: XCTestCase {
    func testCompletedDatasetEventDecodesTypedResult() throws {
        let json = """
        {"type":"completed","tool":"dataset_preview","timestamp":1,"progress":1,
         "result":{"detection":{"schema":"bare_qa","prompt_keys":["q"],"response_keys":["a"],"confidence":1,"total_sampled":2},
         "result":{"template":"qa","turn_policy":"first","dropped_system":0,"dropped_other":0,"qa_degradation_hint":false,"record_count":2},
         "preview":[],"inspection":null,
         "pagination":{"cache_path":"/tmp/preview.jsonl","total":42,"page_size":20,"page_count":3},
         "available_keys":["q","a"],"turn_policy":"first"}}
        """
        let event = try JSONDecoder().decode(ToolEvent.self, from: Data(json.utf8))
        XCTAssertEqual(event.type, .completed)
        let result = try XCTUnwrap(event.result).decode(DatasetPreviewResult.self)
        XCTAssertEqual(result.detection.schema, "bare_qa")
        XCTAssertEqual(result.result?.recordCount, 2)
        XCTAssertEqual(result.pagination?.pageCount, 3)
    }

    func testProgressEventDecodes() throws {
        let json = """
        {"type":"progress","tool":"model_conversion","timestamp":2,
         "phase":"convert","message":"转换张量","current":50,"total":100,"progress":0.5}
        """
        let event = try JSONDecoder().decode(ToolEvent.self, from: Data(json.utf8))
        XCTAssertEqual(event.type, .progress)
        XCTAssertEqual(event.progress, 0.5)
        XCTAssertEqual(event.current, 50)
    }

    func testDatasetPreviewPageDecodesOnlyCurrentPage() throws {
        let json = """
        {"type":"completed","tool":"dataset_preview_page","timestamp":3,
         "result":{"preview":[],"page":3,"page_size":20,"page_count":3,"total":42}}
        """
        let event = try JSONDecoder().decode(ToolEvent.self, from: Data(json.utf8))
        let result = try XCTUnwrap(event.result).decode(DatasetPreviewPageResult.self)
        XCTAssertEqual(result.page, 3)
        XCTAssertEqual(result.total, 42)
        XCTAssertTrue(result.preview.isEmpty)
    }
}

@MainActor
final class ToolboxStoreTests: XCTestCase {
    func testModelSourceRequiresExistingPthFile() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let store = ToolboxStore()
        let wrongType = root.appendingPathComponent("model.json")
        try Data().write(to: wrongType)

        XCTAssertFalse(store.selectModelSource(path: wrongType.path))
        XCTAssertEqual(store.errorMessage, "源模型必须是 .pth 文件")
        XCTAssertTrue(store.modelSourcePath.isEmpty)
        XCTAssertFalse(store.canConvertModel)

        let missingModel = root.appendingPathComponent("missing.pth")
        XCTAssertFalse(store.selectModelSource(path: missingModel.path))
        XCTAssertEqual(store.errorMessage, "找不到所选的 .pth 模型文件")

        let validModel = root.appendingPathComponent("RWKV.PTH")
        try Data().write(to: validModel)
        XCTAssertTrue(store.selectModelSource(path: validModel.path))
        XCTAssertEqual(store.modelSourcePath, validModel.path)
        XCTAssertNil(store.errorMessage)

        store.modelOutputPath = root.appendingPathComponent("converted").path
        XCTAssertTrue(store.canConvertModel)
    }

    func testModelOutputConfirmationOnlyForNonEmptyDirectory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let store = ToolboxStore()
        store.modelOutputPath = root.path
        XCTAssertFalse(store.modelOutputRequiresConfirmation)

        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        XCTAssertFalse(store.modelOutputRequiresConfirmation)

        try Data("existing".utf8).write(to: root.appendingPathComponent("config.json"))
        XCTAssertTrue(store.modelOutputRequiresConfirmation)
    }

    func testNavigationClearsTransientErrorButKeepsInputs() {
        let store = ToolboxStore()
        store.datasetSourcePath = "/tmp/dataset.jsonl"
        store.previewDataset(modelPath: "")
        XCTAssertNotNil(store.errorMessage)

        store.clearPresentationForNavigation()

        XCTAssertNil(store.errorMessage)
        XCTAssertEqual(store.datasetSourcePath, "/tmp/dataset.jsonl")
    }
}
