import XCTest
@testable import Preen

final class BuiltinDatasetTests: XCTestCase {
    private var resourceDirectory: URL {
        let root = PythonResolver.repoRoot ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return root.appendingPathComponent("macos/Preen/Resources/Datasets/NekoQA200")
    }

    func testManifestDataLicenseAndNoticeValidateTogether() throws {
        let dataset = try BuiltinDataset.load(directory: resourceDirectory)
        XCTAssertEqual(dataset.manifest.id, "builtin:nekoqa_200")
        XCTAssertEqual(dataset.manifest.subsetVersion, "1.1.0-1")
        XCTAssertEqual(dataset.manifest.sampleCount, 200)
        XCTAssertEqual(dataset.manifest.sha256, "435f9a3ac9d5b1151fb917955fe94180a82da8602a23a0d3a84dd105dcc5939f")
        XCTAssertEqual(dataset.manifest.recommendedTemplate, "qa")
    }

    func testBuiltApplicationContainsValidatedDataset() throws {
        let dataset = try BuiltinDataset.nekoQA200()
        XCTAssertEqual(dataset.manifest.id, "builtin:nekoqa_200")
        XCTAssertEqual(dataset.manifest.sampleCount, 200)
    }

    func testApplyingBuiltinDatasetWritesRunProvenanceAndSemanticDefaults() throws {
        let dataset = try BuiltinDataset.nekoQA200()
        var config = TrainingConfig.defaultConfig
        dataset.apply(to: &config)
        XCTAssertEqual(config.dataPath, dataset.dataURL.path)
        XCTAssertEqual(config.template, .qa)
        XCTAssertEqual(config.ctxLen, 512)
        XCTAssertEqual(config.persisted.datasetSource, "builtin:nekoqa_200")
        XCTAssertEqual(config.persisted.datasetVersion, "1.1.0-1")
        XCTAssertEqual(config.persisted.datasetSHA256, dataset.manifest.sha256)

        let bundledPath = dataset.dataURL.standardizedFileURL.path
        XCTAssertEqual(config.dataPath, bundledPath)

        config.markDataAsUserSelected(path: "/tmp/custom.jsonl")
        XCTAssertNil(config.datasetSource)
        XCTAssertNil(config.datasetVersion)
        XCTAssertNil(config.datasetSHA256)
    }
}
