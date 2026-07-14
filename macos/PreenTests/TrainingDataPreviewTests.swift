import XCTest
@testable import Preen

final class TrainingDataPreviewTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("PreenDataPreview-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func write(_ name: String, _ content: String) throws -> String {
        let url = dir.appendingPathComponent(name)
        try content.write(to: url, atomically: true, encoding: .utf8)
        return url.path
    }

    func testJSONLReadsRecordsAndFlagsMore() throws {
        let lines = (1...7).map { "{\"prompt\":\"q\($0)\",\"response\":\"a\($0)\"}" }
        let path = try write("d.jsonl", lines.joined(separator: "\n"))
        let preview = TrainingDataPreview.load(path: path, limit: 5)
        XCTAssertNil(preview.error)
        XCTAssertEqual(preview.samples.count, 5)
        XCTAssertTrue(preview.hasMore)
        // 常见字段优先:prompt 排在 response 前。
        XCTAssertEqual(preview.samples[0].fields.map(\.key), ["prompt", "response"])
        XCTAssertEqual(preview.samples[0].fields[0].value, "q1")
    }

    func testJSONArrayNoOverflow() throws {
        let path = try write("d.json", "[{\"instruction\":\"i\",\"output\":\"o\"}]")
        let preview = TrainingDataPreview.load(path: path, limit: 5)
        XCTAssertNil(preview.error)
        XCTAssertEqual(preview.samples.count, 1)
        XCTAssertFalse(preview.hasMore)
        XCTAssertEqual(preview.samples[0].fields.first?.key, "instruction")
    }

    func testJSONObjectWrappedList() throws {
        let path = try write("d.json", "{\"data\":[{\"q\":\"x\",\"a\":\"y\"}]}")
        let preview = TrainingDataPreview.load(path: path, limit: 5)
        XCTAssertEqual(preview.samples.count, 1)
        XCTAssertEqual(preview.samples[0].fields.count, 2)
    }

    func testCSVWithQuotedComma() throws {
        let csv = "question,answer\n\"hello, world\",\"hi there\"\nq2,a2\n"
        let path = try write("d.csv", csv)
        let preview = TrainingDataPreview.load(path: path, limit: 5)
        XCTAssertNil(preview.error)
        XCTAssertEqual(preview.samples.count, 2)
        let first = Dictionary(uniqueKeysWithValues: preview.samples[0].fields.map { ($0.key, $0.value) })
        XCTAssertEqual(first["question"], "hello, world")
        XCTAssertEqual(first["answer"], "hi there")
    }

    func testNestedMessagesStringified() throws {
        let path = try write("d.jsonl",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
        let preview = TrainingDataPreview.load(path: path, limit: 5)
        XCTAssertEqual(preview.samples.count, 1)
        let value = preview.samples[0].fields.first { $0.key == "messages" }?.value
        XCTAssertTrue(value?.contains("\"role\"") == true, "嵌套结构应转成 JSON 字符串")
    }

    func testEmptyPathReturnsEmpty() {
        let preview = TrainingDataPreview.load(path: "", limit: 5)
        XCTAssertTrue(preview.samples.isEmpty)
        XCTAssertNil(preview.error)
    }

    func testMissingFileReturnsError() {
        let preview = TrainingDataPreview.load(path: "/nonexistent/xyz.jsonl", limit: 5)
        XCTAssertNotNil(preview.error)
    }
}
