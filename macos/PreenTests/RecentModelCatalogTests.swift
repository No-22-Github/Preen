import XCTest
@testable import Preen

final class RecentModelCatalogTests: XCTestCase {
    func testSelectionPersistsAndValidationRemovesMissingDirectories() throws {
        let suiteName = "RecentModelCatalogTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("PreenRecentModels-\(UUID().uuidString)", isDirectory: true)
        let first = root.appendingPathComponent("first-model", isDirectory: true)
        let second = root.appendingPathComponent("second-model", isDirectory: true)
        try FileManager.default.createDirectory(at: first, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: second, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        var catalog = RecentModelCatalog(defaults: defaults)
        catalog.select(path: first.path, at: Date(timeIntervalSince1970: 1))
        catalog.select(path: second.path, at: Date(timeIntervalSince1970: 2))
        catalog.select(path: first.path, at: Date(timeIntervalSince1970: 3))

        XCTAssertEqual(catalog.selectedPath, first.path)
        XCTAssertEqual(catalog.entries.map(\.path), [first.path, second.path])

        var restored = RecentModelCatalog(defaults: defaults)
        XCTAssertEqual(restored.selectedPath, first.path)
        XCTAssertEqual(restored.entries.map(\.path), [first.path, second.path])

        try FileManager.default.removeItem(at: first)
        restored.validate()

        XCTAssertEqual(restored.selectedPath, "")
        XCTAssertEqual(restored.entries.map(\.path), [second.path])
    }

    func testCatalogKeepsOnlyMostRecentModels() throws {
        let suiteName = "RecentModelCatalogTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        var catalog = RecentModelCatalog(defaults: defaults, maximumCount: 2)
        catalog.select(path: "/models/one", at: Date(timeIntervalSince1970: 1))
        catalog.select(path: "/models/two", at: Date(timeIntervalSince1970: 2))
        catalog.select(path: "/models/three", at: Date(timeIntervalSince1970: 3))

        XCTAssertEqual(catalog.entries.map(\.path), ["/models/three", "/models/two"])
        XCTAssertEqual(catalog.selectedPath, "/models/three")
    }
}
