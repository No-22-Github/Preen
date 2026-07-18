import CryptoKit
import Foundation

struct BuiltinDatasetManifest: Decodable, Equatable {
    let id: String
    let displayName: String
    let subsetVersion: String
    let dataFile: String
    let sampleCount: Int
    let sha256: String
    let source: Source
    let license: String
    let recommendedTemplate: String
    let contentReviewedAt: String

    struct Source: Decodable, Equatable {
        let name: String
        let repositoryOwner: String
        let citationAuthor: String
        let url: String

        enum CodingKeys: String, CodingKey {
            case name, url
            case repositoryOwner = "repository_owner"
            case citationAuthor = "citation_author"
        }
    }

    enum CodingKeys: String, CodingKey {
        case id, source, license, sha256
        case displayName = "display_name"
        case subsetVersion = "subset_version"
        case dataFile = "data_file"
        case sampleCount = "sample_count"
        case recommendedTemplate = "recommended_template"
        case contentReviewedAt = "content_reviewed_at"
    }
}

struct BuiltinDataset: Equatable {
    let directoryURL: URL
    let dataURL: URL
    let manifest: BuiltinDatasetManifest

    enum ValidationError: LocalizedError {
        case missingResource(String)
        case invalidManifest
        case sampleCount(Int)
        case checksum(expected: String, actual: String)

        var errorDescription: String? {
            switch self {
            case .missingResource(let name): return L10n.format("内置数据资源缺失：%@", name)
            case .invalidManifest: return L10n.string("内置数据 manifest 无法读取")
            case .sampleCount(let count): return L10n.format("内置数据应为 200 条，实际为 %d 条", count)
            case .checksum(let expected, let actual):
                return L10n.format("内置数据校验失败：预期 %@，实际 %@", expected, actual)
            }
        }
    }

    static func nekoQA200(bundle: Bundle = .main) throws -> BuiltinDataset {
        guard let resources = bundle.resourceURL else {
            throw ValidationError.missingResource("NekoQA200")
        }
        let candidates = [
            resources.appendingPathComponent("Datasets/NekoQA200", isDirectory: true),
            resources.appendingPathComponent("Resources/Datasets/NekoQA200", isDirectory: true),
            resources.appendingPathComponent("NekoQA200", isDirectory: true),
            // Xcode 26 flattens files discovered under a synchronized Resources
            // group into the built product's Resources root.
            resources,
        ]
        if let directory = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            return try load(directory: directory)
        }
        let enumerator = FileManager.default.enumerator(
            at: resources,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        if let manifestURL = enumerator?.compactMap({ $0 as? URL }).first(where: {
            $0.lastPathComponent == "manifest.json" && $0.deletingLastPathComponent().lastPathComponent == "NekoQA200"
        }) {
            return try load(directory: manifestURL.deletingLastPathComponent())
        }
        throw ValidationError.missingResource("NekoQA200/manifest.json")
    }

    static func load(directory: URL) throws -> BuiltinDataset {
        let manifestURL = directory.appendingPathComponent("manifest.json")
        let licenseURL = directory.appendingPathComponent("LICENSE")
        let noticeURL = directory.appendingPathComponent("NOTICE.md")
        for url in [manifestURL, licenseURL, noticeURL] where !FileManager.default.fileExists(atPath: url.path) {
            throw ValidationError.missingResource(url.lastPathComponent)
        }
        guard let manifest = try? JSONDecoder().decode(
            BuiltinDatasetManifest.self,
            from: Data(contentsOf: manifestURL)
        ) else { throw ValidationError.invalidManifest }
        let dataURL = directory.appendingPathComponent(manifest.dataFile)
        guard FileManager.default.fileExists(atPath: dataURL.path) else {
            throw ValidationError.missingResource(manifest.dataFile)
        }
        let data = try Data(contentsOf: dataURL)
        let records = try JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        guard let records, records.count == manifest.sampleCount, records.count == 200 else {
            throw ValidationError.sampleCount(records?.count ?? 0)
        }
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard digest == manifest.sha256 else {
            throw ValidationError.checksum(expected: manifest.sha256, actual: digest)
        }
        return BuiltinDataset(directoryURL: directory, dataURL: dataURL, manifest: manifest)
    }

    func apply(to config: inout TrainingConfig) {
        config.dataPath = dataURL.path
        config.datasetSource = manifest.id
        config.datasetVersion = manifest.subsetVersion
        config.datasetSHA256 = manifest.sha256
        config.template = .qa
        config.ctxLen = 512
    }
}
