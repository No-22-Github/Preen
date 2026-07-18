import CryptoKit
import Foundation

struct DatasetPreflightCacheKey: Equatable {
    let value: String
    let previewURL: URL
    let resultURL: URL
}

enum DatasetPreflightCache {
    static var rootURL: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let root = caches.appendingPathComponent("Preen/DatasetPreflight", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    static func makeKey(
        modelPath: String,
        dataPath: String,
        ctxLen: Int,
        template: String,
        turnPolicy: String = "first",
        promptKey: String = "",
        responseKey: String = "",
        trainingDataRoute: Bool
    ) -> DatasetPreflightCacheKey {
        let dataURL = URL(fileURLWithPath: dataPath).standardizedFileURL
        let modelURL = URL(fileURLWithPath: modelPath).standardizedFileURL
        let parts = [
            "v2",
            dataIdentity(dataURL),
            tokenizerIdentity(modelURL),
            "ctx=\(ctxLen)",
            "template=\(template)",
            "turn=\(turnPolicy)",
            "prompt=\(promptKey)",
            "response=\(responseKey)",
            "training=\(trainingDataRoute)",
        ]
        let digest = SHA256.hash(data: Data(parts.joined(separator: "\n").utf8))
            .map { String(format: "%02x", $0) }.joined()
        let base = rootURL.appendingPathComponent(digest)
        return DatasetPreflightCacheKey(
            value: digest,
            previewURL: base.appendingPathExtension("preview.jsonl"),
            resultURL: base.appendingPathExtension("result.json")
        )
    }

    static func load(_ key: DatasetPreflightCacheKey) -> DatasetPreviewResult? {
        guard let data = try? Data(contentsOf: key.resultURL),
              let result = try? JSONDecoder().decode(DatasetPreviewResult.self, from: data)
        else { return nil }
        if let cachePath = result.pagination?.cachePath,
           !FileManager.default.fileExists(atPath: cachePath) {
            return nil
        }
        return result
    }

    static func save(_ result: DatasetPreviewResult, for key: DatasetPreflightCacheKey) {
        guard let data = try? JSONEncoder().encode(result) else { return }
        try? data.write(to: key.resultURL, options: .atomic)
    }

    private static func fileIdentity(_ url: URL) -> String {
        let values = try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        return "\(url.path)|\(values?.fileSize ?? -1)|\(values?.contentModificationDate?.timeIntervalSince1970 ?? -1)"
    }

    private static func dataIdentity(_ dataURL: URL) -> String {
        let sidecarURL = dataURL.appendingPathExtension("import.json")
        let sidecar = FileManager.default.fileExists(atPath: sidecarURL.path)
            ? fileIdentity(sidecarURL)
            : "sidecar=none"
        return "\(fileIdentity(dataURL))|\(sidecar)"
    }

    private static func tokenizerIdentity(_ modelURL: URL) -> String {
        let names = (try? FileManager.default.contentsOfDirectory(
            at: modelURL,
            includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ))?.filter {
            let name = $0.lastPathComponent.lowercased()
            return name == "config.json" || name.contains("tokenizer")
                || name.hasPrefix("vocab") || name.hasPrefix("merges")
        }.sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []
        return (["model=\(modelURL.path)"] + names.map(fileIdentity)).joined(separator: "|")
    }
}
