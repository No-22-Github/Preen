import Foundation

struct RecentModel: Codable, Equatable, Identifiable {
    var id: String { path }

    let path: String
    var lastUsedAt: Date

    var displayName: String {
        URL(fileURLWithPath: path).lastPathComponent
    }
}

/// App 级模型历史。只保存路径；每次展开选择器时由调用方执行 validate()。
struct RecentModelCatalog {
    private enum Key {
        static let entries = "recentModels.v1"
        static let selectedPath = "selectedModelPath.v1"
    }

    private let defaults: UserDefaults
    private let fileManager: FileManager
    private let maximumCount: Int

    private(set) var entries: [RecentModel]
    private(set) var selectedPath: String

    init(
        defaults: UserDefaults = .standard,
        fileManager: FileManager = .default,
        maximumCount: Int = 8
    ) {
        self.defaults = defaults
        self.fileManager = fileManager
        self.maximumCount = maximumCount
        if let data = defaults.data(forKey: Key.entries),
           let decoded = try? JSONDecoder().decode([RecentModel].self, from: data) {
            entries = decoded
        } else {
            entries = []
        }
        selectedPath = defaults.string(forKey: Key.selectedPath) ?? ""
    }

    mutating func select(path: String, at date: Date = Date()) {
        let normalized = Self.normalize(path)
        guard !normalized.isEmpty else {
            selectedPath = ""
            persist()
            return
        }

        entries.removeAll { $0.path == normalized }
        entries.insert(RecentModel(path: normalized, lastUsedAt: date), at: 0)
        if entries.count > maximumCount {
            entries.removeLast(entries.count - maximumCount)
        }
        selectedPath = normalized
        persist()
    }

    /// 删除已移动/删除或不再是目录的记录，并清除失效的当前选择。
    mutating func validate() {
        entries = entries.filter { model in
            var isDirectory: ObjCBool = false
            return fileManager.fileExists(atPath: model.path, isDirectory: &isDirectory)
                && isDirectory.boolValue
        }
        if !selectedPath.isEmpty && !entries.contains(where: { $0.path == selectedPath }) {
            selectedPath = ""
        }
        persist()
    }

    private mutating func persist() {
        if let data = try? JSONEncoder().encode(entries) {
            defaults.set(data, forKey: Key.entries)
        }
        if selectedPath.isEmpty {
            defaults.removeObject(forKey: Key.selectedPath)
        } else {
            defaults.set(selectedPath, forKey: Key.selectedPath)
        }
    }

    private static func normalize(_ path: String) -> String {
        guard !path.isEmpty else { return "" }
        return URL(fileURLWithPath: path).standardizedFileURL.path
    }
}
