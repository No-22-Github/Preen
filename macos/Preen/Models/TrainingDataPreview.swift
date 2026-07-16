//
//  TrainingDataPreview.swift
//  Preen
//
//  配参界面的轻量训练集预览:纯 Swift 端就地读文件前几条原始记录,
//  不起 Python 进程、不加载 tokenizer。目的是"看训练集里到底有啥",
//  而非工具箱那套 token 级模板渲染(那套重型预览仍在工具箱·数据集预览)。
//
//  支持 .jsonl / .json(数组或对象包 list)/ .csv,与 TrainingEmptyView 接受的格式一致。
//

import Foundation

/// 单条原始记录的预览(键值对,保留源字段名)。
struct TrainingDataSample: Identifiable {
    let id: Int
    let fields: [(key: String, value: String)]
}

/// 轻量预览结果:前几条样本 + 是否还有更多。
struct TrainingDataPreview {
    let samples: [TrainingDataSample]
    let hasMore: Bool          // 文件里还有超出预览条数的记录
    let error: String?         // 读取/解析失败的简短说明(nil 表示成功)

    static let empty = TrainingDataPreview(samples: [], hasMore: false, error: nil)

    /// 快速数记录条数(不 tokenize,只为 30K 阈值判断"改动时是否即时检查")。
    /// jsonl 数非空行、json 取数组长度、csv 数行数减表头;失败返回 nil。
    static func countRecords(path: String) -> Int? {
        guard !path.isEmpty else { return nil }
        let url = URL(fileURLWithPath: path)
        let ext = url.pathExtension.lowercased()
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        switch ext {
        case "jsonl", "":
            return text.split(separator: "\n", omittingEmptySubsequences: true)
                .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.count
        case "json":
            guard let data = text.data(using: .utf8),
                  let root = try? JSONSerialization.jsonObject(with: data) else { return nil }
            if let arr = root as? [Any] { return arr.count }
            if let dict = root as? [String: Any] {
                for v in dict.values { if let arr = v as? [Any] { return arr.count } }
                return 1
            }
            return nil
        case "csv":
            let lines = text.split(separator: "\n", omittingEmptySubsequences: true).count
            return max(0, lines - 1)  // 减表头(近似,够阈值判断)
        default:
            return nil
        }
    }

    /// 读取文件前 `limit` 条记录。只读到 limit+1 条即停(jsonl 逐行、json/csv 读全量后截断)。
    /// 任何失败降级为 error 文案,不抛异常(预览是辅助信息,不应阻塞配参)。
    static func load(path: String, limit: Int = 5) -> TrainingDataPreview {
        guard !path.isEmpty else { return .empty }
        let url = URL(fileURLWithPath: path)
        let ext = url.pathExtension.lowercased()
        do {
            switch ext {
            case "jsonl", "":
                return try loadJSONL(url: url, limit: limit)
            case "json":
                return try loadJSON(url: url, limit: limit)
            case "csv":
                return try loadCSV(url: url, limit: limit)
            default:
                return TrainingDataPreview(samples: [], hasMore: false,
                                           error: L10n.format("不支持预览 .%@ 格式", ext))
            }
        } catch {
            return TrainingDataPreview(samples: [], hasMore: false,
                                       error: L10n.format("读取失败：%@", error.localizedDescription))
        }
    }

    // MARK: - 各格式读取

    private static func loadJSONL(url: URL, limit: Int) throws -> TrainingDataPreview {
        let text = try String(contentsOf: url, encoding: .utf8)
        var objects: [[String: Any]] = []
        var overflow = false
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            if objects.count >= limit { overflow = true; break }
            if let data = line.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                objects.append(obj)
            }
        }
        return makePreview(from: objects, hasMore: overflow, limit: limit)
    }

    private static func loadJSON(url: URL, limit: Int) throws -> TrainingDataPreview {
        let data = try Data(contentsOf: url)
        let root = try JSONSerialization.jsonObject(with: data)
        let array = extractArray(from: root)
        let slice = Array(array.prefix(limit))
        return makePreview(from: slice, hasMore: array.count > limit, limit: limit)
    }

    private static func loadCSV(url: URL, limit: Int) throws -> TrainingDataPreview {
        let text = try String(contentsOf: url, encoding: .utf8)
        let rows = parseCSV(text, maxRows: limit + 1)  // +1 判 hasMore(不含表头)
        guard let header = rows.first else { return .empty }
        let bodyRows = rows.dropFirst()
        var objects: [[String: Any]] = []
        for row in bodyRows.prefix(limit) {
            var obj: [String: Any] = [:]
            for (i, key) in header.enumerated() where i < row.count {
                obj[key] = row[i]
            }
            objects.append(obj)
        }
        return makePreview(from: objects, hasMore: bodyRows.count > limit, limit: limit)
    }

    // MARK: - 辅助

    /// 从 JSON 顶层取记录数组:数组直接用;对象则取第一个 list 值(对齐 importer.read_records)。
    private static func extractArray(from root: Any) -> [[String: Any]] {
        if let arr = root as? [[String: Any]] { return arr }
        if let dict = root as? [String: Any] {
            for value in dict.values {
                if let arr = value as? [[String: Any]] { return arr }
            }
            return [dict]
        }
        return []
    }

    private static func makePreview(
        from objects: [[String: Any]], hasMore: Bool, limit: Int
    ) -> TrainingDataPreview {
        let samples = objects.prefix(limit).enumerated().map { index, obj in
            TrainingDataSample(id: index, fields: flatten(obj))
        }
        return TrainingDataPreview(samples: Array(samples), hasMore: hasMore, error: nil)
    }

    /// 把一条记录压平成 (键, 值) 数组。嵌套结构(messages/conversations)转紧凑 JSON 字符串,
    /// 让用户至少看得到原文;键顺序稳定(字典无序 → 按键名排序,常见字段优先)。
    private static func flatten(_ obj: [String: Any]) -> [(key: String, value: String)] {
        let priority = ["instruction", "input", "prompt", "question", "q",
                        "output", "response", "answer", "a", "messages", "conversations"]
        let sortedKeys = obj.keys.sorted { lhs, rhs in
            let li = priority.firstIndex(of: lhs) ?? Int.max
            let ri = priority.firstIndex(of: rhs) ?? Int.max
            return li != ri ? li < ri : lhs < rhs
        }
        return sortedKeys.map { key in (key, stringify(obj[key])) }
    }

    private static func stringify(_ value: Any?) -> String {
        switch value {
        case let s as String: return s
        case let n as NSNumber: return n.stringValue
        case .none, is NSNull: return ""
        default:
            if let value,
               let data = try? JSONSerialization.data(withJSONObject: value),
               let s = String(data: data, encoding: .utf8) {
                return s
            }
            return String(describing: value ?? "")
        }
    }

    /// 极简 CSV 解析:支持双引号包裹字段内的逗号与换行、"" 转义。够预览用,非全 RFC 4180。
    private static func parseCSV(_ text: String, maxRows: Int) -> [[String]] {
        var rows: [[String]] = []
        var field = ""
        var row: [String] = []
        var inQuotes = false
        let chars = Array(text)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if inQuotes {
                if c == "\"" {
                    if i + 1 < chars.count && chars[i + 1] == "\"" {
                        field.append("\""); i += 1
                    } else {
                        inQuotes = false
                    }
                } else {
                    field.append(c)
                }
            } else {
                switch c {
                case "\"": inQuotes = true
                case ",": row.append(field); field = ""
                case "\n", "\r":
                    if c == "\r" && i + 1 < chars.count && chars[i + 1] == "\n" { i += 1 }
                    row.append(field); field = ""
                    rows.append(row); row = []
                    if rows.count >= maxRows { return rows }
                default: field.append(c)
                }
            }
            i += 1
        }
        if !field.isEmpty || !row.isEmpty {
            row.append(field)
            rows.append(row)
        }
        return rows
    }
}
