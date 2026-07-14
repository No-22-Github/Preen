import Foundation

/// 读取模型目录的 config.json,判断权重精度。
///
/// 对齐 Python 侧 `service.validate_training_request`(service.py:78-83):
/// config.json 含 `quantization` 或 `quantization_config` 字段 → 量化模型(int8);
/// 否则 → bf16。读取失败降级为 bf16,不阻塞 UI。
enum ModelConfigProbe {
    /// 返回精度标记字符串:"int8" 或 "bf16"。
    /// 读 config.json 失败(文件缺失/JSON 解析失败)时降级为 "bf16",
    /// 因为 toolbar 标记是装饰性信息,不应让异常路径阻塞模型选择。
    static func precisionBadge(for modelPath: String) -> String {
        guard !modelPath.isEmpty else { return "bf16" }
        let configURL = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return "bf16" }
        // quantization 或 quantization_config(HF 镜像字段)非空即量化模型
        if json["quantization"] != nil || json["quantization_config"] != nil {
            return "int8"
        }
        return "bf16"
    }

    /// 该模型是否可训练。int8 量化模型仅用于推理(state tuning 需 bf16 权重,
    /// 对齐 service.validate_training_request 的 quantization 早拦),故返回 false。
    /// 空路径 / 读取失败降级为 bf16 → 视为可训练,不提前阻塞 UI(真失败仍由 Python 侧兜底)。
    static func isTrainable(modelPath: String) -> Bool {
        precisionBadge(for: modelPath) != "int8"
    }
}
