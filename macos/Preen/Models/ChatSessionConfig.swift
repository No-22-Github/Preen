//
//  ChatSessionConfig.swift
//  Preen
//
//  UI 中立的会话口径。Swift 只选择协议参数，prompt 文本仍由 Python
//  templates.py / render_prompt 统一渲染。
//

import Foundation

enum ChatTemplate: String, CaseIterable, Codable, Identifiable {
    case qa
    case instruction
    case raw

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .qa: return "QA"
        case .instruction: return "Instruction"
        case .raw: return "Raw"
        }
    }
}

enum ThinkMode: String, CaseIterable, Codable, Identifiable {
    case off
    case fast
    case on

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .off: return "Off"
        case .fast: return "Fast"
        case .on: return "On"
        }
    }
}

struct ChatSessionConfig: Equatable {
    var template: ChatTemplate = .qa
    var reasoning = false
    var think: ThinkMode = .off
    var genConfig: GenConfig = .defaultConfig

    static let defaultConfig = ChatSessionConfig()

    /// 后端也会校验；这里用于 UI 提前禁用非法组合。
    var isValid: Bool {
        guard template == .qa || !reasoning else { return false }
        guard reasoning || think == .off else { return false }
        return true
    }

    /// 模板切换时收敛为协议合法值，不在 Swift 拼接任何 prompt。
    func normalized() -> ChatSessionConfig {
        var value = self
        if value.template != .qa {
            value.reasoning = false
            value.think = .off
        } else if !value.reasoning {
            value.think = .off
        }
        return value
    }

    var formatSummary: String {
        if reasoning {
            return "\(template.rawValue) · \(think.rawValue)"
        }
        return "\(template.rawValue) · off"
    }

    var formatFields: FormatFields {
        FormatFields(template: template, reasoning: reasoning, think: think)
    }

    struct FormatFields: Equatable {
        let template: ChatTemplate
        let reasoning: Bool
        let think: ThinkMode
    }
}
