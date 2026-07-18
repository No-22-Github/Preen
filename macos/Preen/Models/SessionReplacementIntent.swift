import Foundation

enum SessionReplacementPolicy {
    static func requiresConfirmation(
        messageCount: Int,
        comparisonHasContent: Bool,
        isGenerating: Bool
    ) -> Bool {
        messageCount > 0 || comparisonHasContent || isGenerating
    }
}

/// 会结束、清空或替换当前推理会话的 App 级意图。
///
/// 所有入口只负责描述目标；AppState 在确认前不改变 selection、模型、State 或配置。
enum SessionReplacementIntent {
    case activateState(
        request: StateActivationRequest,
        template: ChatTemplate,
        useSuggestedModel: Bool
    )
    case clearState
    case applySessionConfig(ChatSessionConfig)
    case selectModel(String)
    case disconnect

    var destructiveButtonTitle: String {
        switch self {
        case .activateState: return L10n.string("打开新会话")
        case .clearState: return L10n.string("卸下 State")
        case .applySessionConfig: return L10n.string("更改格式")
        case .selectModel: return L10n.string("切换模型")
        case .disconnect: return L10n.string("断开")
        }
    }

    func title(isGenerating: Bool) -> String {
        let suffix = isGenerating ? L10n.string("并停止当前生成？") : L10n.string("会结束当前会话？")
        switch self {
        case .activateState(_, _, _): return L10n.format("打开这个 State %@", suffix)
        case .clearState: return L10n.format("卸下 State %@", suffix)
        case .applySessionConfig: return L10n.format("更改模板 %@", suffix)
        case .selectModel: return L10n.format("切换模型 %@", suffix)
        case .disconnect: return L10n.format("断开推理连接 %@", suffix)
        }
    }

    func consequence(currentModelPath: String, isGenerating: Bool) -> String {
        var target: String
        switch self {
        case .activateState(let request, _, let useSuggestedModel):
            let modelPath = useSuggestedModel ? request.suggestedModelPath : currentModelPath
            let model = modelPath.flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0).lastPathComponent }
                ?? L10n.string("当前模型")
            target = L10n.format("新会话将使用模型 %@ 与 State %@。", model, request.stateURL.lastPathComponent)
        case .clearState:
            target = L10n.string("新会话将使用当前模型且不加载 State。")
        case .applySessionConfig(let config):
            target = L10n.format("新会话将使用 %@ 格式。", config.formatSummary)
        case .selectModel(let path):
            target = L10n.format("新会话将使用模型 %@，且不会继承当前 State。", URL(fileURLWithPath: path).lastPathComponent)
        case .disconnect:
            target = L10n.string("推理后端将结束并进入断开状态。")
        }
        let generation = isGenerating ? L10n.string("当前生成也会停止。") : ""
        return [L10n.string("当前消息与模型 cache 将被清除。"), generation, target,
                L10n.string("此操作无法撤销。")]
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}

struct PendingSessionReplacement: Identifiable {
    let id = UUID()
    let intent: SessionReplacementIntent
    let wasGenerating: Bool
}
