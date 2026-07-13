//
//  AppState.swift
//  Preen
//
//  App 全局状态(@Observable)。持有 TrainStore / ChatStore、当前面板、当前模型、
//  跨面板 state 传递(训练完成 → 跳对话自动加载产物 state)。
//

import Foundation
import Observation

/// 侧边栏选中项。
enum SidebarItem: String, CaseIterable, Identifiable {
    case training
    case chat
    case library
    var id: String { rawValue }
    var label: String {
        switch self {
        case .training: return "训练"
        case .chat: return "对话"
        case .library: return "State 库"
        }
    }
    var systemImage: String {
        switch self {
        case .training: return "graduationcap"
        case .chat: return "bubble.left.and.bubble.right"
        case .library: return "shippingbox"
        }
    }
}

@Observable
@MainActor
final class AppState {
    // === 当前面板 ===
    var selection: SidebarItem = .training

    // === 模型(侧边栏底部选,全 app 共享)===
    var modelPath: String = ""

    // === 子 store ===
    let trainStore = TrainStore()
    let chatStore = ChatStore()
    let backendStore = BackendStore()
    let runRepository = RunRepository()

    init() {
        PythonResolver.ensureApplicationDirectories()
    }

    // === 跨面板:训练完成 → 跳对话,自动设上产物 state ===
    var injectedStatePath: String?

    /// 「去对话」入口:训练完成态的按钮调用。
    func goToChat(stateURL: URL) {
        injectedStatePath = stateURL.path
        selection = .chat
        // 如果对话面板已连接,立即设 state;否则用户点连接后 onChange 会接住。
        if chatStore.isConnected {
            chatStore.setState(path: stateURL.path)
        }
    }
}
