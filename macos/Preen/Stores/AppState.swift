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
    case history
    case toolbox
    var id: String { rawValue }
    var label: String {
        switch self {
        case .training: return "训练"
        case .chat: return "对话"
        case .history: return "训练记录"
        case .toolbox: return "工具箱"
        }
    }
    var systemImage: String {
        switch self {
        case .training: return "graduationcap"
        case .chat: return "bubble.left.and.bubble.right"
        case .history: return "clock.arrow.circlepath"
        case .toolbox: return "wrench.and.screwdriver"
        }
    }
}

@Observable
@MainActor
final class AppState {
    // === 当前面板 ===
    var selection: SidebarItem = .training
    var selectedRunID: UUID?

    /// 是否显示欢迎窗口(主窗口的模态 sheet)。为 true 时侧栏也会收起,让背景呈空状态。
    /// 首启 / 「窗口 → 欢迎使用 Preen」菜单翻为 true;sheet 关闭(WelcomeView dismiss / 点背景)翻回 false。
    var isWelcomePresented = false

    // === 模型(顶部中央 toolbar 选,全 app 共享)===
    private var modelCatalog: RecentModelCatalog
    var modelPath: String {
        get { modelCatalog.selectedPath }
        set { selectModel(path: newValue) }
    }
    var recentModels: [RecentModel] { modelCatalog.entries }

    // === 子 store ===
    let trainStore: TrainStore
    let chatStore: ChatStore
    let backendStore: BackendStore
    let toolboxStore: ToolboxStore
    let runRepository: RunRepository
    private(set) var runs: [TrainingRun] = []
    private(set) var isSwitchingWorker = false

    init(defaults: UserDefaults = .standard) {
        PythonResolver.ensureApplicationDirectories()
        let repository = RunRepository()
        let backend = BackendStore()
        self.modelCatalog = RecentModelCatalog(defaults: defaults)
        self.runRepository = repository
        self.backendStore = backend
        self.trainStore = TrainStore(repository: repository, backendStore: backend)
        self.chatStore = ChatStore(backendStore: backend)
        self.toolboxStore = ToolboxStore()
    }

    // MARK: - 模型与进程协调

    func selectModel(path: String) {
        let previousPath = modelCatalog.selectedPath
        modelCatalog.select(path: path)
        if modelCatalog.selectedPath != previousPath {
            // 换模型不会自动重连；先终止旧模型，避免 Metal 内存池继续驻留。
            if chatStore.hasActiveProcess {
                chatStore.disconnect()
            }
            // state 针对特定模型训练,换模型必须清,否则旧模型的 state 会继承给新模型。
            chatStore.clearState()
            injectedStatePath = nil
        }
    }

    /// 模型列表每次展开前调用，移除已经移动/删除的目录。
    func validateRecentModels() {
        let previousPath = modelCatalog.selectedPath
        modelCatalog.validate()
        if previousPath != modelCatalog.selectedPath {
            if chatStore.hasActiveProcess {
                chatStore.disconnect()
            }
            // 同 selectModel:被移除的模型其 state 也应失效。
            chatStore.clearState()
            injectedStatePath = nil
        }
    }

    /// 强制单工作进程：释放推理模型后才允许训练进程启动。
    func startTraining(config: TrainingConfig) {
        guard !isSwitchingWorker else { return }
        isSwitchingWorker = true
        Task { [weak self] in
            guard let self else { return }
            defer { isSwitchingWorker = false }
            if chatStore.hasActiveProcess {
                backendStore.updateInference(
                    phase: .stopping,
                    pid: chatStore.processID,
                    message: "正在释放推理模型"
                )
                await chatStore.disconnectAndWait()
            }
            trainStore.start(config: config)
        }
    }

    /// 强制单工作进程：取消并等训练进程退出后才加载推理模型。
    func connectInference() {
        guard !modelPath.isEmpty, !isSwitchingWorker else { return }
        let model = URL(fileURLWithPath: modelPath)
        isSwitchingWorker = true
        Task { [weak self] in
            guard let self else { return }
            defer { isSwitchingWorker = false }
            if trainStore.hasActiveProcess {
                await trainStore.cancelAndWait()
            }
            chatStore.connect(model: model)
        }
    }

    func disconnectInference() {
        chatStore.disconnect()
    }

    /// 深链:切到工具箱并打开「模型转换」页(欢迎窗口 / 训练空态无模型时调用)。
    func goToModelConversion() {
        toolboxStore.pendingTool = "modelConversion"
        selection = .toolbox
    }

    func restoreRuns() async {
        _ = try? await runRepository.markUnfinishedRunsInterrupted()
        runs = await runRepository.scan()
    }

    func refreshRuns() async {
        runs = await runRepository.scan()
    }

    func deleteRun(id: UUID) async throws {
        let deletedIndex = runs.firstIndex { $0.id == id } ?? 0
        try await runRepository.delete(id: id)
        await refreshRuns()
        if selectedRunID == id {
            selectedRunID = runs.isEmpty ? nil : runs[min(deletedIndex, runs.count - 1)].id
        }
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
