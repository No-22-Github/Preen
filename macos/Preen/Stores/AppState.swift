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
        case .training: return L10n.string("训练")
        case .chat: return L10n.string("对话")
        case .history: return L10n.string("训练记录")
        case .toolbox: return L10n.string("工具箱")
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

struct StateActivationRequest: Identifiable, Equatable {
    let id = UUID()
    let stateURL: URL
    let suggestedTemplate: ChatTemplate?
    let source: ChatConfigurationSource
    let suggestedModelPath: String?
    let suggestedModelName: String?
    let requiresModelChoice: Bool
    let autoSwitchModel: Bool
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
    private(set) var pendingStateActivation: StateActivationRequest?

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
                    message: L10n.string("正在释放推理模型")
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

    // === 跨面板:训练完成 → 跳对话,一键启动模型 + 加载产物 state ===
    var injectedStatePath: String?

    /// 「去对话」入口:训练完成态(或历史记录详情)的按钮调用。
    ///
    /// 一键流程(用户裁决):
    ///  1. 若 `trainingModelPath` 与当前全局模型不同 → 切到训练用的模型(selectModel 会
    ///     disconnect 旧模型 + 清旧 state);
    ///  2. 把产物 state 路径预注入 chatStore.statePath(未连接时 newSession 会自动带上);
    ///  3. 切到对话面板;
    ///  4. 若后端未连接 → 自动 connect(ready 后 newSession 用预注入的 state)。
    ///     已连接则直接 setState 下发。
    func goToChat(stateURL: URL, trainingConfig: PersistedTrainingConfig?) {
        requestStateActivation(stateURL: stateURL, trainingConfig: trainingConfig)
    }

    /// 训练记录优先、相邻 metadata 次之、App 默认最后。缺模板或外部 State
    /// 模型名不同时先形成待确认请求，不提前清空/切页。
    func requestStateActivation(
        stateURL: URL,
        trainingConfig: PersistedTrainingConfig? = nil
    ) {
        let metadata = StateMetadata.loadAdjacent(to: stateURL)
        let templateRaw = trainingConfig?.template ?? metadata?.template
        let template = templateRaw.flatMap(ChatTemplate.init(rawValue:))
        let source: ChatConfigurationSource = trainingConfig != nil
            ? .trainingRecord
            : (metadata != nil ? .stateMetadata : .appDefault)
        let suggestedModelPath = trainingConfig?.modelPath ?? metadata?.modelPath
        let suggestedModelName = metadata?.modelName
            ?? suggestedModelPath.map { URL(fileURLWithPath: $0).lastPathComponent }
        let currentModelName = modelPath.isEmpty
            ? nil
            : URL(fileURLWithPath: modelPath).lastPathComponent
        let requiresModelChoice = trainingConfig == nil
            && suggestedModelName != nil
            && suggestedModelName != currentModelName

        let request = StateActivationRequest(
            stateURL: stateURL,
            suggestedTemplate: template,
            source: source,
            suggestedModelPath: suggestedModelPath,
            suggestedModelName: suggestedModelName,
            requiresModelChoice: requiresModelChoice,
            autoSwitchModel: trainingConfig != nil
        )
        if template == nil || requiresModelChoice {
            pendingStateActivation = request
        } else {
            performStateActivation(
                request,
                template: template,
                useSuggestedModel: request.autoSwitchModel
            )
        }
    }

    func confirmStateActivation(template: ChatTemplate, useSuggestedModel: Bool) {
        guard let request = pendingStateActivation else { return }
        pendingStateActivation = nil
        performStateActivation(request, template: template, useSuggestedModel: useSuggestedModel)
    }

    func cancelStateActivation() {
        pendingStateActivation = nil
    }

    private func performStateActivation(
        _ request: StateActivationRequest,
        template: ChatTemplate?,
        useSuggestedModel: Bool
    ) {
        if useSuggestedModel,
           let suggestedModel = request.suggestedModelPath,
           !suggestedModel.isEmpty,
           FileManager.default.fileExists(atPath: suggestedModel),
           suggestedModel != modelPath {
            selectModel(path: suggestedModel)
        }

        chatStore.prepareSessionReplacement(
            statePath: request.stateURL.path,
            suggestedTemplate: template,
            source: request.source
        )
        injectedStatePath = request.stateURL.path
        selection = .chat
        if chatStore.isConnected {
            chatStore.activatePreparedSession()
        } else if !modelPath.isEmpty {
            connectInference()
        }
    }
}
