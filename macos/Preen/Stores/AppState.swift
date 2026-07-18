//
//  AppState.swift
//  Preen
//
//  App 全局状态(@Observable)。持有 TrainStore / ChatStore、当前面板、当前模型、
//  跨面板 state 传递(训练完成 → 跳对话自动加载产物 state)。
//

import Foundation
import Observation

struct TrainingDataSelectionRequest: Equatable {
    let id: UUID
    let path: String
}

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
    let dataPath: String?
    let runID: UUID?
}

@Observable
@MainActor
final class AppState {
    // === 当前面板 ===
    var selection: SidebarItem = .training
    var selectedRunID: UUID?
    private(set) var builtinTrainingRequestID: UUID?
    private(set) var importedTrainingDataRequest: TrainingDataSelectionRequest?
    private var returnsToTrainingAfterDatasetImport = false

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
    private(set) var pendingSessionReplacement: PendingSessionReplacement?
    private(set) var sessionReplacementError: String?
    private(set) var isInspectingState = false
    private let stateInspectionRunner = StateInspectionRunner()

    /// 本次运行内抑制会话替换确认弹窗。仅存内存,重启 App 自动重置;
    /// 在 ContentView 的确认弹窗里勾选「本次运行内不再提醒」时置 true。
    /// PRD P0-04 §七「不增加永久不再提醒」的边界由"仅本次运行"维持。
    var suppressSessionReplacementConfirmation = false

    init(defaults: UserDefaults = .standard) {
        PythonResolver.ensureApplicationDirectories()
        let repository = RunRepository()
        let backend = BackendStore()
        self.modelCatalog = RecentModelCatalog(defaults: defaults)
        self.runRepository = repository
        self.backendStore = backend
        self.trainStore = TrainStore(repository: repository, backendStore: backend)
        self.chatStore = ChatStore(backendStore: backend, runRepository: repository, defaults: defaults)
        self.toolboxStore = ToolboxStore()
    }

    // MARK: - 模型与进程协调

    func selectModel(path: String) {
        guard path != modelCatalog.selectedPath else { return }
        requestSessionReplacement(.selectModel(path))
    }

    private func performModelSelection(path: String) async {
        let previousPath = modelCatalog.selectedPath
        if path != previousPath, chatStore.hasActiveProcess {
            await chatStore.disconnectAndWait()
        }
        modelCatalog.select(path: path)
        if modelCatalog.selectedPath != previousPath {
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
        requestSessionReplacement(.disconnect)
    }

    // MARK: - 会话替换事务

    /// 所有会清空/替换会话的入口都进入这里。确认前只保存意图，不改变页面、模型、
    /// State、格式或当前生成；空会话或本次运行抑制时直接执行。
    func requestSessionReplacement(_ intent: SessionReplacementIntent) {
        // 本次运行内用户已勾选抑制:跳过确认,直接执行。
        if suppressSessionReplacementConfirmation {
            Task { [weak self] in
                guard let self else { return }
                if chatStore.isGenerating {
                    await chatStore.stopGenerationForSessionReplacement()
                }
                await executeSessionReplacement(intent)
            }
            return
        }
        if chatStore.hasReplaceableSessionContent {
            pendingSessionReplacement = PendingSessionReplacement(
                intent: intent,
                wasGenerating: chatStore.isGenerating
            )
        } else {
            Task { await executeSessionReplacement(intent) }
        }
    }

    func confirmSessionReplacement(suppressFuture: Bool) {
        guard let pending = pendingSessionReplacement else { return }
        pendingSessionReplacement = nil
        if suppressFuture {
            suppressSessionReplacementConfirmation = true
        }
        Task { [weak self] in
            guard let self else { return }
            if pending.wasGenerating {
                await chatStore.stopGenerationForSessionReplacement()
            }
            await executeSessionReplacement(pending.intent)
        }
    }

    func cancelSessionReplacement() {
        pendingSessionReplacement = nil
    }

    func requestSessionConfig(_ proposed: ChatSessionConfig) {
        let normalized = proposed.normalized()
        guard normalized.isValid else { return }
        if normalized.formatFields == chatStore.sessionConfig.formatFields {
            chatStore.applySessionConfig(normalized)
        } else {
            requestSessionReplacement(.applySessionConfig(normalized))
        }
    }

    private func executeSessionReplacement(_ intent: SessionReplacementIntent) async {
        switch intent {
        case .activateState(let request, let template, let useSuggestedModel):
            await performStateActivation(
                request,
                template: template,
                useSuggestedModel: useSuggestedModel
            )
        case .clearState:
            chatStore.clearState()
            injectedStatePath = nil
        case .applySessionConfig(let config):
            chatStore.applySessionConfig(config)
        case .selectModel(let path):
            await performModelSelection(path: path)
        case .disconnect:
            await chatStore.disconnectAndWait()
        }
    }

    /// 深链:切到工具箱并打开「模型转换」页(欢迎窗口 / 训练空态无模型时调用)。
    func goToModelConversion() {
        toolboxStore.pendingTool = "modelConversion"
        selection = .toolbox
    }

    func configureTrainingDataImport(path: String, ctxLen: Int) {
        toolboxStore.selectDatasetSource(path: path)
        toolboxStore.datasetContextLength = ctxLen
        toolboxStore.datasetOutputPath = PythonResolver.datasetsDirectory
            .appendingPathComponent(
                URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
                    + ".standard.jsonl"
            ).path
        toolboxStore.pendingTool = "datasetPreview"
        returnsToTrainingAfterDatasetImport = true
        selection = .toolbox
    }

    func consumeImportedDatasetForTraining(_ path: String?) {
        guard returnsToTrainingAfterDatasetImport, let path, !path.isEmpty else { return }
        returnsToTrainingAfterDatasetImport = false
        importedTrainingDataRequest = TrainingDataSelectionRequest(id: UUID(), path: path)
        selection = .training
    }

    /// 欢迎页的一键示例入口；TrainingPanel 收到 revision 后从 Bundle 读取固定数据。
    func requestBuiltinExampleTraining() {
        builtinTrainingRequestID = UUID()
        selection = .training
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
    func goToChat(
        stateURL: URL,
        trainingConfig: PersistedTrainingConfig?,
        runID: UUID?
    ) {
        requestStateActivation(
            stateURL: stateURL,
            trainingConfig: trainingConfig,
            associatedRunID: runID
        )
    }

    /// 训练记录优先、相邻 metadata 次之、App 默认最后。缺模板或外部 State
    /// 模型名不同时先形成待确认请求，不提前清空/切页。
    func requestStateActivation(
        stateURL: URL,
        trainingConfig: PersistedTrainingConfig? = nil,
        associatedRunID: UUID? = nil
    ) {
        guard !isInspectingState else { return }
        isInspectingState = true
        sessionReplacementError = nil
        Task { [weak self] in
            guard let self else { return }
            let outcome = await stateInspectionRunner.inspect(stateURL: stateURL)
            isInspectingState = false
            switch outcome {
            case .success:
                prepareStateActivation(
                    stateURL: stateURL,
                    trainingConfig: trainingConfig,
                    associatedRunID: associatedRunID
                )
            case .failure(let message):
                sessionReplacementError = message
            }
        }
    }

    /// 路径、格式与 RWKV-7 shape 预检成功后才形成 UI 请求。
    private func prepareStateActivation(
        stateURL: URL,
        trainingConfig: PersistedTrainingConfig?,
        associatedRunID: UUID?
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
            autoSwitchModel: trainingConfig != nil,
            dataPath: trainingConfig?.dataPath,
            runID: associatedRunID
        )
        if template == nil || requiresModelChoice {
            pendingStateActivation = request
        } else {
            requestSessionReplacement(.activateState(
                request: request,
                template: template ?? .qa,
                useSuggestedModel: request.autoSwitchModel
            ))
        }
    }

    func confirmStateActivation(template: ChatTemplate, useSuggestedModel: Bool) {
        guard let request = pendingStateActivation else { return }
        pendingStateActivation = nil
        requestSessionReplacement(.activateState(
            request: request,
            template: template,
            useSuggestedModel: useSuggestedModel
        ))
    }

    func cancelStateActivation() {
        pendingStateActivation = nil
    }

    func clearSessionReplacementError() {
        sessionReplacementError = nil
    }

    private func performStateActivation(
        _ request: StateActivationRequest,
        template: ChatTemplate,
        useSuggestedModel: Bool
    ) async {
        if useSuggestedModel,
           let suggestedModel = request.suggestedModelPath,
           !suggestedModel.isEmpty,
           FileManager.default.fileExists(atPath: suggestedModel),
           suggestedModel != modelPath {
            await performModelSelection(path: suggestedModel)
        }

        chatStore.prepareSessionReplacement(
            statePath: request.stateURL.path,
            suggestedTemplate: template,
            source: request.source
        )
        chatStore.isComparisonMode = true
        chatStore.configureComparisonContext(dataPath: request.dataPath, runID: request.runID)
        injectedStatePath = request.stateURL.path
        selection = .chat
        if chatStore.isConnected {
            chatStore.activatePreparedSession()
        } else if !modelPath.isEmpty {
            connectInference()
        }
    }
}
