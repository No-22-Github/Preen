//
//  ContentView.swift
//  Preen
//
//  主容器。NavigationSplitView 两栏:
//   - Sidebar(训练/对话/训练记录/工具箱)
//   - detail(按 selection 切面板)
//   - 模型选择器位于 detail toolbar 中央
//
//  最小 1000×680(design.md §3)。
//

import SwiftUI

struct ContentView: View {
    @Bindable var appState: AppState
    @State private var isShowingChatGenerationParameters = false
    /// 「去对话」的一键加载真正完成后,顶部模型 chip 短暂变绿。
    @State private var isModelChipAcknowledged = false
    @State private var modelChipPulseID = UUID()

    var body: some View {
        VStack(spacing: 0) {
            NavigationSplitView {
                Sidebar(appState: appState)
                    .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 280)
            } detail: {
                detail
                    .toolbar {
                        if appState.selection == .chat && appState.chatStore.isConnected {
                            ToolbarItemGroup(placement: .primaryAction) {
                                Button(action: requestDisconnect) {
                                    HStack(spacing: 6) {
                                        if #available(macOS 15.0, *) {
                                            Image(systemName: "personalhotspot.slash")
                                        } else {
                                            Image(systemName: "xmark.circle")
                                        }
                                        Text("断开")
                                    }
                                }
                                .help("断开推理连接")

                                Button {
                                    appState.chatStore.isComparisonMode.toggle()
                                } label: {
                                    HStack(spacing: 6) {
                                        Image(systemName: "rectangle.split.2x1")
                                        Text("A/B")
                                    }
                                }
                                .disabled(appState.chatStore.statePath == nil || appState.chatStore.isGenerating)
                                .help(appState.chatStore.statePath == nil
                                      ? "请先加载 State，再开始基线对比"
                                      : "切换单轮 State A/B 对比")

                                if let badge = chatStateBadgeLabel {
                                    // 已选 state:文件名按钮(点此重选,透明底)+ 紧邻的 ×(点此卸下)。
                                    // 两者用 plain 样式去掉系统 bezel,避免被渲染成两个分离气泡;
                                    // 只有 × 带圆形淡色背景,主区域保持 toolbar 原底色。
                                    Button(action: requestLoadState) {
                                        HStack(spacing: 6) {
                                            Image(systemName: "doc.fill")
                                            Text(badge).lineLimit(1).truncationMode(.middle)
                                        }
                                        // plain toolbar 按钮不套强调色且可能塌缩成零宽:
                                        // primary 保证内容可见,fixedSize 强制按内容撑开。
                                        .foregroundStyle(.primary)
                                        .fixedSize(horizontal: true, vertical: false)
                                        .padding(.vertical, 5)
                                        .contentShape(Rectangle())
                                    }
                                    .buttonStyle(.plain)
                                    .help("加载或替换对话用的 State")

                                    Button(action: requestClearState) {
                                        Image(systemName: "xmark")
                                            .font(.system(size: 9, weight: .bold))
                                            .frame(width: 18, height: 18)
                                            .foregroundStyle(.secondary)
                                            .background(Color.secondary.opacity(0.18), in: Circle())
                                    }
                                    .buttonStyle(.plain)
                                    .help("卸下当前 State，回到基线模式")
                                } else {
                                    // 未选 state:加载入口(原生单按钮)。
                                    Button(action: requestLoadState) {
                                        HStack(spacing: 6) {
                                            Image(systemName: "folder")
                                            Text("加载State…")
                                        }
                                    }
                                    .help("加载或替换对话用的 State")
                                }

                                Button {
                                    isShowingChatGenerationParameters = true
                                } label: {
                                    HStack(spacing: 6) {
                                        Image(systemName: "slider.horizontal.3")
                                        Text(appState.chatStore.sessionConfig.formatSummary)
                                    }
                                }
                                .help("会话格式、Reasoning、思考与生成参数")
                            }
                        }

                        ToolbarItem(id: "model-picker", placement: .principal) {
                            modelPickerMenu
                        }
                    }
            }
            Divider()
            GlobalStatusBar(appState: appState)
        }
        .frame(minWidth: 1000, minHeight: 680)
        .confirmationDialog(
            sessionReplacementTitle,
            isPresented: Binding(
                get: { appState.pendingSessionReplacement != nil },
                set: { if !$0 { appState.cancelSessionReplacement() } }
            ),
            titleVisibility: .visible
        ) {
            Button(sessionReplacementButtonTitle, role: .destructive) {
                appState.confirmSessionReplacement()
            }
            Button("取消", role: .cancel) { appState.cancelSessionReplacement() }
                .keyboardShortcut(.cancelAction)
        } message: {
            Text(sessionReplacementMessage)
        }
        .alert(
            "无法打开 State",
            isPresented: Binding(
                get: { appState.sessionReplacementError != nil },
                set: { if !$0 { appState.clearSessionReplacementError() } }
            )
        ) {
            Button("好") { appState.clearSessionReplacementError() }
        } message: {
            Text(appState.sessionReplacementError ?? "")
        }
        // 欢迎窗口:作为主窗口的模态 sheet,从顶部滑出、盖在主窗口上方、锁定(点背景不响应)。
        // 首启 / 「窗口 → 欢迎使用 Preen」翻 isWelcomePresented=true 弹出;同一标志也驱动侧栏收起。
        .sheet(isPresented: $appState.isWelcomePresented) {
            WelcomeView(appState: appState)
        }
        .sheet(isPresented: Binding(
            get: { appState.pendingStateActivation != nil },
            set: { if !$0 { appState.cancelStateActivation() } }
        )) {
            if let request = appState.pendingStateActivation {
                StateActivationSheet(
                    request: request,
                    currentModelPath: appState.modelPath,
                    onCancel: { appState.cancelStateActivation() },
                    onConfirm: { template, useSuggestedModel in
                        appState.confirmStateActivation(
                            template: template,
                            useSuggestedModel: useSuggestedModel
                        )
                    }
                )
            }
        }
        .onChange(of: appState.chatStore.activationRevision) { _, _ in
            // 模型 + State 真正落到可用会话:连接就绪(new_session 成功)或切换 state 后,
            // 顶部模型 chip 短暂变绿。覆盖正常点「连接」、去对话一键加载、手动切 state 等所有就绪时刻。
            appState.injectedStatePath = nil
            acknowledgeModelChip()
        }
        .onChange(of: appState.toolboxStore.importedDatasetPath) { _, path in
            appState.consumeImportedDatasetForTraining(path)
        }
    }

    /// 模型选择器(toolbar 下拉菜单 + 精度胶囊,位于窗口中央)。
    private var modelPickerMenu: some View {
        HStack(spacing: 8) {
            Menu {
                if appState.recentModels.isEmpty {
                    Text("还没有模型记录")
                } else {
                    ForEach(appState.recentModels) { model in
                        Button {
                            appState.selectModel(path: model.path)
                        } label: {
                            menuItemLabel(for: model, isSelected: appState.modelPath == model.path)
                        }
                    }
                }
                Divider()
                Button("选择其他模型…") {
                    pickModel()
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "shippingbox")
                    Text(menuLabelText)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            // Menu 的 label 会被系统样式化覆盖,自定义颜色/背景塞在 label 内不生效;
            // 把就绪反馈做成独立 overlay 叠在 Menu 外,绕开 Menu 的样式化。
            .overlay {
                Capsule()
                    .strokeBorder(Color.green.opacity(isModelChipAcknowledged ? 0.9 : 0), lineWidth: 1.5)
                    .padding(-5)
            }
            .animation(.easeInOut(duration: 0.2), value: isModelChipAcknowledged)
            .help(modelPickerHelp)

            // 精度胶囊:独立于 Menu label,避免 Menu 对多视图渲染的限制。
            if !appState.modelPath.isEmpty {
                precisionBadge
            }
        }
    }

    /// 下拉菜单项:模型名 + 右对齐精度标记(int8 附带橙色"仅推理")。
    /// 已选中项用 Label 显示 checkmark;否则纯 Text。
    /// 用 AttributedString 拼成单一 Text,避免 Menu item 对多视图的渲染限制。
    private func menuItemLabel(for model: RecentModel, isSelected: Bool) -> some View {
        let badge = ModelConfigProbe.precisionBadge(for: model.path)
        var label = AttributedString(model.displayName)
        var detail = AttributedString("\t\(badge.uppercased())")
        detail.foregroundColor = .secondary
        if badge == "int8" {
            var hint = AttributedString(L10n.string(" · 仅推理"))
            hint.foregroundColor = .orange
            detail += hint
        }
        label += detail
        if isSelected {
            return AnyView(Label(title: { Text(label) }, icon: { Image(systemName: "checkmark") }))
        } else {
            return AnyView(Text(label))
        }
    }

    /// 精度胶囊标签:圆角背景 + 大写文字,颜色按精度语义区分。
    /// int8=橙色(提速量化),bf16=灰色(标准)。
    private var precisionBadge: some View {
        let badge = ModelConfigProbe.precisionBadge(for: appState.modelPath)
        let isQuantized = badge == "int8"
        return Text(badge.uppercased())
            .font(.caption.weight(.semibold))
            .monospacedDigit()
            .foregroundStyle(isQuantized ? Color.orange : Color.secondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(
                (isQuantized ? Color.orange : Color.secondary).opacity(0.15),
                in: RoundedRectangle(cornerRadius: 5, style: .continuous)
            )
            .help(L10n.string(isQuantized ? "INT8 · 仅支持推理，不支持训练" : "BF16 标准精度"))
            .contentShape(Rectangle())
    }

    /// Menu label 文字:模型名用 semibold。用 AttributedString 携带 font 属性,
    /// 能穿透 Menu label 的系统样式覆盖(.fontWeight 修饰符会被 Menu 忽略)。
    private var menuLabelText: AttributedString {
        if appState.modelPath.isEmpty {
            return AttributedString(L10n.string("选择模型"))
        }
        var attr = AttributedString(URL(fileURLWithPath: appState.modelPath).lastPathComponent)
        attr.font = .body.weight(.semibold)
        return attr
    }

    /// 模型 chip 的辅助提示:不占页面空间,但能核对实际模型与 State 路径。
    private var modelPickerHelp: String {
        guard appState.chatStore.isConnected else {
            return appState.modelPath.isEmpty
                ? L10n.string("选择 RWKV-7 模型")
                : L10n.format("选择 RWKV-7 模型\n当前：%@", appState.modelPath)
        }
        let state = appState.chatStore.statePath ?? L10n.string("未加载（基线模式）")
        return L10n.format("模型已加载\n模型：%@\nState：%@", appState.modelPath, state)
    }

    private func acknowledgeModelChip() {
        let pulseID = UUID()
        modelChipPulseID = pulseID
        withAnimation(.easeOut(duration: 0.16)) {
            isModelChipAcknowledged = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            guard modelChipPulseID == pulseID else { return }
            withAnimation(.easeInOut(duration: 0.35)) {
                isModelChipAcknowledged = false
            }
        }
    }

    private func pickModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = L10n.string("选择模型目录")
        if panel.runModal() == .OK, let url = panel.url {
            appState.selectModel(path: url.path)
        }
    }

    /// 先选路径并做只读结构预检；只有目标明确后才进入统一会话替换确认。
    private func requestLoadState() {
        performLoadState()
    }

    private func requestClearState() {
        appState.requestSessionReplacement(.clearState)
    }

    private func requestDisconnect() {
        appState.disconnectInference()
    }

    /// 实际执行:弹文件选择器 → 应用 state。
    private func performLoadState() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.data]  // .npz 走 UTI data
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            appState.requestStateActivation(stateURL: url)
        }
    }

    /// 当前对话 state 的文件名(nil = 未选);驱动 toolbar 胶囊显示。
    private var chatStateBadgeLabel: String? {
        guard let path = appState.chatStore.statePath else { return nil }
        return URL(fileURLWithPath: path).lastPathComponent
    }

    private var sessionReplacementTitle: String {
        guard let pending = appState.pendingSessionReplacement else { return "" }
        return pending.intent.title(isGenerating: pending.wasGenerating)
    }

    private var sessionReplacementButtonTitle: String {
        appState.pendingSessionReplacement?.intent.destructiveButtonTitle ?? ""
    }

    private var sessionReplacementMessage: String {
        guard let pending = appState.pendingSessionReplacement else { return "" }
        return pending.intent.consequence(
            currentModelPath: appState.modelPath,
            isGenerating: pending.wasGenerating
        )
    }

    @ViewBuilder
    private var detail: some View {
        switch appState.selection {
        case .training:
            TrainingPanel(
                store: appState.trainStore,
                modelPath: $appState.modelPath,
                recentRuns: recentRuns,
                onSelectRun: { run in
                    appState.selectedRunID = run.id
                    appState.selection = .history
                },
                onConvertModel: { appState.goToModelConversion() },
                welcomePresented: appState.isWelcomePresented,
                builtinTrainingRequestID: appState.builtinTrainingRequestID,
                importedTrainingDataRequest: appState.importedTrainingDataRequest,
                onStart: { appState.startTraining(config: $0) },
                onConfigureImport: {
                    appState.configureTrainingDataImport(path: $0, ctxLen: $1)
                },
                onGoToChat: {
                    appState.goToChat(stateURL: $0, trainingConfig: $1, runID: $2)
                }
            )
        case .chat:
            if appState.modelPath.isEmpty {
                chatNeedsModel
            } else {
                ChatPanel(store: appState.chatStore,
                          modelPath: appState.modelPath,
                          isShowingGenerationParameters: $isShowingChatGenerationParameters,
                          onConnect: { appState.connectInference() },
                          onApplySessionConfig: { appState.requestSessionConfig($0) })
            }
        case .history:
            TrainingHistoryView(appState: appState)
        case .toolbox:
            ToolboxView(
                store: appState.toolboxStore,
                modelPath: appState.modelPath,
                onSelectModel: { appState.selectModel(path: $0) }
            )
        }
    }

    /// 训练面板空态用的最近训练列表(合并当前进行中的 run + 历史,按时间降序)。
    private var recentRuns: [TrainingRun] {
        var runs = appState.runs
        if let current = appState.trainStore.currentRun,
           !runs.contains(where: { $0.id == current.id }) {
            runs.insert(current, at: 0)
        }
        return runs.sorted { $0.createdAt > $1.createdAt }
    }

    private var chatNeedsModel: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 40))
                .foregroundStyle(.orange)
            Text("需要选择模型")
                .font(.title3)
            Text("对话面板需要一个 HF 格式的 RWKV-7 模型目录")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button("选择模型…") { pickModel() }
                .buttonStyle(.borderedProminent)
            Text("也可以使用窗口顶部的模型菜单。")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

}

private struct StateActivationSheet: View {
    let request: StateActivationRequest
    let currentModelPath: String
    let onCancel: () -> Void
    let onConfirm: (ChatTemplate, Bool) -> Void

    @State private var template: ChatTemplate

    init(
        request: StateActivationRequest,
        currentModelPath: String,
        onCancel: @escaping () -> Void,
        onConfirm: @escaping (ChatTemplate, Bool) -> Void
    ) {
        self.request = request
        self.currentModelPath = currentModelPath
        self.onCancel = onCancel
        self.onConfirm = onConfirm
        _template = State(initialValue: request.suggestedTemplate ?? .qa)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text(request.suggestedTemplate == nil ? "未找到模板信息" : "确认 State 配置")
                    .font(.title3.weight(.semibold))
                Text(request.source.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if request.suggestedTemplate == nil {
                Text("这个 State 没有可读取的训练模板。请选择实际训练时使用的格式，确认前不会开始生成。")
                    .foregroundStyle(.secondary)
            }

            Picker("模板", selection: $template) {
                ForEach(ChatTemplate.allCases) { value in
                    Text(value.displayName).tag(value)
                }
            }
            .pickerStyle(.segmented)

            if request.requiresModelChoice {
                VStack(alignment: .leading, spacing: 8) {
                    Text("模型名称不同")
                        .font(.headline)
                    LabeledContent("当前模型", value: currentModelName)
                    LabeledContent("训练模型", value: request.suggestedModelName ?? L10n.string("未记录"))
                    Text("名称不是可靠指纹；继续使用当前模型时，后端仍会在加载 State 时校验层数与每层 shape。")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(12)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
            }

            Divider()

            HStack {
                Button("取消", role: .cancel, action: onCancel)
                Spacer()
                if request.requiresModelChoice {
                    Button("仍用当前模型") { onConfirm(template, false) }
                    if canSwitchToSuggestedModel {
                        Button("切换到训练模型") { onConfirm(template, true) }
                            .buttonStyle(.borderedProminent)
                    }
                } else {
                    Button("加载 State") { onConfirm(template, request.autoSwitchModel) }
                        .buttonStyle(.borderedProminent)
                }
            }
        }
        .padding(22)
        .frame(width: 520)
    }

    private var currentModelName: String {
        currentModelPath.isEmpty
            ? L10n.string("未选择")
            : URL(fileURLWithPath: currentModelPath).lastPathComponent
    }

    private var canSwitchToSuggestedModel: Bool {
        guard let path = request.suggestedModelPath else { return false }
        return FileManager.default.fileExists(atPath: path)
    }
}
