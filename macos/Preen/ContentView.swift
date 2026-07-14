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
    /// 待执行的 state 动作(有聊天记录时拦截,确认后才执行,与垃圾桶同逻辑)。
    @State private var pendingStateAction: PendingStateAction?

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
                                Button(action: appState.disconnectInference) {
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
                                        Text("生成参数…")
                                    }
                                }
                                .help("温度、top_p、惩罚等生成参数")
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
            pendingStateAction == .load ? "替换 State 会清空当前会话？" : "卸下 State 会清空当前会话？",
            isPresented: Binding(
                get: { pendingStateAction != nil },
                set: { if !$0 { pendingStateAction = nil } }
            ),
            titleVisibility: .visible
        ) {
            Button(pendingStateAction == .load ? "替换" : "卸下", role: .destructive) {
                let action = pendingStateAction
                pendingStateAction = nil
                switch action {
                case .load: performLoadState()
                case .clear: performClearState()
                case .none: break
                }
            }
            Button("取消", role: .cancel) { pendingStateAction = nil }
        } message: {
            Text("加载或卸下 State 都会重置会话历史，此操作无法撤销。")
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
            .help("选择 RWKV-7 模型")

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
            var hint = AttributedString(" · 仅推理")
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
            .help(isQuantized ? "INT8 · 仅支持推理，不支持训练" : "BF16 标准精度")
            .contentShape(Rectangle())
    }

    /// Menu label 文字:模型名用 semibold。用 AttributedString 携带 font 属性,
    /// 能穿透 Menu label 的系统样式覆盖(.fontWeight 修饰符会被 Menu 忽略)。
    private var menuLabelText: AttributedString {
        if appState.modelPath.isEmpty {
            return AttributedString("选择模型")
        }
        var attr = AttributedString(URL(fileURLWithPath: appState.modelPath).lastPathComponent)
        attr.font = .body.weight(.semibold)
        return attr
    }

    private func pickModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "选择模型目录"
        if panel.runModal() == .OK, let url = panel.url {
            appState.selectModel(path: url.path)
        }
    }

    /// 请求加载/替换 state:有聊天记录时先拦截确认(加载会清空会话)。
    private func requestLoadState() {
        if appState.chatStore.messages.isEmpty {
            performLoadState()
        } else {
            pendingStateAction = .load
        }
    }

    /// 请求卸下 state:有聊天记录时先拦截确认(卸下会清空会话)。
    private func requestClearState() {
        if appState.chatStore.messages.isEmpty {
            performClearState()
        } else {
            pendingStateAction = .clear
        }
    }

    /// 实际执行:弹文件选择器 → 应用 state。
    private func performLoadState() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.data]  // .npz 走 UTI data
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            appState.chatStore.setState(path: url.path)
        }
    }

    /// 实际执行:卸下 state(已连接走后端 set_state(nil) 重置会话) + 清跨面板意图。
    private func performClearState() {
        appState.chatStore.clearState()
        appState.injectedStatePath = nil
    }

    /// 当前对话 state 的文件名(nil = 未选);驱动 toolbar 胶囊显示。
    private var chatStateBadgeLabel: String? {
        guard let path = appState.chatStore.statePath else { return nil }
        return URL(fileURLWithPath: path).lastPathComponent
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
                onStart: { appState.startTraining(config: $0) },
                onGoToChat: { appState.goToChat(stateURL: $0) }
            )
        case .chat:
            if appState.modelPath.isEmpty {
                chatNeedsModel
            } else {
                ChatPanel(store: appState.chatStore,
                          modelPath: appState.modelPath,
                          injectedStatePath: $appState.injectedStatePath,
                          isShowingGenerationParameters: $isShowingChatGenerationParameters,
                          onConnect: { appState.connectInference() })
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

/// 待执行的对话 state 动作(加载/卸下),用于有聊天记录时的确认拦截。
private enum PendingStateAction {
    case load   // 加载/替换 state
    case clear  // 卸下 state
}
