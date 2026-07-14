//
//  ContentView.swift
//  Preen
//
//  主容器。NavigationSplitView 两栏:
//   - Sidebar(训练/对话/训练记录/工具箱)
//   - detail(按 selection 切面板)
//   - 模型选择器位于 detail toolbar 右上角
//
//  最小 1000×680(design.md §3)。
//

import SwiftUI

struct ContentView: View {
    @Bindable var appState: AppState

    var body: some View {
        VStack(spacing: 0) {
            NavigationSplitView {
                Sidebar(appState: appState)
                    .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 280)
            } detail: {
                detail
                    .toolbar {
                        ToolbarItem(placement: .primaryAction) {
                            modelPickerMenu
                        }
                    }
            }
            Divider()
            GlobalStatusBar(appState: appState)
        }
        .frame(minWidth: 1000, minHeight: 680)
    }

    /// 模型选择器(toolbar 下拉菜单 + 精度胶囊,靠近窗口右边缘)。
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
            .help(isQuantized ? "INT8 · 仅支持推理,不支持训练" : "BF16 标准精度")
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
                          onConnect: { appState.connectInference() },
                          onDisconnect: { appState.disconnectInference() })
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
            Text("也可以使用窗口右上角的模型菜单。")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

}
