//
//  ContentView.swift
//  Preen
//
//  主容器。NavigationSplitView 两栏:
//   - Sidebar(训练/对话/State 库 + 模型选择器钉底部)
//   - detail(按 selection 切面板)
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

    /// 模型选择器(toolbar 下拉菜单,靠近窗口右边缘)。
    private var modelPickerMenu: some View {
        Menu {
            if appState.recentModels.isEmpty {
                Text("还没有模型记录")
            } else {
                ForEach(appState.recentModels) { model in
                    Button {
                        appState.selectModel(path: model.path)
                    } label: {
                        if appState.modelPath == model.path {
                            Label(model.displayName, systemImage: "checkmark")
                        } else {
                            Text(model.displayName)
                        }
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
                Text(appState.modelPath.isEmpty
                     ? "选择模型"
                     : URL(fileURLWithPath: appState.modelPath).lastPathComponent)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .help("选择 RWKV-7 模型")
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
            Text("请先在侧边栏底部选择模型")
                .font(.title3)
            Text("对话面板需要一个 HF 格式的 RWKV-7 模型目录")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

}
