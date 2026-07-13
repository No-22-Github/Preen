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
        NavigationSplitView {
            Sidebar(appState: appState)
                .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 280)
        } detail: {
            detail
        }
        .frame(minWidth: 1000, minHeight: 680)
    }

    @ViewBuilder
    private var detail: some View {
        switch appState.selection {
        case .home:
            HomeView(appState: appState)
        case .training:
            TrainingPanel(store: appState.trainStore, modelPath: $appState.modelPath) { stateURL in
                appState.goToChat(stateURL: stateURL)
            }
        case .chat:
            if appState.modelPath.isEmpty {
                chatNeedsModel
            } else {
                ChatPanel(store: appState.chatStore,
                          modelPath: appState.modelPath,
                          injectedStatePath: $appState.injectedStatePath)
            }
        case .history:
            TrainingHistoryView(appState: appState)
        }
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
