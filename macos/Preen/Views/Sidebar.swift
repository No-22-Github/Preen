//
//  Sidebar.swift
//  Preen
//
//  侧边栏。design.md §3 铁律:
//   - **模型选择器钉在侧边栏底部**(不是每面板各选一次)。
//     理由:serve 单进程单模型,换模型 = 重启 sidecar;且连续 load_model 累积 Metal 内存池。
//   - 单窗口 NavigationSplitView 两栏 + 常驻底部状态栏。
//
//  本期 placeholder:State 库面板未做,点击显示「待 #9 实现」。
//

import SwiftUI

struct Sidebar: View {
    @Bindable var appState: AppState
    @State private var showingBackendStatus = false

    var body: some View {
        VStack(spacing: 0) {
            // 导航项。
            navList
                .padding(.top, 12)

            Spacer()

            Divider()
            backendStatus
                .padding(.horizontal, 12)
                .padding(.top, 10)
            // 模型选择器(钉底部)。
            modelPicker
                .padding(12)
        }
        .frame(minWidth: 200)
        .sheet(isPresented: $showingBackendStatus) {
            BackendStatusView(store: appState.backendStore)
        }
    }

    private var backendStatus: some View {
        Button { showingBackendStatus = true } label: {
            HStack(spacing: 8) {
                Circle()
                    .fill(runtimeColor)
                    .frame(width: 8, height: 8)
                Text(appState.backendStore.runtime.message)
                    .font(.caption)
                    .lineLimit(1)
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("查看 Python、MLX 与后端日志")
    }

    private var runtimeColor: Color {
        switch appState.backendStore.runtime.phase {
        case .checking: return .orange
        case .ready: return .green
        case .unavailable: return .red
        }
    }

    private var navList: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(SidebarItem.allCases) { item in
                Button {
                    appState.selection = item
                } label: {
                    Label(item.label, systemImage: item.systemImage)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .padding(.vertical, 6)
                .padding(.horizontal, 12)
                .background(appState.selection == item ? Color.accentColor.opacity(0.15) : .clear,
                            in: .rect)
            }
        }
    }

    private var modelPicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("模型")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Text(appState.modelPath.isEmpty ? "(未选)" :
                     URL(fileURLWithPath: appState.modelPath).lastPathComponent)
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .foregroundStyle(appState.modelPath.isEmpty ? .secondary : .primary)
                Spacer()
                Button("选…") { pickModel() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
        }
    }

    private func pickModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "选择模型目录"
        if panel.runModal() == .OK, let url = panel.url {
            // 换模型 = 重启 serve(design.md §3 铁律)。
            if appState.chatStore.isConnected {
                appState.chatStore.disconnect()
            }
            appState.modelPath = url.path
        }
    }
}
