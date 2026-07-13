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
    @State private var showingModelList = false

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
                    .fill(backendColor)
                    .frame(width: 8, height: 8)
                Text(backendEntryLabel)
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

    private var backendColor: Color {
        if backendHasError { return .red }
        if backendIsTransitioning { return .orange }
        return .green
    }

    private var backendEntryLabel: String {
        if backendHasError { return "后端异常 · 查看日志" }
        if backendIsTransitioning { return "后端切换中" }
        return "后端状态与日志"
    }

    private var backendHasError: Bool {
        let backend = appState.backendStore
        return backend.runtime.phase == .unavailable ||
            backend.training.phase == .failed || backend.inference.phase == .failed
    }

    private var backendIsTransitioning: Bool {
        let backend = appState.backendStore
        return backend.runtime.phase == .checking ||
            backend.training.phase == .starting || backend.training.phase == .stopping ||
            backend.inference.phase == .starting || backend.inference.phase == .stopping
    }

    private var navList: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(SidebarItem.allCases) { item in
                let isSelected = appState.selection == item
                Button {
                    appState.selection = item
                } label: {
                    Label(item.label, systemImage: item.systemImage)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .buttonStyle(.plain)
                .foregroundStyle(isSelected ? Color.accentColor : Color.primary)
                .background(
                    isSelected ? Color.accentColor.opacity(0.15) : .clear,
                    in: RoundedRectangle(cornerRadius: 8, style: .continuous)
                )
                .padding(.horizontal, 8)
            }
        }
    }

    private var modelPicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("模型")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button {
                // 用户每次展开列表都重新校验，路径失效的项不会继续显示。
                appState.validateRecentModels()
                showingModelList.toggle()
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "shippingbox")
                        .foregroundStyle(.secondary)
                    Text(appState.modelPath.isEmpty ? "选择模型" :
                         URL(fileURLWithPath: appState.modelPath).lastPathComponent)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .foregroundStyle(appState.modelPath.isEmpty ? .secondary : .primary)
                    Spacer(minLength: 4)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 7)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
            .preenGlassSurface(cornerRadius: 7, interactive: true)
            .popover(isPresented: $showingModelList, arrowEdge: .trailing) {
                recentModelList
            }
        }
    }

    private var recentModelList: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("最近使用的模型")
                .font(.headline)
                .padding(.horizontal, 10)
                .padding(.top, 8)

            if appState.recentModels.isEmpty {
                Text("还没有模型记录")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
            } else {
                ForEach(appState.recentModels) { model in
                    Button {
                        appState.selectModel(path: model.path)
                        showingModelList = false
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "checkmark")
                                .frame(width: 12)
                                .opacity(appState.modelPath == model.path ? 1 : 0)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(model.displayName)
                                    .lineLimit(1)
                                Text(model.path)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            Spacer()
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                }
            }

            Divider()
            Button {
                showingModelList = false
                pickModel()
            } label: {
                Label("选择其他模型…", systemImage: "folder")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(10)
        }
        .frame(width: 330)
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
}
