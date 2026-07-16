import SwiftUI

struct TrainingHistoryView: View {
    @Bindable var appState: AppState
    @State private var statusFilter: TrainingRunStatus?
    @State private var importError: String?
    @State private var deleteCandidate: TrainingRun?
    @State private var deleteError: String?
    // 默认收起:inspector 默认展开会占 220pt,把中间详情夹窄甚至撑窗(见 min 220 注释)。
    // 换 V4 key 让默认收起对所有人生效(V3 存过 true 的老用户不会卡在展开态)。
    @SceneStorage("trainingHistoryInspectorPresentedV4") private var isInspectorPresented = false

    private var filteredRuns: [TrainingRun] {
        appState.runs.filter { statusFilter == nil || $0.status == statusFilter }
    }

    private var selectedRun: TrainingRun? {
        filteredRuns.first { $0.id == appState.selectedRunID }
    }

    var body: some View {
        Group {
            if appState.runs.isEmpty {
                noRunsView
            } else {
                historySplitView
            }
        }
        .task {
            await appState.refreshRuns()
            if appState.selectedRunID == nil {
                appState.selectedRunID = appState.runs.first?.id
            }
        }
        .onChange(of: statusFilter) { _, _ in
            if selectedRun == nil {
                appState.selectedRunID = filteredRuns.first?.id
            }
        }
        .confirmationDialog(
            "删除这条训练记录？",
            isPresented: Binding(
                get: { deleteCandidate != nil },
                set: { if !$0 { deleteCandidate = nil } }
            ),
            titleVisibility: .visible
        ) {
            if let run = deleteCandidate {
                Button("删除记录", role: .destructive) {
                    deleteCandidate = nil
                    Task {
                        do {
                            try await appState.deleteRun(id: run.id)
                        } catch {
                            deleteError = error.localizedDescription
                        }
                    }
                }
            }
            Button("取消", role: .cancel) { deleteCandidate = nil }
        } message: {
            Text("相关事件和运行日志也会被删除。State、PTH 和 Checkpoint 文件不会被删除。")
        }
        .alert("无法删除训练记录", isPresented: Binding(
            get: { deleteError != nil }, set: { if !$0 { deleteError = nil } }
        )) {
            Button("关闭") { deleteError = nil }
        } message: {
            Text(deleteError ?? "")
        }
        .alert("无法导入 State", isPresented: Binding(
            get: { importError != nil }, set: { if !$0 { importError = nil } }
        )) {
            Button("关闭") { importError = nil }
        } message: {
            Text(importError ?? "")
        }
    }

    private var historySplitView: some View {
        historyLayout
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .toolbar {
            ToolbarItem(
                id: "training-history-inspector",
                placement: .primaryAction,
                showsByDefault: true
            ) {
                Button {
                    isInspectorPresented.toggle()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "sidebar.trailing")
                        Text(L10n.string(isInspectorPresented ? "隐藏参数" : "参数"))
                    }
                }
                .help(L10n.string(isInspectorPresented ? "隐藏参数与结果" : "显示参数与结果"))
                .accessibilityValue(L10n.string(isInspectorPresented ? "已显示" : "已隐藏"))
                .disabled(selectedRun == nil)
            }
        }
        .inspector(isPresented: $isInspectorPresented) {
            if let run = selectedRun {
                TrainingRunInspectorView(run: run)
                    .inspectorColumnWidth(min: 240, ideal: 300, max: 360)
            } else {
                ContentUnavailableView(
                    "没有选中的记录",
                    systemImage: "sidebar.trailing",
                    description: Text("选择一条训练记录查看参数与结果。")
                )
            }
        }
    }

    private var historyLayout: some View {
        HStack(spacing: 0) {
            recordListPane
                // 可压缩范围,避免窗口偏窄时硬固定撑出 detail 区(原 width: 260)。
                .frame(minWidth: 200, idealWidth: 240, maxWidth: 260)

            Divider()

            selectedRunDetail
        }
    }

    private var recordListPane: some View {
        VStack(spacing: 0) {
            HStack {
                Picker("状态", selection: $statusFilter) {
                    Text("全部").tag(TrainingRunStatus?.none)
                    ForEach(TrainingRunStatus.allCases, id: \.self) { status in
                        Text(status.label).tag(Optional(status))
                    }
                }
                Button { importState() } label: {
                    Image(systemName: "plus")
                }
                .help("登记外部 State…")
                .labelStyle(.iconOnly)
            }
            .padding(10)

            if filteredRuns.isEmpty {
                ContentUnavailableView {
                    Label("没有符合筛选条件的记录", systemImage: "line.3.horizontal.decrease.circle")
                } description: {
                    Text("当前状态筛选下没有训练记录。")
                } actions: {
                    Button("显示全部") { statusFilter = nil }
                }
            } else {
                List(filteredRuns, selection: $appState.selectedRunID) { run in
                    HStack(spacing: 9) {
                        Image(systemName: run.status.systemImage)
                            .foregroundStyle(run.status.color)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(runDisplayName(run))
                                .lineLimit(1)
                            HStack {
                                Text(run.kind == .imported ? L10n.string("外部导入") : run.status.label)
                                Text(run.createdAt, format: .dateTime.month().day().hour().minute())
                            }
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        }
                    }
                    .tag(run.id)
                    .contextMenu {
                        Button("删除记录…", role: .destructive) {
                            requestDelete(run)
                        }
                        .disabled(!run.status.isTerminal)
                    }
                }
                .listStyle(.inset)
                .onDeleteCommand {
                    if let selectedRun { requestDelete(selectedRun) }
                }
            }
        }
    }

    @ViewBuilder
    private var selectedRunDetail: some View {
        if let run = selectedRun {
            TrainingRunDetailView(
                run: run,
                appState: appState,
                onDelete: { requestDelete(run) }
            )
        } else if filteredRuns.isEmpty {
            ContentUnavailableView {
                Label("没有符合筛选条件的记录", systemImage: "line.3.horizontal.decrease.circle")
            } description: {
                Text("请调整状态筛选。")
            } actions: {
                Button("显示全部") { statusFilter = nil }
            }
        } else {
            ContentUnavailableView(
                "选择一条训练记录",
                systemImage: "clock.arrow.circlepath",
                description: Text("成功、失败、取消和中断记录都会保留。")
            )
        }
    }

    private func runDisplayName(_ run: TrainingRun) -> String {
        run.config.map {
            URL(fileURLWithPath: $0.dataPath).lastPathComponent
        } ?? URL(fileURLWithPath: run.artifacts.statePath ?? "State").lastPathComponent
    }

    private var noRunsView: some View {
        ContentUnavailableView {
            Label("还没有训练记录", systemImage: "clock.arrow.circlepath")
        } description: {
            Text("完成一次训练或登记已有 State 后，记录会显示在这里。")
        } actions: {
            HStack {
                Button("开始训练") { appState.selection = .training }
                    .buttonStyle(.borderedProminent)
                Button("登记外部 State…") { importState() }
            }
        }
    }

    private func requestDelete(_ run: TrainingRun) {
        guard run.status.isTerminal else { return }
        deleteCandidate = run
    }

    private func importState() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.data]
        panel.allowsMultipleSelection = false
        panel.prompt = L10n.string("登记 State")
        guard panel.runModal() == .OK, let stateURL = panel.url else { return }
        let metadataURL = stateURL.deletingPathExtension().appendingPathExtension("meta.json")
        Task {
            do {
                let run = try await appState.runRepository.registerImportedState(
                    stateURL: stateURL,
                    metadataURL: metadataURL
                )
                await appState.refreshRuns()
                appState.selectedRunID = run.id
            } catch {
                importError = error.localizedDescription
            }
        }
    }
}
