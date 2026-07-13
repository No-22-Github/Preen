import SwiftUI

struct TrainingHistoryView: View {
    @Bindable var appState: AppState
    @State private var statusFilter: TrainingRunStatus?
    @State private var importError: String?

    private var filteredRuns: [TrainingRun] {
        appState.runs.filter { statusFilter == nil || $0.status == statusFilter }
    }

    private var selectedRun: TrainingRun? {
        appState.runs.first { $0.id == appState.selectedRunID }
    }

    var body: some View {
        HSplitView {
            VStack(spacing: 0) {
                HStack {
                    Picker("状态", selection: $statusFilter) {
                        Text("全部").tag(TrainingRunStatus?.none)
                        ForEach(TrainingRunStatus.allCases, id: \.self) { status in
                            Text(status.label).tag(Optional(status))
                        }
                    }
                    Button { importState() } label: { Image(systemName: "plus") }
                        .help("登记外部 State")
                        .labelStyle(.iconOnly)
                }
                .padding(10)
                List(filteredRuns, selection: $appState.selectedRunID) { run in
                    HStack(spacing: 9) {
                        Image(systemName: run.status.systemImage).foregroundStyle(run.status.color)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(run.config.map { URL(fileURLWithPath: $0.dataPath).lastPathComponent }
                                 ?? URL(fileURLWithPath: run.artifacts.statePath ?? "State").lastPathComponent)
                                .lineLimit(1)
                            HStack {
                                Text(run.kind == .imported ? "外部导入" : run.status.label)
                                Text(run.createdAt, format: .dateTime.month().day().hour().minute())
                            }
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        }
                    }
                    .tag(run.id)
                }
                .listStyle(.inset)
            }
            .frame(minWidth: 260, idealWidth: 300, maxWidth: 360)

            if let run = selectedRun {
                TrainingRunDetailView(run: run, appState: appState)
            } else {
                ContentUnavailableView(
                    "选择一条训练记录",
                    systemImage: "clock.arrow.circlepath",
                    description: Text("成功、失败、取消和中断记录都会保留。")
                )
            }
        }
        .task {
            await appState.refreshRuns()
            if appState.selectedRunID == nil { appState.selectedRunID = appState.runs.first?.id }
        }
        .alert("无法导入 State", isPresented: Binding(
            get: { importError != nil }, set: { if !$0 { importError = nil } }
        )) { Button("关闭") { importError = nil } } message: { Text(importError ?? "") }
    }

    private func importState() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.data]
        panel.allowsMultipleSelection = false
        panel.prompt = "登记 State"
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
