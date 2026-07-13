import SwiftUI

struct TrainingRunDetailView: View {
    let run: TrainingRun
    @Bindable var appState: AppState

    @State private var events: [TrainEvent] = []
    @State private var stderrLog = ""
    @State private var metadata: StateMetadata?
    @State private var replayStore: TrainStore?
    @State private var exportMessage: String?
    @State private var exportError: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                Divider()
                summary
                if let replayStore, !replayStore.lossPoints.isEmpty {
                    TrainingChartView(store: replayStore)
                        .frame(height: 340)
                }
                artifacts
                if !stderrLog.isEmpty { logSection }
                EventLogView(events: events)
            }
            .padding(22)
        }
        .task(id: run.updatedAt) { await loadDetails() }
        .alert("导出失败", isPresented: Binding(
            get: { exportError != nil }, set: { if !$0 { exportError = nil } }
        )) { Button("关闭") { exportError = nil } } message: { Text(exportError ?? "") }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Image(systemName: run.status.systemImage).foregroundStyle(run.status.color)
                Text(run.kind == .imported ? "外部 State" : "训练记录").font(.title2)
                Text(run.status.label).foregroundStyle(.secondary)
                Spacer()
                Menu("操作") {
                    Button("复制日志") { copyLog() }
                    Button("导出事件") { exportEvents() }
                    Button("在 Finder 中显示") { revealRun() }
                    if run.artifacts.statePath != nil {
                        Button("去对话") { goToChat() }
                        Button("导出 .pth") { exportPth() }
                    }
                }
            }
            Text(run.id.uuidString.lowercased())
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
                .textSelection(.enabled)
            if let failure = run.failureMessage {
                Label(failure, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(run.status == .failed ? .red : .orange)
                    .textSelection(.enabled)
            }
            if let exportMessage {
                Label(exportMessage, systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.green)
            }
        }
    }

    private var summary: some View {
        Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 8) {
            detailRow("创建时间", run.createdAt.formatted(date: .abbreviated, time: .standard))
            if let config = run.config {
                detailRow("模型", config.modelPath)
                detailRow("数据", config.dataPath)
                detailRow("模板", config.template)
                detailRow("超参数", "lr \(config.learningRate) · ctx \(config.contextLength) · \(config.epochs) 轮 · seed \(config.seed)")
            } else if let metadata {
                detailRow("来源模型", metadata.model)
                detailRow("来源数据", metadata.data)
                detailRow("模板", metadata.template)
            }
            if let hash = run.summary.dataHash ?? metadata?.dataSHA256 { detailRow("数据 SHA-256", hash) }
            if let epochs = run.summary.actualEpochs ?? metadata?.result.epochsRun { detailRow("实际轮数", "\(epochs)") }
            if let loss = run.summary.finalLoss ?? metadata?.result.finalLoss { detailRow("Final loss", String(format: "%.4f", loss)) }
            if let held = run.summary.heldOutLoss ?? metadata?.result.bestHeldOutLoss { detailRow("Held-out loss", String(format: "%.4f", held)) }
            if let std = run.summary.stateStd ?? metadata?.result.finalStateStd { detailRow("State std", String(format: "%.4f", std)) }
            if let elapsed = run.summary.elapsedSeconds ?? metadata?.result.elapsed { detailRow("耗时", TrainStore.formatDuration(elapsed)) }
        }
        .textSelection(.enabled)
    }

    private func detailRow(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary).frame(width: 110, alignment: .leading)
            Text(value).lineLimit(2).truncationMode(.middle)
        }
    }

    private var artifacts: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("产物").font(.headline)
            artifactRow("State", run.artifacts.statePath)
            artifactRow("Metadata", run.artifacts.metadataPath)
            artifactRow("PTH", run.artifacts.pthPath)
            ForEach(Array(run.artifacts.checkpoints.enumerated()), id: \.offset) { index, path in
                artifactRow("Checkpoint \(index + 1)", path)
            }
            if run.artifacts.statePath == nil && run.artifacts.checkpoints.isEmpty {
                Text("这条记录没有产物").foregroundStyle(.secondary)
            }
        }
    }

    private func artifactRow(_ label: String, _ path: String?) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary).frame(width: 110, alignment: .leading)
            if let path {
                Text(path).font(.caption.monospaced()).lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                Spacer()
                Button { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) } label: {
                    Image(systemName: "folder")
                }
                .buttonStyle(.borderless)
            } else {
                Text("—").foregroundStyle(.tertiary)
            }
        }
    }

    private var logSection: some View {
        DisclosureGroup("stderr.log") {
            ScrollView {
                Text(stderrLog)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 180)
            .padding(8)
            .background(.quaternary, in: .rect)
        }
    }

    private func loadDetails() async {
        events = await appState.runRepository.loadEvents(id: run.id)
        stderrLog = await appState.runRepository.loadStderr(id: run.id)
        if let path = run.artifacts.metadataPath {
            metadata = try? StateMetadata.load(from: URL(fileURLWithPath: path))
        } else {
            metadata = nil
        }
        let store = TrainStore()
        events.forEach { store.consume(event: $0) }
        replayStore = store
    }

    private func copyLog() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(stderrLog, forType: .string)
    }

    private func exportEvents() {
        let source = appState.runRepository.rootURL
            .appendingPathComponent(run.id.uuidString.lowercased())
            .appendingPathComponent(RunRepository.eventsFilename)
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "\(run.id.uuidString.lowercased())-events.jsonl"
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        do { try Data(contentsOf: source).write(to: destination, options: .atomic) }
        catch { exportError = error.localizedDescription }
    }

    private func revealRun() {
        NSWorkspace.shared.open(appState.runRepository.rootURL
            .appendingPathComponent(run.id.uuidString.lowercased()))
    }

    private func goToChat() {
        guard let path = run.artifacts.statePath else { return }
        appState.goToChat(stateURL: URL(fileURLWithPath: path))
    }

    private func exportPth() {
        guard let statePath = run.artifacts.statePath else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = URL(fileURLWithPath: statePath)
            .deletingPathExtension().lastPathComponent + ".pth"
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        exportMessage = "正在导出…"
        Task {
            do {
                let result = try await StateExportRunner().export(
                    state: URL(fileURLWithPath: statePath), output: destination
                )
                _ = try await appState.runRepository.setPthArtifact(runID: run.id, path: result.output.path)
                await appState.refreshRuns()
                exportMessage = "已导出 \(result.output.lastPathComponent)"
            } catch {
                exportMessage = nil
                exportError = error.localizedDescription
            }
        }
    }
}
