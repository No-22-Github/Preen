import SwiftUI

struct TrainingRunDetailView: View {
    let run: TrainingRun
    @Bindable var appState: AppState
    var onDelete: () -> Void

    @State private var events: [TrainEvent] = []
    @State private var stderrLog = ""
    @State private var replayStore: TrainStore?
    @State private var comparisons: [SavedComparison] = []
    @State private var exportMessage: String?
    @State private var exportError: String?
    @State private var metadata: StateMetadata?
    @State private var isLogExpanded = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                Divider()
                if run.kind == .training || metadata?.result != nil {
                    TrainingResultSummaryView(facts: explanation)
                }
                if let replayStore, !replayStore.lossPoints.isEmpty {
                    TrainingChartView(store: replayStore)
                        .frame(minHeight: 460)
                }
                artifacts
                if !comparisons.isEmpty { savedComparisonsSection }
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
                Text(L10n.string(run.kind == .imported ? "外部 State" : "训练记录")).font(.title2)
                Text(run.status.label).foregroundStyle(.secondary)
                Spacer()
                if run.artifacts.statePath != nil {
                    Button("比较效果") { goToChat() }
                        .buttonStyle(.borderedProminent)
                }
                Menu("操作") {
                    Button("复制诊断") { copyDiagnostics() }
                    Button("复制日志") { copyLog() }
                    Button("导出事件") { exportEvents() }
                    Button("在 Finder 中显示") { revealRun() }
                    if run.artifacts.statePath != nil {
                        Button("导出 .pth") { exportPth() }
                    }
                    Divider()
                    Button("删除记录…", role: .destructive, action: onDelete)
                        .disabled(!run.status.isTerminal)
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
            Text(L10n.string(label)).foregroundStyle(.secondary).frame(width: 110, alignment: .leading)
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
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeInOut(duration: 0.15)) {
                    isLogExpanded.toggle()
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .rotationEffect(.degrees(isLogExpanded ? 90 : 0))
                    Text("运行日志").font(.headline)
                    Spacer()
                    Text("训练进程的标准错误输出")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 32, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityValue(L10n.string(isLogExpanded ? "已展开" : "已折叠"))

            if isLogExpanded {
                GroupBox {
            ScrollView {
                Text(stderrLog)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 180)
            .padding(8)
                }
                .transition(.opacity)
            }
        }
    }

    private var savedComparisonsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("已保存比较").font(.headline)
            ForEach(comparisons.reversed()) { comparison in
                DisclosureGroup {
                    HStack(alignment: .top, spacing: 12) {
                        savedComparisonText("无 State（基线）", comparison.baselineText)
                        savedComparisonText("有 State", comparison.stateText)
                    }
                    Text("\(comparison.template) · \(comparison.reasoning ? comparison.think : "off") · seed \(comparison.genConfig.seed)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(comparison.prompt).lineLimit(1)
                        Text(comparison.createdAt, format: .dateTime.year().month().day().hour().minute())
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private func savedComparisonText(_ title: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(L10n.string(title)).font(.caption.weight(.semibold))
            Text(text).textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
    }

    private func loadDetails() async {
        events = await appState.runRepository.loadEvents(id: run.id)
        stderrLog = await appState.runRepository.loadStderr(id: run.id)
        comparisons = await appState.runRepository.loadComparisons(runID: run.id)
        if let path = run.artifacts.metadataPath {
            metadata = try? StateMetadata.load(from: URL(fileURLWithPath: path))
        } else {
            metadata = nil
        }
        let store = TrainStore()
        store.replay(events: events)
        replayStore = store
    }

    private func copyLog() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(stderrLog, forType: .string)
    }

    private var explanation: TrainingResultExplanation {
        TrainingResultExplanation(run: run, events: events, metadata: metadata)
    }

    private func copyDiagnostics() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(explanation.diagnosticText, forType: .string)
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
        appState.goToChat(
            stateURL: URL(fileURLWithPath: path),
            trainingConfig: run.config,
            runID: run.id
        )
    }

    private func exportPth() {
        guard let statePath = run.artifacts.statePath else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = URL(fileURLWithPath: statePath)
            .deletingPathExtension().lastPathComponent + ".pth"
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        exportMessage = L10n.string("正在导出…")
        Task {
            do {
                let result = try await StateExportRunner().export(
                    state: URL(fileURLWithPath: statePath), output: destination
                )
                _ = try await appState.runRepository.setPthArtifact(runID: run.id, path: result.output.path)
                await appState.refreshRuns()
                exportMessage = L10n.format("已导出 %@", result.output.lastPathComponent)
            } catch {
                exportMessage = nil
                exportError = error.localizedDescription
            }
        }
    }
}

/// 训练记录的参数与结果 Inspector；主内容保留曲线、产物和日志。
struct TrainingRunInspectorView: View {
    let run: TrainingRun
    @State private var metadata: StateMetadata?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("参数与结果")
                    .font(.headline)

                inspectorSection("来源") {
                    inspectorRow("创建时间", run.createdAt.formatted(date: .abbreviated, time: .standard))
                    if let config = run.config {
                        inspectorPathRow("模型", config.modelPath)
                        inspectorPathRow("数据", config.dataPath)
                        inspectorRow("模板", config.template)
                    } else if let metadata {
                        inspectorPathRow("模型", metadata.model)
                        inspectorPathRow("数据", metadata.data)
                        inspectorRow("模板", metadata.template ?? L10n.string("未记录"))
                    }
                    if let hash = run.summary.dataHash ?? metadata?.dataSHA256 {
                        inspectorRow("数据 SHA-256", abbreviatedHash(hash), help: hash)
                    }
                }

                if let config = run.config {
                    inspectorSection("训练参数") {
                        inspectorRow("学习率", String(config.learningRate))
                        inspectorRow("上下文长度", "\(config.contextLength)")
                        inspectorRow("训练轮数", "\(config.epochs)")
                        inspectorRow("随机种子", "\(config.seed)")
                    }
                }

                inspectorSection("结果") {
                    if let epochs = run.summary.actualEpochs ?? metadata?.result?.epochsRun {
                        inspectorRow("实际轮数", "\(epochs)")
                    }
                    if let loss = run.summary.finalLoss ?? metadata?.result?.finalLoss {
                        inspectorRow("Final loss", String(format: "%.4f", loss))
                    }
                    if let held = run.summary.heldOutLoss ?? metadata?.result?.bestHeldOutLoss {
                        inspectorRow("Held-out loss", String(format: "%.4f", held))
                    }
                    if let std = run.summary.stateStd ?? metadata?.result?.finalStateStd {
                        inspectorRow("State std", String(format: "%.4f", std))
                    }
                    if let elapsed = run.summary.elapsedSeconds ?? metadata?.result?.elapsed {
                        inspectorRow("耗时", TrainStore.formatDuration(elapsed))
                    }
                }
            }
            .padding(16)
        }
        .task(id: run.updatedAt) {
            if let path = run.artifacts.metadataPath {
                metadata = try? StateMetadata.load(from: URL(fileURLWithPath: path))
            } else {
                metadata = nil
            }
        }
    }

    private func inspectorSection<Content: View>(
        _ title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(L10n.string(title))
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                content()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func inspectorPathRow(_ label: String, _ path: String) -> some View {
        inspectorRow(label, URL(fileURLWithPath: path).lastPathComponent, help: path)
    }

    private func abbreviatedHash(_ hash: String) -> String {
        guard hash.count > 16 else { return hash }
        return "\(hash.prefix(12))…\(hash.suffix(4))"
    }

    @ViewBuilder
    private func inspectorRow(_ label: String, _ value: String, help: String? = nil) -> some View {
        GridRow {
            Text(L10n.string(label))
                .foregroundStyle(.secondary)
                .frame(width: 86, alignment: .trailing)
            if let help {
                inspectorValue(value)
                    .help(help)
            } else {
                inspectorValue(value)
            }
        }
    }

    private func inspectorValue(_ value: String) -> some View {
        Text(value)
            .lineLimit(2)
            .truncationMode(.middle)
            .textSelection(.enabled)
    }
}
