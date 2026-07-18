import SwiftUI

struct TrainingDoneView: View {
    @Bindable var store: TrainStore
    var onGoToChat: (URL, PersistedTrainingConfig?, UUID?) -> Void
    var onGoHome: () -> Void
    var onShowChart: () -> Void

    @State private var exportMessage: String?
    @State private var exportError: String?

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 48))
                    .foregroundStyle(.green)
                    .accessibilityHidden(true)

                Text("训练完成")
                    .font(.title.weight(.semibold))
                    .padding(.top, 14)

                TrainingResultSummaryView(facts: facts)
                    .frame(maxWidth: 760)
                    .padding(.top, 22)

                if let exportMessage {
                    Label(exportMessage, systemImage: "checkmark.circle")
                        .font(.caption)
                        .foregroundStyle(.green)
                        .padding(.top, 12)
                }

                actions
                    .padding(.top, 22)

                Button("返回首页") { onGoHome() }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .padding(.top, 14)
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 32)
            .frame(maxWidth: .infinity)
        }
        .alert("导出失败", isPresented: Binding(
            get: { exportError != nil },
            set: { if !$0 { exportError = nil } }
        )) {
            Button("关闭") { exportError = nil }
        } message: {
            Text(exportError ?? "")
        }
    }

    private var facts: TrainingResultExplanation {
        TrainingResultExplanation(store: store)
    }

    private var actions: some View {
        HStack(spacing: 10) {
            if let path = store.outputPath {
                Button {
                    onGoToChat(
                        URL(fileURLWithPath: path),
                        store.currentRun?.config,
                        store.currentRun?.id
                    )
                } label: {
                    Text("比较效果").frame(minWidth: 120)
                }
                .buttonStyle(.borderedProminent)

                Button("在 Finder 中显示") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                }
                .buttonStyle(.bordered)

                Menu("更多") {
                    Button("导出 .pth") { exportPth(statePath: path) }
                    Button("查看完整曲线", action: onShowChart)
                        .disabled(store.lossPoints.isEmpty)
                    Button("复制诊断", action: copyDiagnostics)
                }
            } else {
                Button("查看完整曲线", action: onShowChart)
                    .disabled(store.lossPoints.isEmpty)
                Button("复制诊断", action: copyDiagnostics)
            }
        }
    }

    private func copyDiagnostics() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(facts.diagnosticText, forType: .string)
    }

    private func exportPth(statePath: String) {
        let source = URL(fileURLWithPath: statePath)
        let panel = NSSavePanel()
        panel.nameFieldStringValue = source.deletingPathExtension().lastPathComponent + ".pth"
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        exportMessage = L10n.string("正在导出…")
        Task {
            do {
                let result = try await StateExportRunner().export(state: source, output: destination)
                try await store.recordPthArtifact(path: result.output.path)
                exportMessage = L10n.format("已导出 %@", result.output.lastPathComponent)
            } catch {
                exportMessage = nil
                exportError = error.localizedDescription
            }
        }
    }
}
