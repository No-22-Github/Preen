import SwiftUI

struct BackendStatusView: View {
    @Bindable var store: BackendStore
    @Environment(\.dismiss) private var dismiss
    @State private var showLogs = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("后端环境").font(.title2)
                Spacer()
                if store.runtime.phase == .checking {
                    ProgressView()
                        .controlSize(.small)
                    Text("检查中…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Button("重新检查") { Task { await store.checkRuntime() } }
                }
            }
            statusRow("运行时", status: store.runtime.message, phase: store.runtime.phase == .ready ? .ready : (store.runtime.phase == .checking ? .starting : .failed))
            statusRow("推理", status: store.inference.message, phase: store.inference.phase)
            statusRow("训练", status: store.training.message, phase: store.training.phase)

            if let report = store.runtime.report {
                Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 8) {
                    GridRow { Text("Python").foregroundStyle(.secondary); Text(report.python) }
                    GridRow { Text("平台").foregroundStyle(.secondary); Text("\(report.machine) · Metal \(report.metalAvailable ? "可用" : "不可用")") }
                    if let memory = report.memorySizeGB, let workingSet = report.workingSetGB {
                        GridRow { Text("内存").foregroundStyle(.secondary); Text(String(format: "物理 %.2fG · working set %.2fG", memory, workingSet)) }
                    }
                }
            }
            Spacer()
            HStack {
                Button("查看日志") { showLogs = true }
                Spacer()
                Button("完成") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 560, height: 380)
        .sheet(isPresented: $showLogs) { BackendLogSheet(store: store) }
    }

    private func statusRow(_ label: String, status: String, phase: WorkerPhase) -> some View {
        HStack {
            Circle().fill(phase.statusColor).frame(width: 9, height: 9)
            Text(label).frame(width: 60, alignment: .leading)
            Text(status).foregroundStyle(.secondary).textSelection(.enabled)
            Spacer()
        }
    }
}

extension WorkerPhase {
    var statusColor: Color {
        switch self {
        case .ready: return .green
        case .starting, .running, .stopping: return .orange
        case .failed: return .red
        case .idle: return .secondary
        }
    }
}
