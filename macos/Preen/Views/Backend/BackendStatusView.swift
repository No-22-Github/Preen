import AppKit
import SwiftUI

struct BackendStatusView: View {
    @Bindable var store: BackendStore
    @Environment(\.dismiss) private var dismiss
    @State private var showLogs = false
    @State private var showComponents = false
    @State private var didCopy = false
    @State private var copyResetTask: Task<Void, Never>?

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, 20)
                .padding(.vertical, 16)

            Divider()

            Form {
                Section("运行状态") {
                    statusRow("运行时", status: store.runtime.message, phase: runtimeWorkerPhase)
                    statusRow("推理服务", status: store.inference.message, phase: store.inference.phase)
                    statusRow("训练任务", status: store.training.message, phase: store.training.phase)
                }

                if let report = store.runtime.report {
                    Section("设备") {
                        detailRow(
                            "芯片",
                            value: report.chipName
                                ?? (report.appleSilicon ? "Apple Silicon" : report.machine),
                            note: report.operatingSystemLabel
                        )
                        detailRow(
                            "统一内存",
                            value: report.memorySizeLabel ?? "未知",
                            note: "设备物理容量"
                        )
                        detailRow(
                            "MLX 工作集上限",
                            value: report.workingSetLabel ?? "未知",
                            note: "建议上限，非当前占用"
                        )
                    }

                    Section {
                        componentToggle
                        if showComponents {
                            componentRow(
                                "系统", value: report.operatingSystemLabel, available: true
                            )
                            componentRow("Python", value: report.python, available: true)
                            componentRow(
                                "MLX", value: moduleValue(report.mlx), available: report.mlx.ok
                            )
                            componentRow(
                                "MLX-LM", value: moduleValue(report.mlxLM),
                                available: report.mlxLM.ok
                            )
                            componentRow(
                                "NumPy", value: moduleValue(report.numpy),
                                available: report.numpy.ok
                            )
                            componentRow(
                                "ml-dtypes", value: moduleValue(report.mlDtypes),
                                available: report.mlDtypes.ok
                            )
                            componentRow(
                                "Metal", value: report.metalAvailable ? "可用" : "不可用",
                                available: report.metalAvailable
                            )
                        }
                    }
                } else if store.runtime.phase == .unavailable {
                    Section {
                        Label(
                            "未取得结构化环境报告；诊断信息仍可复制应用版本与运行状态。",
                            systemImage: "exclamationmark.triangle.fill"
                        )
                        .foregroundStyle(.secondary)
                    }
                }
            }
            .formStyle(.grouped)

            Divider()
            footer
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(.bar)
        }
        .frame(width: 600, height: 500)
        .sheet(isPresented: $showLogs) { BackendLogSheet(store: store) }
        .onDisappear { copyResetTask?.cancel() }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("后端环境")
                    .font(.title2.weight(.semibold))
                if let checkedAt = store.runtime.checkedAt {
                    Text("检查于 \(checkedAt, style: .relative)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("检查 Python、MLX 与 Metal")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            HStack(spacing: 6) {
                if store.runtime.phase == .checking {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Image(systemName: runtimeWorkerPhase.statusSymbol)
                        .foregroundStyle(runtimeWorkerPhase.statusColor)
                }
                Text(runtimeSummary)
                    .font(.subheadline.weight(.medium))
            }
            .accessibilityElement(children: .combine)

            if store.runtime.phase != .checking {
                Button {
                    Task { await store.checkRuntime() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("重新检查")
                .accessibilityLabel("重新检查")
            }
        }
    }

    private func statusRow(_ label: String, status: String, phase: WorkerPhase) -> some View {
        LabeledContent(label) {
            HStack(spacing: 7) {
                Image(systemName: phase.statusSymbol)
                    .foregroundStyle(phase.statusColor)
                Text(status)
                    .lineLimit(1)
                    .help(status)
                    .textSelection(.enabled)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private func detailRow(_ label: String, value: String, note: String) -> some View {
        LabeledContent {
            Text(value)
                .fontWeight(.medium)
                .textSelection(.enabled)
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .help(note)
            }
        }
    }

    private var componentToggle: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.16)) {
                showComponents.toggle()
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .rotationEffect(.degrees(showComponents ? 90 : 0))
                Text("组件版本")
                Spacer()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("组件版本")
        .accessibilityValue(showComponents ? "已展开" : "已折叠")
    }

    private func componentRow(_ label: String, value: String, available: Bool) -> some View {
        LabeledContent(label) {
            HStack(spacing: 7) {
                if !available {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.red)
                }
                Text(value)
                    .monospacedDigit()
                    .lineLimit(1)
                    .help(value)
                    .textSelection(.enabled)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var footer: some View {
        HStack(spacing: 10) {
            Button(didCopy ? "已复制" : "复制诊断信息", action: copyDiagnostics)
            .help("复制适合粘贴到 Issue 的 Markdown，不包含序列号、日志和本地路径")

            Button("诊断日志…") { showLogs = true }

            Spacer()

            Button("完成") { dismiss() }
                .keyboardShortcut(.defaultAction)
        }
    }

    private var runtimeWorkerPhase: WorkerPhase {
        switch store.runtime.phase {
        case .checking: return .starting
        case .ready: return .ready
        case .unavailable: return .failed
        }
    }

    private var runtimeSummary: String {
        switch store.runtime.phase {
        case .checking: return "检查中…"
        case .ready: return "后端正常"
        case .unavailable: return "运行时异常"
        }
    }

    private func moduleValue(_ module: DoctorModule) -> String {
        guard module.ok else { return "不可用" }
        return module.version ?? "可用"
    }

    @MainActor
    private func copyDiagnostics() {
        let info = Bundle.main.infoDictionary
        let markdown = BackendDiagnostics.markdown(
            runtime: store.runtime,
            inference: store.inference,
            training: store.training,
            appVersion: info?["CFBundleShortVersionString"] as? String ?? "未知",
            appBuild: info?["CFBundleVersion"] as? String ?? "未知",
            systemVersionFallback: ProcessInfo.processInfo.operatingSystemVersionString
        )
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(markdown, forType: .string)

        didCopy = true
        copyResetTask?.cancel()
        copyResetTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            didCopy = false
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

    var statusSymbol: String {
        switch self {
        case .ready: return "checkmark.circle.fill"
        case .starting, .running, .stopping: return "circle.dotted"
        case .failed: return "exclamationmark.triangle.fill"
        case .idle: return "circle.fill"
        }
    }
}
