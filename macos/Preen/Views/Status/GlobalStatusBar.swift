import SwiftUI

struct GlobalStatusBar: View {
    @Bindable var appState: AppState
    @Environment(\.openWindow) private var openWindow
    @State private var showingBackendStatus = false
    @State private var isHoveringStatus = false

    private var backend: BackendStore { appState.backendStore }
    private var train: TrainStore { appState.trainStore }

    var body: some View {
        HStack(spacing: 12) {
            // 绿点 + 健康文字 + 运行时消息(可点击,打开后端状态)。
            statusCluster

            if backend.training.phase == .running {
                Divider().frame(height: 14)
                Label("\(Int(train.progress * 100))%", systemImage: "chart.line.uptrend.xyaxis")
                Text("loss \(train.lossDisplay)")
                if let metric = backend.processMetrics.last {
                    Text(String(format: "RSS %.2f G", metric.physicalFootprintGB))
                    Text(String(format: "swap %.2f G", metric.swapUsedGB))
                    Text(pressureLabel(metric.pressure))
                        .foregroundStyle(pressureColor(metric.pressure))
                    if let seconds = metric.secondsPerStep {
                        Text(String(format: "%.2f s/步", seconds))
                    }
                } else {
                    Text("正在读取内存…").foregroundStyle(.secondary)
                }
            } else if backend.inference.phase == .ready || backend.inference.phase == .running {
                Divider().frame(height: 14)
                Label(backend.inference.message, systemImage: "bubble.left.and.bubble.right")
            }

            Spacer(minLength: 8)
            if let remaining = train.remainingSeconds,
               backend.training.phase == .running {
                Text("剩余 \(TrainStore.formatDuration(remaining))")
                    .foregroundStyle(.secondary)
            }

            // 关于入口(最右侧)。
            Button {
                openWindow(id: "about")
            } label: {
                Image(systemName: "info.circle")
            }
            .buttonStyle(.borderless)
            .help("关于 Preen")
            .labelStyle(.iconOnly)
        }
        .font(.caption.monospacedDigit())
        .lineLimit(1)
        .padding(.horizontal, 12)
        .frame(height: 30)
        .background(.bar)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("全局后端状态：\(healthLabel)")
        .sheet(isPresented: $showingBackendStatus) {
            BackendStatusView(store: appState.backendStore)
        }
    }

    /// 绿点 + 健康标签 + 运行时消息,整体可点击(hover 淡背景)。
    private var statusCluster: some View {
        Button {
            showingBackendStatus = true
        } label: {
            HStack(spacing: 12) {
                Circle()
                    .fill(healthColor)
                    .frame(width: 7, height: 7)
                Text(healthLabel)
                    .fontWeight(.medium)
                Divider().frame(height: 14)
                Label(backend.runtime.message, systemImage: "terminal")
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .fill(isHoveringStatus ? Color.primary.opacity(0.06) : .clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHoveringStatus = $0 }
        .help("查看 Python、MLX 与后端日志")
    }

    private var healthColor: Color {
        if backend.runtime.phase == .unavailable ||
            backend.training.phase == .failed || backend.inference.phase == .failed {
            return .red
        }
        if backend.runtime.phase == .checking || backend.training.phase == .starting ||
            backend.inference.phase == .starting {
            return .orange
        }
        return .green
    }

    private var healthLabel: String {
        if backend.training.phase == .failed { return "训练异常" }
        if backend.inference.phase == .failed { return "推理异常" }
        switch backend.runtime.phase {
        case .checking: return "正在检查"
        case .unavailable: return "运行时异常"
        case .ready: return backend.training.phase == .running ? "训练中" : "后端正常"
        }
    }

    private func pressureLabel(_ pressure: MemoryPressureLevel) -> String {
        switch pressure {
        case .normal: return "压力正常"
        case .warning: return "压力警告"
        case .critical: return "压力严重"
        }
    }

    private func pressureColor(_ pressure: MemoryPressureLevel) -> Color {
        switch pressure {
        case .normal: return .secondary
        case .warning: return .orange
        case .critical: return .red
        }
    }
}
