import SwiftUI

struct EventLogView: View {
    let events: [TrainEvent]
    @State private var filter = "all"

    private var types: [String] { Array(Set(events.map(\.typeName))).sorted() }

    private var filtered: [(Int, TrainEvent)] {
        Array(events.enumerated()).filter { filter == "all" || $0.element.typeName == filter }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("事件").font(.headline)
                Spacer()
                Picker("类型", selection: $filter) {
                    Text("全部").tag("all")
                    ForEach(types, id: \.self) { Text($0).tag($0) }
                }
                .frame(width: 180)
            }
            if events.isEmpty {
                Text("没有可读取的事件")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 90)
            } else {
                List(filtered, id: \.0) { _, event in
                    HStack(alignment: .firstTextBaseline) {
                        Text(event.typeName)
                            .font(.caption.monospaced())
                            .frame(width: 92, alignment: .leading)
                        Text(event.summaryText)
                            .font(.caption)
                            .textSelection(.enabled)
                        Spacer()
                        Text(Date(timeIntervalSince1970: event.timestamp), style: .time)
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.tertiary)
                    }
                }
                .listStyle(.inset)
                .frame(minHeight: 180)
            }
        }
    }
}

extension TrainEvent {
    var typeName: String {
        switch self {
        case .start: return "start"
        case .dataSummary: return "data_summary"
        case .resume: return "resume"
        case .epochStart: return "epoch_start"
        case .step: return "step"
        case .epochEnd: return "epoch_end"
        case .stdWarning: return "std_warning"
        case .checkpoint: return "checkpoint"
        case .earlyStop: return "early_stop"
        case .final: return "final"
        case .completed: return "completed"
        case .failed: return "failed"
        case .cancelled: return "cancelled"
        case .unknown(let type, _, _): return type
        }
    }

    var summaryText: String {
        switch self {
        case .start(let config, _): return L10n.format("%lld 条 · %lld 轮 · ctx %lld", config.nSamples, config.epochs, config.ctxLen)
        case .dataSummary(_, _, let train, let heldOut, let truncated, let dropped, _, _):
            return L10n.format(
                "训练 %lld · 验证 %lld · 截断 %lld · 丢弃 %lld",
                train, heldOut, truncated, dropped
            )
        case .resume(let epoch, let message, _):
            return L10n.format("第 %lld 轮 · %@", epoch + 1, L10n.backendMessage(message, fallback: "正在恢复训练"))
        case .epochStart(let epoch, _): return L10n.format("第 %lld 轮开始", epoch + 1)
        case .step(let step, let total, let loss, let lr, _, _):
            return L10n.format("步 %d/%d · loss %.4f · lr %.6f", step + 1, total, loss, lr)
        case .epochEnd(let epoch, let loss, let std, _, let held, _, _, _):
            var text = L10n.format("第 %d 轮 · train %.4f · std %.4f", epoch + 1, loss, std)
            if let held { text += String(format: " · held-out %.4f", held) }
            return text
        case .stdWarning(_, _, let message, _):
            return L10n.backendMessage(message, fallback: "State 数值出现异常，请检查训练参数")
        case .checkpoint(let epoch, let path, _): return L10n.format("第 %lld 轮 · %@", epoch + 1, path)
        case .earlyStop(let epoch, let best, _, let message, _):
            return L10n.format("第 %lld 轮 · best %g · %@", epoch + 1, best, L10n.backendMessage(message, fallback: "已触发提前停止"))
        case .final(let path, let elapsed, _, _): return "\(path) · \(TrainStore.formatDuration(elapsed))"
        case .completed(let path, let elapsed, _, _): return "\(path) · \(TrainStore.formatDuration(elapsed))"
        case .failed(let message, _, _): return L10n.backendMessage(message, fallback: "训练失败，请查看训练日志")
        case .cancelled(let message, _): return L10n.backendMessage(message, fallback: "训练已取消")
        case .unknown(_, _, _): return L10n.string("未知事件")
        }
    }
}
