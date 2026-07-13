import SwiftUI

struct EventLogView: View {
    let events: [TrainEvent]
    @State private var filter = "全部"

    private var types: [String] {
        ["全部"] + Array(Set(events.map(\.typeName))).sorted()
    }

    private var filtered: [(Int, TrainEvent)] {
        Array(events.enumerated()).filter { filter == "全部" || $0.element.typeName == filter }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("事件").font(.headline)
                Spacer()
                Picker("类型", selection: $filter) {
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
        case .start(let config, _): return "\(config.nSamples) 条 · \(config.epochs) 轮 · ctx \(config.ctxLen)"
        case .resume(let epoch, let message, _): return "第 \(epoch + 1) 轮 · \(message)"
        case .epochStart(let epoch, _): return "第 \(epoch + 1) 轮开始"
        case .step(let step, let total, let loss, let lr, _, _):
            return String(format: "步 %d/%d · loss %.4f · lr %.6f", step + 1, total, loss, lr)
        case .epochEnd(let epoch, let loss, let std, _, let held, _, _, _):
            var text = String(format: "第 %d 轮 · train %.4f · std %.4f", epoch + 1, loss, std)
            if let held { text += String(format: " · held-out %.4f", held) }
            return text
        case .stdWarning(_, _, let message, _): return message
        case .checkpoint(let epoch, let path, _): return "第 \(epoch + 1) 轮 · \(path)"
        case .earlyStop(let epoch, let best, _, let message, _): return "第 \(epoch + 1) 轮 · best \(best) · \(message)"
        case .final(let path, let elapsed, _, _): return "\(path) · \(TrainStore.formatDuration(elapsed))"
        case .completed(let path, let elapsed, _, _): return "\(path) · \(TrainStore.formatDuration(elapsed))"
        case .failed(let message, _, _): return message
        case .cancelled(let message, _): return message
        case .unknown(_, _, _): return "未知事件"
        }
    }
}
