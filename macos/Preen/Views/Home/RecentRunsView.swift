import SwiftUI

struct RecentRunsView: View {
    let runs: [TrainingRun]
    let onSelect: (TrainingRun) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("最近训练")
                .font(.headline)
            if runs.isEmpty {
                ContentUnavailableView(
                    "还没有训练记录",
                    systemImage: "clock.arrow.circlepath",
                    description: Text("从“开始训练”选择数据，第一条记录会在进程启动前保存。")
                )
                .frame(maxWidth: .infinity, minHeight: 170)
            } else {
                List(runs.prefix(6)) { run in
                    Button { onSelect(run) } label: {
                        HStack(spacing: 10) {
                            Image(systemName: run.status.systemImage)
                                .foregroundStyle(run.status.color)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(run.config.map { URL(fileURLWithPath: $0.dataPath).lastPathComponent } ?? "外部 State")
                                    .lineLimit(1)
                                Text(run.createdAt, format: .dateTime.month().day().hour().minute())
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text(run.status.label)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            if let loss = run.summary.finalLoss {
                                Text(String(format: "%.4f", loss))
                                    .font(.caption.monospacedDigit())
                            }
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
                .listStyle(.inset)
                .frame(minHeight: 190)
            }
        }
    }
}

extension TrainingRunStatus {
    var label: String {
        switch self {
        case .preparing: return "准备中"
        case .running: return "训练中"
        case .finishing: return "收尾中"
        case .completed: return "已完成"
        case .failed: return "失败"
        case .cancelled: return "已取消"
        case .interrupted: return "已中断"
        }
    }

    var systemImage: String {
        switch self {
        case .preparing, .running, .finishing: return "progress.indicator"
        case .completed: return "checkmark.circle.fill"
        case .failed: return "xmark.octagon.fill"
        case .cancelled: return "stop.circle.fill"
        case .interrupted: return "bolt.slash.fill"
        }
    }

    var color: Color {
        switch self {
        case .preparing, .running, .finishing: return .accentColor
        case .completed: return .green
        case .failed: return .red
        case .cancelled, .interrupted: return .orange
        }
    }
}
