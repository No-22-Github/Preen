//
//  TrainingDoneView.swift
//  Preen
//
//  训练完成态。轻量结果摘要 + 「去对话」按钮(design.md §4 最重要动线)。
//
//  注意 completed 事件没有最终 loss / data_sha256(子 Agent 契约已确认):
//   - 最终 loss:从 final.best 取(held-out 最佳)。
//   - 耗时:completed.elapsed 或 final.elapsed。
//   - data_sha256 / 轮数:从 configSnapshot.epochs 推;data_sha256 要读 metadata.json(本期暂不)。
//

import SwiftUI

struct TrainingDoneView: View {
    @Bindable var store: TrainStore
    /// 「去对话」回调:把 state 路径 + 训练用的模型路径传给对话面板(一键启动)。
    var onGoToChat: (URL, PersistedTrainingConfig?) -> Void
    /// 「返回首页」回调:reset store + 回训练面板空态落地页。
    var onGoHome: () -> Void
    var onShowChart: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 48))
                .foregroundStyle(.green)
                .accessibilityHidden(true)

            Text("训练完成")
                .font(.title.weight(.semibold))
                .padding(.top, 14)

            resultSummary
                .frame(maxWidth: 640)
                .padding(.top, 22)

            HStack(spacing: 10) {
                if let path = store.outputPath {
                    Button {
                        onGoToChat(URL(fileURLWithPath: path), store.currentRun?.config)
                    } label: {
                        Text("去对话")
                            .frame(minWidth: 120)
                    }
                    .buttonStyle(.borderedProminent)
                }

                Button {
                    onShowChart()
                } label: {
                    Text("查看曲线")
                        .frame(minWidth: 120)
                }
                .buttonStyle(.bordered)
                .disabled(store.lossPoints.isEmpty)
            }
            .padding(.top, 26)

            Button("返回首页") { onGoHome() }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .padding(.top, 14)
        }
        .padding(.horizontal, 32)
        .padding(.vertical, 32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private var resultSummary: some View {
        VStack(spacing: 14) {
            if let best = store.finalBest {
                SummaryRow(label: "held-out loss") {
                    HStack(spacing: 4) {
                        if let initial = initialHeldOutLoss,
                           store.heldOutPoints.count > 1 {
                            Text(formatLoss(initial))
                                .foregroundStyle(.tertiary)
                            Image(systemName: "arrow.right")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                        Text(formatLoss(best))
                            .fontWeight(.semibold)
                            .foregroundStyle(.primary)
                    }
                    .font(.body.monospacedDigit())
                }
            }

            if let dataPath = store.currentRun?.config?.dataPath {
                SummaryRow(label: "训练数据") {
                    Text(URL(fileURLWithPath: dataPath).lastPathComponent)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(dataPath)
                }
            }

            if let modelPath = store.currentRun?.config?.modelPath {
                SummaryRow(label: "基础模型") {
                    HStack(spacing: 8) {
                        Text(URL(fileURLWithPath: modelPath).lastPathComponent)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Text("·")
                            .foregroundStyle(.tertiary)
                        Text(ModelConfigProbe.precisionBadge(for: modelPath).uppercased())
                            .foregroundStyle(.secondary)
                    }
                    .help(modelPath)
                }
            }

            if let path = store.outputPath {
                SummaryRow(label: "State") {
                    HStack(spacing: 12) {
                        Text(URL(fileURLWithPath: path).lastPathComponent)
                            .fontWeight(.semibold)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                        } label: {
                            Image(systemName: "folder")
                                .foregroundStyle(.secondary)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .help("在 Finder 中显示")
                    }
                }
            }

            SummaryRow(label: "训练轮数") {
                HStack(spacing: 6) {
                    Text("\(actualEpochs) 轮")
                    if let elapsed = store.elapsed {
                        Text("·")
                        Text(TrainStore.formatDuration(elapsed))
                    }
                    if store.earlyStopInfo != nil {
                        Text("· 提前停止")
                            .foregroundStyle(.orange)
                    }
                }
                .foregroundStyle(.secondary)
            }
        }
    }

    private var initialHeldOutLoss: Double? {
        store.heldOutPoints.first?.loss
    }

    private var actualEpochs: Int {
        store.currentRun?.summary.actualEpochs
            ?? (store.epochLossPoints.last.map { $0.epoch + 1 })
            ?? store.configSnapshot?.epochs
            ?? 0
    }

    private func formatLoss(_ loss: Double) -> String {
        String(format: "%.2f", loss)
    }
}

/// 训练摘要行:左侧固定标签列,右侧结果列。
private struct SummaryRow<Content: View>: View {
    let label: String
    @ViewBuilder var content: Content

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 16) {
            Text(L10n.string(label))
                .font(.body)
                .foregroundStyle(.secondary)
                .frame(width: 140, alignment: .leading)
            content
                .font(.body)
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
    }
}
