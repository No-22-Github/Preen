//
//  TrainingDoneView.swift
//  Preen
//
//  训练完成态。产物卡片 + 「去对话」按钮(design.md §4 最重要动线)。
//
//  注意 completed 事件没有最终 loss / data_sha256(子 Agent 契约已确认):
//   - 最终 loss:从 final.best 取(held-out 最佳)。
//   - 耗时:completed.elapsed 或 final.elapsed。
//   - data_sha256 / 轮数:从 configSnapshot.epochs 推;data_sha256 要读 metadata.json(本期暂不)。
//

import SwiftUI

struct TrainingDoneView: View {
    @Bindable var store: TrainStore
    /// 「去对话」回调:把 state 路径传给对话面板。
    var onGoToChat: (URL) -> Void

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 48))
                .foregroundStyle(.green)
                .accessibilityHidden(true)

            Text("训练完成")
                .font(.largeTitle)

            // 产物卡片。
            productCard
                .frame(maxWidth: 500)

            // 按钮组。
            HStack(spacing: 12) {
                if let path = store.outputPath {
                    Button {
                        onGoToChat(URL(fileURLWithPath: path))
                    } label: {
                        Label("去对话", systemImage: "bubble.left.and.bubble.right")
                            .frame(minWidth: 120)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                }
                if let path = store.outputPath {
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                    } label: {
                        Label("在 Finder 中显示", systemImage: "folder")
                    }
                }
            }

            // 「再训一个」。
            Button("再训一个") {
                store.reset()
            }
            .padding(.top, 8)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private var productCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            // state 路径。
            if let path = store.outputPath {
                LabeledRow(label: "state 文件") {
                    Text(URL(fileURLWithPath: path).lastPathComponent)
                        .font(.body)
                        .textSelection(.enabled)
                }
            }
            // 最终 loss(held-out 最佳)。
            if let best = store.finalBest {
                LabeledRow(label: "最终 held-out loss") {
                    Text(String(format: "%.4f", best))
                        .font(.body.monospacedDigit())
                }
            }
            // 轮数。
            if let cfg = store.configSnapshot {
                LabeledRow(label: "训练轮数") {
                    Text("\(cfg.epochs)")
                }
            }
            // 耗时。
            if let elapsed = store.elapsed {
                LabeledRow(label: "总耗时") {
                    Text(TrainStore.formatDuration(elapsed))
                }
            }
            // 早停信息(若触发)。
            if let early = store.earlyStopInfo {
                LabeledRow(label: "早停") {
                    Text("第 \(early.epoch + 1) 轮触发")
                        .foregroundStyle(.orange)
                }
            }
        }
        .padding(16)
        .background(.quaternary, in: .rect)
    }
}

/// 标签行:左侧副标签,右侧值。
struct LabeledRow<Content: View>: View {
    let label: String
    @ViewBuilder var content: Content

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            content
        }
    }
}
