//
//  TrainingRunningView.swift
//  Preen
//
//  训练运行中视图。Swift Charts loss 折线 + epoch 分隔线(RuleMark)。
//
//  design.md §5「3 秒判据」:训练跑满 10 分钟后从后台切回窗口,
//  不点击不滚动,3 秒内能读到:
//   - 机器是否在换页 / 当前第几轮第几步 / loss 是否在降 / 预计剩余时间
//
//  loss / RSS 双轨共用 step 轴，机器压力同时固定在摘要区。
//

import SwiftUI

struct TrainingRunningView: View {
    @Bindable var store: TrainStore
    @State private var showingCancelConfirm = false

    var body: some View {
        VStack(spacing: 0) {
            // 顶部摘要(3 秒判据:不点不滚能看到)。
            summaryHeader
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            Divider()

            // loss 折线。
            TrainingChartView(store: store)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            // 底部取消。
            HStack {
                Spacer()
                Button(role: .destructive) {
                    showingCancelConfirm = true
                } label: {
                    Label("取消训练", systemImage: "stop.circle")
                }
                .keyboardShortcut(.cancelAction)
            }
            .padding(16)
            .confirmationDialog(
                "确认取消训练？",
                isPresented: $showingCancelConfirm,
                titleVisibility: .visible
            ) {
                Button("取消训练", role: .destructive) {
                    store.cancel()
                }
                Button("继续训练", role: .cancel) {}
            } message: {
                if store.totalSteps > 0 {
                    Text("训练进行到第 \(store.currentEpoch + 1) 轮 \(store.displayedCurrentStep)/\(store.totalSteps) 步。取消后训练进程会终止，已计算的 loss 曲线会保留，但未完成的进度无法恢复。")
                } else {
                    Text("取消后训练进程会终止，已计算的 loss 曲线会保留，但未完成的进度无法恢复。")
                }
            }
        }
    }

    // MARK: - 顶部摘要

    private var summaryHeader: some View {
        VStack(alignment: .leading, spacing: 6) {
            // 第一行:第几轮第几步 / 进度。
            HStack(spacing: 12) {
                Text("第 \(store.currentEpoch + 1) 轮")
                    .font(.headline)
                Text("步 \(store.displayedCurrentStep) / \(store.totalSteps)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                if store.totalSteps > 0 {
                    Text("\(Int(store.progress * 100))%")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }

            if store.totalSteps > 0 {
                ProgressView(value: store.progress)
                    .progressViewStyle(.linear)
                    .accessibilityLabel("训练进度")
                    .accessibilityValue(
                        "\(store.displayedCurrentStep) / \(store.totalSteps)，\(Int(store.progress * 100))%"
                    )
            }

            // 第二行:loss / lr / 预计剩余。
            HStack(spacing: 16) {
                Label("loss \(store.lossDisplay)", systemImage: "chart.line.downtrend")
                    .foregroundStyle(.primary)
                Label("lr \(store.lrDisplay)", systemImage: "speedometer")
                    .foregroundStyle(.secondary)
                if let remain = store.remainingSeconds {
                    Label("剩余 \(TrainStore.formatDuration(remain))",
                          systemImage: "clock")
                        .foregroundStyle(.secondary)
                }
                if let metric = store.latestProcessMetric {
                    Label(String(format: "RSS %.2f G", metric.physicalFootprintGB),
                          systemImage: "memorychip")
                        .foregroundStyle(.secondary)
                    Text(pressureLabel(metric.pressure))
                        .foregroundStyle(metric.pressure == .normal ? Color.secondary : Color.orange)
                }
            }
            .font(.subheadline)

            // 早停旗(若触发)。
            if let early = store.earlyStopInfo {
                Label("已停在第 \(early.epoch + 1) 轮（后续没有更好）",
                      systemImage: "flag.fill")
                    .foregroundStyle(.orange)
                    .font(.caption)
            }

            // 未知事件计数(演进兜底命中提示)。
            if store.unknownEventCount > 0 {
                Text("收到 \(store.unknownEventCount) 个未知事件（可能需升级 app）")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func pressureLabel(_ pressure: MemoryPressureLevel) -> String {
        switch pressure {
        case .normal: return "压力正常"
        case .warning: return "压力警告"
        case .critical: return "压力严重"
        }
    }

}
