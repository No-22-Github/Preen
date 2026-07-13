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
//  本期不画 RSS(留 #7 双轨共轴图),先把 loss 折线 + 摘要做出来。
//

import SwiftUI
import Charts

struct TrainingRunningView: View {
    @Bindable var store: TrainStore

    var body: some View {
        VStack(spacing: 0) {
            // 顶部摘要(3 秒判据:不点不滚能看到)。
            summaryHeader
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            Divider()

            // loss 折线。
            lossChart
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            // 底部取消。
            HStack {
                Spacer()
                Button(role: .destructive) {
                    store.cancel()
                } label: {
                    Label("取消训练", systemImage: "stop.circle")
                }
                .keyboardShortcut(.cancelAction)
            }
            .padding(16)
        }
    }

    // MARK: - 顶部摘要

    private var summaryHeader: some View {
        VStack(alignment: .leading, spacing: 6) {
            // 第一行:第几轮第几步 / 进度。
            HStack(spacing: 12) {
                Text("第 \(store.currentEpoch + 1) 轮")
                    .font(.headline)
                Text("步 \(store.currentStep) / \(store.totalSteps)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                if store.totalSteps > 0 {
                    Text("\(Int(store.progress * 100))%")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
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
            }
            .font(.subheadline)

            // 早停旗(若触发)。
            if let early = store.earlyStopInfo {
                Label("已停在第 \(early.epoch + 1) 轮(后续没有更好)",
                      systemImage: "flag.fill")
                    .foregroundStyle(.orange)
                    .font(.caption)
            }

            // 未知事件计数(演进兜底命中提示)。
            if store.unknownEventCount > 0 {
                Text("收到 \(store.unknownEventCount) 个未知事件(可能需升级 app)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - loss 折线

    private var lossChart: some View {
        Chart {
            // train loss 实线。
            ForEach(store.lossPoints) { p in
                LineMark(
                    x: .value("步", p.step),
                    y: .value("loss", p.loss)
                )
                .foregroundStyle(Color.accentColor)
                .interpolationMethod(.linear)

                // 鼠标悬停用点。
                PointMark(
                    x: .value("步", p.step),
                    y: .value("loss", p.loss)
                )
                .foregroundStyle(Color.accentColor)
                .symbolSize(4)
            }

            // held-out loss 虚线(若有)。
            ForEach(store.heldOutPoints) { p in
                LineMark(
                    x: .value("轮", epochToStep(p.epoch)),
                    y: .value("held-out", p.loss)
                )
                .foregroundStyle(.secondary)
                .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
                .interpolationMethod(.linear)
            }

            // epoch 分隔线(RuleMark)。
            ForEach(store.epochBoundaries) { b in
                RuleMark(
                    x: .value("epoch", b.step)
                )
                .foregroundStyle(.quaternary)
                .lineStyle(StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
            }
        }
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 6)) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let v = value.as(Int.self) {
                        Text("\(v)")
                    }
                }
            }
        }
        .chartYAxis {
            AxisMarks(values: .automatic(desiredCount: 5)) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let v = value.as(Double.self) {
                        Text(String(format: "%.2f", v))
                    }
                }
            }
        }
        .chartXScale(domain: 0...max(store.totalSteps, 1))
        // 禁用数据变化的隐式动画:Charts 在 lossPoints/epochBoundaries 追加时会
        // 用默认动画过渡,过渡途中折线 x 域重映射会产生「折返再归位」的视觉假象
        // (尤其 epoch 边界 RuleMark 加入触发整图重建时)。流式数据应即时落位,不插值。
        .animation(nil, value: store.lossPoints)
        .animation(nil, value: store.epochBoundaries)
        .animation(nil, value: store.heldOutPoints)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .accessibilityLabel("训练 loss 曲线")
    }

    /// 把 epoch 号映射到 step 轴(用该 epoch 的第一个 step 点近似)。
    private func epochToStep(_ epoch: Int) -> Int {
        // 简单映射:用 epoch 占总数的比例反推。
        guard store.totalSteps > 0, let cfg = store.configSnapshot else { return epoch }
        let perEpoch = cfg.nSamples
        return epoch * perEpoch
    }
}
