import SwiftUI
import Charts

struct TrainingChartView: View {
    @Bindable var store: TrainStore
    @State private var smoothing = 0.6
    @State private var selectedStep: Int?

    private var smoothed: [SmoothedLossPoint] {
        TrainingMetricMath.ema(store.lossPoints, smoothing: smoothing)
    }

    private var selectedPoint: SmoothedLossPoint? {
        guard let selectedStep else { return nil }
        return smoothed.min { abs($0.step - selectedStep) < abs($1.step - selectedStep) }
    }

    private var selectedMetric: ProcessMetric? {
        guard let selectedStep else { return nil }
        return store.processMetrics.min { abs($0.step - selectedStep) < abs($1.step - selectedStep) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Loss").font(.headline)
                Spacer()
                Text("平滑 \(smoothing, format: .number.precision(.fractionLength(2)))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Slider(value: $smoothing, in: 0...0.95, step: 0.05)
                    .frame(width: 160)
            }

            lossChart
                .frame(minHeight: store.processMetrics.isEmpty ? 250 : 190)

            if !store.processMetrics.isEmpty {
                Divider()
                HStack {
                    Text("进程内存").font(.caption.bold())
                    Text("phys_footprint · 系统 swap · GB (÷1e9)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                memoryChart
                    .frame(height: 115)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var lossChart: some View {
        Chart {
                ForEach(store.lossPoints) { point in
                    LineMark(
                        x: .value("步", point.step),
                        y: .value("Raw loss", point.loss),
                        series: .value("曲线", "Raw")
                    )
                    .foregroundStyle(Color.accentColor.opacity(0.22))
                    .lineStyle(StrokeStyle(lineWidth: 0.8))
                    .interpolationMethod(.linear)
                }

                ForEach(smoothed) { point in
                    LineMark(
                        x: .value("步", point.step),
                        y: .value("EMA loss", point.smoothedLoss),
                        series: .value("曲线", "EMA")
                    )
                    .foregroundStyle(Color.accentColor)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    .interpolationMethod(.linear)
                }

                ForEach(store.heldOutPoints) { point in
                    LineMark(
                        x: .value("步", point.step),
                        y: .value("Held-out loss", point.loss),
                        series: .value("曲线", "Held-out")
                    )
                    .foregroundStyle(.secondary)
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
                }

                ForEach(store.epochBoundaries) { boundary in
                    RuleMark(x: .value("Epoch", boundary.step))
                        .foregroundStyle(.quaternary)
                        .lineStyle(StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
                }

                if let point = selectedPoint {
                    RuleMark(x: .value("选中步", point.step))
                        .foregroundStyle(.secondary)
                    RuleMark(y: .value("选中 loss", point.smoothedLoss))
                        .foregroundStyle(.secondary.opacity(0.55))
                    PointMark(
                        x: .value("选中步", point.step),
                        y: .value("选中 loss", point.smoothedLoss)
                    )
                    .foregroundStyle(Color.accentColor)
                    .symbolSize(45)
                    .annotation(position: .top, spacing: 8) {
                        tooltip(point, metric: selectedMetric)
                    }
                }
            }
            .chartXSelection(value: $selectedStep)
            .chartXScale(domain: 0...max(store.totalSteps - 1, 1))
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 6)) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let step = value.as(Int.self) { Text("\(step + 1)") }
                    }
                }
            }
            .chartYAxis {
                AxisMarks(values: .automatic(desiredCount: 5)) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let loss = value.as(Double.self) { Text(String(format: "%.2f", loss)) }
                    }
                }
            }
            .animation(nil, value: store.lossPoints)
            .animation(nil, value: store.epochBoundaries)
            .animation(nil, value: store.heldOutPoints)
            .accessibilityLabel("训练 loss 曲线")
    }

    private var memoryChart: some View {
        Chart {
            ForEach(store.processMetrics) { metric in
                LineMark(
                    x: .value("步", metric.step),
                    y: .value("RSS GB", metric.physicalFootprintGB),
                    series: .value("内存", "RSS")
                )
                .foregroundStyle(.purple)
                .lineStyle(StrokeStyle(lineWidth: 1.5))
                .interpolationMethod(.linear)

                LineMark(
                    x: .value("步", metric.step),
                    y: .value("Swap GB", metric.swapUsedGB),
                    series: .value("内存", "Swap")
                )
                .foregroundStyle(.orange.opacity(0.75))
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                .interpolationMethod(.linear)
            }

            if let selectedStep {
                RuleMark(x: .value("选中步", selectedStep))
                    .foregroundStyle(.secondary)
            }
        }
        .chartXSelection(value: $selectedStep)
        .chartXScale(domain: 0...max(store.totalSteps - 1, 1))
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 6)) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let step = value.as(Int.self) { Text("\(step + 1)") }
                }
            }
        }
        .chartYAxis {
            AxisMarks(values: .automatic(desiredCount: 4)) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let gb = value.as(Double.self) { Text(String(format: "%.1f G", gb)) }
                }
            }
        }
        .animation(nil, value: store.processMetrics)
        .accessibilityLabel("训练进程 RSS 曲线")
    }

    private func tooltip(_ point: SmoothedLossPoint, metric: ProcessMetric?) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("步 \(point.step + 1) · 第 \(point.epoch + 1) 轮").font(.caption.bold())
            Text(String(format: "raw %.4f · EMA %.4f", point.rawLoss, point.smoothedLoss))
            Text(String(format: "lr %.6f", point.learningRate))
            if let metric {
                Text(String(format: "RSS %.2f G · swap %.2f G",
                            metric.physicalFootprintGB, metric.swapUsedGB))
                if let seconds = metric.secondsPerStep {
                    Text(String(format: "%.2f s/步 · %@", seconds, pressureText(metric.pressure)))
                } else {
                    Text(pressureText(metric.pressure))
                }
            }
        }
        .font(.caption.monospacedDigit())
        .padding(7)
        .background(.regularMaterial, in: .rect)
    }

    private func pressureText(_ pressure: MemoryPressureLevel) -> String {
        switch pressure {
        case .normal: return "压力正常"
        case .warning: return "压力警告"
        case .critical: return "压力严重"
        }
    }
}
