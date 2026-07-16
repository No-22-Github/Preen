import SwiftUI
import Charts

struct TrainingChartView: View {
    static let minimumHeight: CGFloat = 500

    @Bindable var store: TrainStore
    @State private var smoothing = 0.6
    private let yAxisLabelWidth: CGFloat = 66

    private var learningRateDomain: ClosedRange<Double> {
        let observedPeak = store.lossPoints.map(\.learningRate).max() ?? 0
        let configuredPeak = store.configSnapshot?.lr ?? 0
        let upperBound = max(max(observedPeak, configuredPeak), 1e-6) * 1.05
        return 0...upperBound
    }

    private var warmupEndDisplayedStep: Int? {
        guard let warmup = store.configSnapshot?.warmup,
              warmup > 0,
              store.totalSteps > 0 else { return nil }
        return min(warmup, store.totalSteps)
    }


    private var displayedStepDomain: ClosedRange<Int> {
        1...max(store.totalSteps, 2)
    }

    private var memoryCapacityGiB: Double {
        max(store.memoryCapacityGiB, 1)
    }

    private var criticalMemoryGiB: Double {
        memoryCapacityGiB * MemoryMetricMath.criticalRatio
    }

    var body: some View {
        GeometryReader { geometry in
            let usableHeight = max(geometry.size.height - 37, 1)
            let lossHeight = floor(usableHeight * 0.62)
            let lowerHeight = usableHeight - lossHeight
            let lossPoints = TrainingMetricMath.ema(store.lossPoints, smoothing: smoothing)
            let lossYDomain = TrainingMetricMath.lossYAxisDomain(
                values: store.lossPoints.map(\.loss)
                    + lossPoints.map(\.smoothedLoss)
                    + store.heldOutPoints.map(\.loss)
            )
            let memoryPoints = MemoryMetricMath.ema(
            store.processMetrics,
            physicalMemoryGiB: memoryCapacityGiB
            )
            let memoryAreaGradient = memoryPressureGradient(
                points: memoryPoints,
                opacity: 0.50
            )
            let memoryLineGradient = memoryPressureGradient(points: memoryPoints, opacity: 1)

            VStack(alignment: .leading, spacing: 8) {
                IndependentChartSelection { selection in
                    lossSection(
                        points: lossPoints,
                        yDomain: lossYDomain,
                        selection: selection
                    )
                }
                .frame(height: lossHeight)

                Divider()
                if !store.processMetrics.isEmpty {
                    HStack(alignment: .top, spacing: 16) {
                        IndependentChartSelection { selection in
                            learningRateSection(selection: selection)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                        Divider()

                        IndependentChartSelection { selection in
                            memorySection(
                                points: memoryPoints,
                                areaGradient: memoryAreaGradient,
                                lineGradient: memoryLineGradient,
                                selection: selection
                            )
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                    .frame(height: lowerHeight)
                } else {
                    IndependentChartSelection { selection in
                        learningRateSection(selection: selection)
                    }
                    .frame(height: lowerHeight)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .frame(minHeight: Self.minimumHeight)
    }

    private func lossSection(
        points: [SmoothedLossPoint],
        yDomain: ClosedRange<Double>,
        selection: Binding<Int?>
    ) -> some View {
        let selectedPoint = nearestLossPoint(to: selection.wrappedValue, in: points)

        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Loss").font(.headline)
                Spacer()
                Text("平滑 \(smoothing, format: .number.precision(.fractionLength(2)))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Slider(value: $smoothing, in: 0...0.95, step: 0.05)
                    .frame(width: 160)
            }

            HStack(alignment: .firstTextBaseline, spacing: 14) {
                if let point = selectedPoint {
                    Text("步 \(point.step + 1) · 第 \(point.epoch + 1) 轮")
                        .fontWeight(.semibold)
                }
                Spacer(minLength: 8)
                HStack(spacing: 14) {
                    ChartLineLegend(
                        label: "Raw", color: Color.accentColor.opacity(0.35), lineWidth: 0.8
                    )
                    ChartLineLegend(
                        label: "EMA", color: Color.accentColor, lineWidth: 2
                    )
                    if !store.heldOutPoints.isEmpty {
                        ChartLineLegend(
                            label: "Held-out", color: .secondary, lineWidth: 1.5, dash: [4, 3]
                        )
                    }
                }
            }
            .font(.caption.monospacedDigit())
            .lineLimit(1)
            .frame(minHeight: 18)

            lossChart(
                points: points,
                yDomain: yDomain,
                selectedPoint: selectedPoint,
                selection: selection
            )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func learningRateSection(selection: Binding<Int?>) -> some View {
        let selectedPoint = nearestLearningRatePoint(to: selection.wrappedValue)

        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Learning Rate")
                    .font(.headline)
                if let warmup = store.configSnapshot?.warmup, warmup > 0 {
                    Text("warmup \(warmup) 步")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            HStack(alignment: .firstTextBaseline, spacing: 10) {
                if let point = selectedPoint {
                    Text("步 \(point.displayedStep) · 第 \(point.epoch + 1) 轮")
                        .fontWeight(.semibold)
                }
                Spacer(minLength: 8)
                ChartLineLegend(label: "LR", color: .teal, lineWidth: 2)
            }
            .font(.caption.monospacedDigit())
            .lineLimit(1)
            .frame(minHeight: 18)

            learningRateChart(selectedPoint: selectedPoint, selection: selection)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func memorySection(
        points: [SmoothedMemoryPoint],
        areaGradient: LinearGradient,
        lineGradient: LinearGradient,
        selection: Binding<Int?>
    ) -> some View {
        let selectedPoint = nearestMemoryPoint(to: selection.wrappedValue, in: points)
        let selectedMetric = nearestMemoryMetric(to: selection.wrappedValue)

        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("进程内存")
                    .font(.headline)
                Text(String(format: "上限 %.1f GB", memoryCapacityGiB))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
            }

            HStack(alignment: .firstTextBaseline, spacing: 8) {
                if let point = selectedPoint {
                    Text("步 \(point.step + 1)")
                        .fontWeight(.semibold)
                    Text(point.pressure.displayLabel)
                        .foregroundStyle(point.pressure.chartColor)
                }
                Spacer(minLength: 4)
            }
            .font(.caption2.monospacedDigit())
            .lineLimit(1)
            .minimumScaleFactor(0.75)
            .frame(minHeight: 16)

            memoryChart(
                points: points,
                areaGradient: areaGradient,
                lineGradient: lineGradient,
                selectedPoint: selectedPoint,
                selectedMetric: selectedMetric,
                selection: selection
            )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func lossChart(
        points: [SmoothedLossPoint],
        yDomain: ClosedRange<Double>,
        selectedPoint: SmoothedLossPoint?,
        selection: Binding<Int?>
    ) -> some View {
        Chart {
                ForEach(store.lossPoints) { point in
                    LineMark(
                        x: .value("步", point.displayedStep),
                        y: .value("Raw loss", point.loss),
                        series: .value("曲线", "Raw")
                    )
                    .foregroundStyle(Color.accentColor.opacity(0.22))
                    .lineStyle(StrokeStyle(lineWidth: 0.8))
                    .interpolationMethod(.linear)
                }

                ForEach(points) { point in
                    LineMark(
                        x: .value("步", point.step + 1),
                        y: .value("EMA loss", point.smoothedLoss),
                        series: .value("曲线", "EMA")
                    )
                    .foregroundStyle(Color.accentColor)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    .interpolationMethod(.linear)
                }

                ForEach(store.heldOutPoints) { point in
                    LineMark(
                        x: .value("步", point.step + 1),
                        y: .value("Held-out loss", point.loss),
                        series: .value("曲线", "Held-out")
                    )
                    .foregroundStyle(.secondary)
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
                }

                ForEach(store.epochBoundaries) { boundary in
                    RuleMark(x: .value("Epoch", boundary.step + 1))
                        .foregroundStyle(.quaternary)
                        .lineStyle(StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
                }

                if let point = selectedPoint {
                    RuleMark(x: .value("选中步", point.step + 1))
                        .foregroundStyle(Color.accentColor.opacity(0.75))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                        .annotation(
                            position: .top,
                            alignment: .center,
                            spacing: 5,
                            overflowResolution: .init(x: .fit(to: .chart), y: .disabled)
                        ) {
                            ChartHoverValueLabel(
                                text: String(
                                    format: "Raw %.4f · EMA %.4f",
                                    point.rawLoss,
                                    point.smoothedLoss
                                ),
                                color: Color.accentColor
                            )
                        }
                    PointMark(
                        x: .value("选中步", point.step + 1),
                        y: .value("选中 EMA loss", point.smoothedLoss)
                    )
                    .foregroundStyle(Color.accentColor)
                    .symbolSize(45)
                }
            }
            .chartXScale(
                domain: displayedStepDomain,
                range: .plotDimension(startPadding: 8, endPadding: 8)
            )
            .chartYScale(domain: yDomain)
            .chartXAxis {
                AxisMarks(values: TrainingMetricMath.displayedStepAxisValues(
                    totalSteps: store.totalSteps,
                    desiredCount: 6
                )) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let step = value.as(Int.self) { Text("\(step)") }
                    }
                }
            }
            .chartYAxis {
                AxisMarks(values: .automatic(desiredCount: 5)) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let loss = value.as(Double.self) {
                            Text(String(format: "%8.2f", loss))
                                .monospacedDigit()
                                .frame(width: yAxisLabelWidth, alignment: .leading)
                        }
                    }
                }
            }
            .animation(nil, value: store.lossPoints)
            .animation(nil, value: store.epochBoundaries)
            .animation(nil, value: store.heldOutPoints)
            .chartHoverTracking(
                value: selection,
                domain: displayedStepDomain,
                selectableSteps: points.map { $0.step + 1 }
            )
            .accessibilityLabel("训练 loss 曲线")
    }

    private func learningRateChart(
        selectedPoint: TrainingMetric?,
        selection: Binding<Int?>
    ) -> some View {
        Chart {
            ForEach(store.lossPoints) { point in
                LineMark(
                    x: .value("步", point.displayedStep),
                    y: .value("Learning rate", point.learningRate),
                    series: .value("曲线", "LR")
                )
                .foregroundStyle(.teal)
                .lineStyle(StrokeStyle(lineWidth: 2))
                .interpolationMethod(.linear)
            }

            ForEach(store.epochBoundaries) { boundary in
                RuleMark(x: .value("Epoch", boundary.step + 1))
                    .foregroundStyle(.quaternary)
                    .lineStyle(StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
            }

            if let warmupEndDisplayedStep {
                RuleMark(x: .value("Warmup 完成", warmupEndDisplayedStep))
                    .foregroundStyle(.teal.opacity(0.55))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
            }

            if let point = selectedPoint {
                RuleMark(x: .value("选中步", point.displayedStep))
                    .foregroundStyle(Color.teal.opacity(0.75))
                    .lineStyle(StrokeStyle(lineWidth: 1))
                    .annotation(
                        position: .top,
                        alignment: .center,
                        spacing: 5,
                        overflowResolution: .init(x: .fit(to: .chart), y: .disabled)
                    ) {
                        ChartHoverValueLabel(
                            text: formatLearningRate(point.learningRate),
                            color: .teal
                        )
                    }
                PointMark(
                    x: .value("选中步", point.displayedStep),
                    y: .value("选中 LR", point.learningRate)
                )
                .foregroundStyle(.teal)
                .symbolSize(40)
            }
        }
        .chartXScale(
            domain: displayedStepDomain,
            range: .plotDimension(startPadding: 8, endPadding: 8)
        )
        .chartYScale(domain: learningRateDomain)
        .chartXAxis {
            AxisMarks(values: TrainingMetricMath.displayedStepAxisValues(
                totalSteps: store.totalSteps,
                desiredCount: store.processMetrics.isEmpty ? 6 : 4
            )) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let step = value.as(Int.self) { Text("\(step)") }
                }
            }
        }
        .chartYAxis {
            AxisMarks(values: .automatic(desiredCount: 4)) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let lr = value.as(Double.self) {
                        Text(formatLearningRate(lr))
                            .monospacedDigit()
                            .frame(width: yAxisLabelWidth, alignment: .leading)
                    }
                }
            }
        }
        .animation(nil, value: store.lossPoints)
        .animation(nil, value: store.epochBoundaries)
        .chartHoverTracking(
            value: selection,
            domain: displayedStepDomain,
            selectableSteps: store.lossPoints.map(\.displayedStep)
        )
        .accessibilityLabel("训练学习率曲线")
    }

    private func memoryChart(
        points: [SmoothedMemoryPoint],
        areaGradient: LinearGradient,
        lineGradient: LinearGradient,
        selectedPoint: SmoothedMemoryPoint?,
        selectedMetric: ProcessMetric?,
        selection: Binding<Int?>
    ) -> some View {
        Chart {
            ForEach(points) { point in
                AreaMark(
                    x: .value("步", point.step + 1),
                    yStart: .value("基线", 0),
                    yEnd: .value("EMA 进程内存 GB", point.physicalFootprintGiB),
                    series: .value("内存", "进程内存")
                )
                .foregroundStyle(areaGradient)
                .interpolationMethod(.linear)
                .alignsMarkStylesWithPlotArea()

                LineMark(
                    x: .value("步", point.step + 1),
                    y: .value("EMA 进程内存 GB", point.physicalFootprintGiB),
                    series: .value("内存", "进程内存")
                )
                .foregroundStyle(lineGradient)
                .lineStyle(StrokeStyle(lineWidth: 1.5))
                .interpolationMethod(.linear)
                .alignsMarkStylesWithPlotArea()
            }

            RuleMark(y: .value("严重阈值", criticalMemoryGiB))
                .foregroundStyle(MemoryPressureLevel.critical.chartColor.opacity(0.70))
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))

            if let point = selectedPoint, let metric = selectedMetric {
                RuleMark(x: .value("选中步", point.step + 1))
                    .foregroundStyle(point.pressure.chartColor.opacity(0.80))
                    .lineStyle(StrokeStyle(lineWidth: 1))
                    .annotation(
                        position: .top,
                        alignment: .center,
                        spacing: 5,
                        overflowResolution: .init(x: .fit(to: .chart), y: .disabled)
                    ) {
                        ChartHoverValueLabel(
                            text: String(format: "%.2f GB", metric.physicalFootprintGiB),
                            color: point.pressure.chartColor
                        )
                    }
                PointMark(
                    x: .value("选中步", point.step + 1),
                    y: .value("选中原始内存", metric.physicalFootprintGiB)
                )
                .foregroundStyle(point.pressure.chartColor)
                .symbolSize(40)
            }
        }
        .chartXScale(
            domain: displayedStepDomain,
            range: .plotDimension(startPadding: 8, endPadding: 8)
        )
        .chartYScale(domain: 0...memoryCapacityGiB)
        .chartXAxis {
            AxisMarks(values: TrainingMetricMath.displayedStepAxisValues(
                totalSteps: store.totalSteps,
                desiredCount: 4
            )) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let step = value.as(Int.self) { Text("\(step)") }
                }
            }
        }
        .chartYAxis {
            AxisMarks(values: [0, memoryCapacityGiB / 2, memoryCapacityGiB]) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let gb = value.as(Double.self) {
                        Text(String(format: "%6.1f GB", gb))
                            .monospacedDigit()
                            .frame(width: yAxisLabelWidth, alignment: .leading)
                    }
                }
            }
        }
        .animation(nil, value: store.processMetrics)
        .chartHoverTracking(
            value: selection,
            domain: displayedStepDomain,
            selectableSteps: points.map { $0.step + 1 }
        )
        .accessibilityLabel("训练进程内存压力面积图")
    }

    private func nearestLossPoint(
        to displayedStep: Int?,
        in points: [SmoothedLossPoint]
    ) -> SmoothedLossPoint? {
        nearestPoint(to: displayedStep, in: points) { $0.step + 1 }
    }

    private func nearestLearningRatePoint(to displayedStep: Int?) -> TrainingMetric? {
        nearestPoint(to: displayedStep, in: store.lossPoints, step: \.displayedStep)
    }

    private func nearestMemoryPoint(
        to displayedStep: Int?,
        in points: [SmoothedMemoryPoint]
    ) -> SmoothedMemoryPoint? {
        nearestPoint(to: displayedStep, in: points) { $0.step + 1 }
    }

    private func nearestMemoryMetric(to displayedStep: Int?) -> ProcessMetric? {
        guard let displayedStep else { return nil }
        return store.processMetrics.min {
            abs(($0.step + 1) - displayedStep) < abs(($1.step + 1) - displayedStep)
        }
    }

    /// 训练点按 step 递增，二分定位避免每个 hover 事件线性扫描整条曲线。
    private func nearestPoint<Point>(
        to target: Int?,
        in points: [Point],
        step: (Point) -> Int
    ) -> Point? {
        guard let target, !points.isEmpty else { return nil }
        var lower = 0
        var upper = points.count
        while lower < upper {
            let middle = (lower + upper) / 2
            if step(points[middle]) < target {
                lower = middle + 1
            } else {
                upper = middle
            }
        }
        if lower == 0 { return points[0] }
        if lower == points.count { return points[points.count - 1] }
        let before = points[lower - 1]
        let after = points[lower]
        return target - step(before) <= step(after) - target ? before : after
    }

    private func formatLearningRate(_ value: Double) -> String {
        return String(format: "%.6f", value)
    }

    private func memoryPressureGradient(
        points: [SmoothedMemoryPoint],
        opacity: Double
    ) -> LinearGradient {
        guard let first = points.first else {
            return LinearGradient(
                colors: [MemoryPressureLevel.normal.chartColor.opacity(opacity)],
                startPoint: .leading,
                endPoint: .trailing
            )
        }

        let denominator = Double(max(store.totalSteps - 1, 1))
        func location(for step: Int) -> CGFloat {
            CGFloat(min(max(Double(step) / denominator, 0), 1))
        }

        var stops = [Gradient.Stop(
            color: first.pressure.chartColor.opacity(opacity),
            location: 0
        )]
        var previous = first
        var previousLocation = location(for: first.step)

        for point in points.dropFirst() {
            let currentLocation = location(for: point.step)
            if point.pressure != previous.pressure {
                let transition = (previousLocation + currentLocation) / 2
                stops.append(Gradient.Stop(
                    color: previous.pressure.chartColor.opacity(opacity),
                    location: transition
                ))
                stops.append(Gradient.Stop(
                    color: point.pressure.chartColor.opacity(opacity),
                    location: transition
                ))
            }
            previous = point
            previousLocation = currentLocation
        }

        stops.append(Gradient.Stop(
            color: previous.pressure.chartColor.opacity(opacity),
            location: 1
        ))
        return LinearGradient(
            gradient: Gradient(stops: stops),
            startPoint: .leading,
            endPoint: .trailing
        )
    }
}

/// 每张图持有自己的 hover 状态，避免移动一张图时让整个统计面板重建。
private struct IndependentChartSelection<Content: View>: View {
    @State private var selection: Int?
    private let content: (Binding<Int?>) -> Content

    init(@ViewBuilder content: @escaping (Binding<Int?>) -> Content) {
        self.content = content
    }

    var body: some View {
        content($selection)
    }
}

private extension View {
    /// macOS 的图表读数跟随指针移动，并在离开当前图表时立即清除。
    func chartHoverTracking(
        value: Binding<Int?>,
        domain: ClosedRange<Int>,
        selectableSteps: [Int]
    ) -> some View {
        modifier(ChartHoverTrackingModifier(
            selection: value,
            domain: domain,
            selectableSteps: selectableSteps
        ))
    }
}

private struct ChartHoverTrackingModifier: ViewModifier {
    @Binding var selection: Int?
    let domain: ClosedRange<Int>
    let selectableSteps: [Int]

    func body(content: Content) -> some View {
        content.chartOverlay { proxy in
            GeometryReader { geometry in
                Color.clear
                    .contentShape(Rectangle())
                    .onContinuousHover { phase in
                        switch phase {
                        case .active(let location):
                            guard let plotFrameAnchor = proxy.plotFrame else {
                                selection = nil
                                return
                            }
                            let plotFrame = geometry[plotFrameAnchor]
                            guard plotFrame.contains(location),
                                  let step: Int = proxy.value(
                                    atX: location.x - plotFrame.origin.x
                                  ) else {
                                selection = nil
                                return
                            }
                            let rawSelection = min(
                                max(step, domain.lowerBound),
                                domain.upperBound
                            )
                            let newSelection = nearestSelectableStep(to: rawSelection)
                            if selection != newSelection {
                                selection = newSelection
                            }
                        case .ended:
                            if selection != nil { selection = nil }
                        }
                    }
            }
        }
    }

    private func nearestSelectableStep(to target: Int) -> Int {
        guard !selectableSteps.isEmpty else { return target }
        var lower = 0
        var upper = selectableSteps.count
        while lower < upper {
            let middle = (lower + upper) / 2
            if selectableSteps[middle] < target {
                lower = middle + 1
            } else {
                upper = middle
            }
        }
        if lower == 0 { return selectableSteps[0] }
        if lower == selectableSteps.count { return selectableSteps[selectableSteps.count - 1] }
        let before = selectableSteps[lower - 1]
        let after = selectableSteps[lower]
        return target - before <= after - target ? before : after
    }
}

private struct ChartLineLegend: View {
    let label: String
    let color: Color
    let lineWidth: CGFloat
    var dash: [CGFloat] = []

    var body: some View {
        HStack(spacing: 5) {
            Path { path in
                path.move(to: CGPoint(x: 0, y: 5))
                path.addLine(to: CGPoint(x: 26, y: 5))
            }
            .stroke(color, style: StrokeStyle(lineWidth: lineWidth, dash: dash))
            .frame(width: 26, height: 10)
            .accessibilityHidden(true)

            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(label) 曲线")
    }
}

private struct ChartHoverValueLabel: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.caption.weight(.semibold).monospacedDigit())
            .foregroundStyle(color)
            .padding(.horizontal, 4)
            .padding(.vertical, 2)
            .background(.background.opacity(0.88))
            .fixedSize()
    }
}
