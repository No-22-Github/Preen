import SwiftUI
import Charts

struct TrainingChartView: View {
    static let minimumHeight: CGFloat = 500

    @Bindable var store: TrainStore
    @State private var smoothing = 0.6
    private let yAxisLabelWidth: CGFloat = 66

    // === 派生量缓存 ===
    // body 每次 store.lossPoints / processMetrics 变化都会被触发;
    // 原实现每次都全量重算 EMA 与内存梯度(O(N) × N 次刷新 = O(N²))。
    // 这里按指纹缓存:输入未变就复用上次结果。
    @State private var lossEMACache: [SmoothedLossPoint] = []
    @State private var lossEMAFingerprint: LossEMAFingerprint?

    @State private var memoryEMACache: [SmoothedMemoryPoint] = []
    @State private var memoryAreaGradientCache: LinearGradient?
    @State private var memoryLineGradientCache: LinearGradient?
    @State private var memoryFingerprint: MemoryDerivedFingerprint?

    private struct LossEMAFingerprint: Equatable {
        let count: Int
        let smoothing: Double
        // TrainingMetric 是值类型,数组身份由 count + 末尾 step 共同决定。
        // count 没变 = 没新增点;count 变了但末尾 step 相同 = 数据被替换,仍需重算。
        let lastStep: Int
    }

    private struct MemoryDerivedFingerprint: Equatable {
        let metricsCount: Int
        let totalSteps: Int
        let capacityGiB: Double
    }

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
            // 缓存复用:指纹命中时跳过 O(N) 重算,直接用上一次的结果。
            let lossPoints = cachedLossEMA
            let lossYDomain = self.lossYDomain(ema: lossPoints)
            let memoryPoints = cachedMemoryEMA
            let memoryAreaGradient = cachedMemoryAreaGradient
            let memoryLineGradient = cachedMemoryLineGradient

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
        // 派生量缓存:只在指纹变化时重算,不在 body 内部写 state。
        // onChange 是 body 之外触发的合法 state 更新点。
        .onChange(of: derivedLossFingerprint) { _, newFingerprint in
            // smoothing 变化也算指纹变化;空数组无需缓存。
            guard newFingerprint.count > 0 else {
                lossEMACache = []
                lossEMAFingerprint = nil
                return
            }
            lossEMACache = TrainingMetricMath.ema(store.lossPoints, smoothing: smoothing)
            lossEMAFingerprint = newFingerprint
        }
        .onChange(of: derivedMemoryFingerprint) { _, newFingerprint in
            guard newFingerprint.metricsCount > 0 else {
                memoryEMACache = []
                memoryAreaGradientCache = nil
                memoryLineGradientCache = nil
                memoryFingerprint = nil
                return
            }
            let computed = MemoryMetricMath.ema(
                store.processMetrics,
                physicalMemoryGiB: newFingerprint.capacityGiB
            )
            memoryEMACache = computed
            memoryAreaGradientCache = memoryPressureGradient(points: computed, opacity: 0.50)
            memoryLineGradientCache = memoryPressureGradient(points: computed, opacity: 1)
            memoryFingerprint = newFingerprint
        }
    }

    // MARK: - 派生量缓存

    /// Loss EMA:仅在 `lossPoints` 数量、末尾 step 或 `smoothing` 变化时重算。
    /// 训练每个 step 追加一个点 → 指纹变化 → 重算一次;后续因 hover/selection 触发的
    /// body 重绘(指纹不变)直接复用结果,从 O(N²) 降到 O(N)。
    ///
    /// 计算属性内部不写 @State(避免 body 期间修改 state 触发额外刷新)。
    /// 缓存刷新由 `.onChange(of: derivedLossFingerprint)` 在 body 外驱动。
    private var cachedLossEMA: [SmoothedLossPoint] {
        if lossEMACache.isEmpty, !store.lossPoints.isEmpty {
            return TrainingMetricMath.ema(store.lossPoints, smoothing: smoothing)
        }
        return lossEMACache
    }

    private var derivedLossFingerprint: LossEMAFingerprint {
        LossEMAFingerprint(
            count: store.lossPoints.count,
            smoothing: smoothing,
            lastStep: store.lossPoints.last?.step ?? -1
        )
    }

    /// yDomain 依赖 loss 原值、EMA 平滑值与 held-out 三组数据。
    /// 注意:hover 时 `IndependentChartSelection.selection` 变化只会重建子树,
    /// 不会触发本 body(TrainingChartView 的输入未变);所以这里只在训练事件时跑,
    /// 频率低,不额外做缓存,保持简单。
    private func lossYDomain(ema: [SmoothedLossPoint]) -> ClosedRange<Double> {
        TrainingMetricMath.lossYAxisDomain(
            values: store.lossPoints.map(\.loss)
                + ema.map(\.smoothedLoss)
                + store.heldOutPoints.map(\.loss)
        )
    }

    private var cachedMemoryEMA: [SmoothedMemoryPoint] {
        if memoryEMACache.isEmpty, !store.processMetrics.isEmpty {
            return MemoryMetricMath.ema(
                store.processMetrics,
                physicalMemoryGiB: memoryCapacityGiB
            )
        }
        return memoryEMACache
    }

    private var derivedMemoryFingerprint: MemoryDerivedFingerprint {
        MemoryDerivedFingerprint(
            metricsCount: store.processMetrics.count,
            totalSteps: store.totalSteps,
            capacityGiB: memoryCapacityGiB
        )
    }

    /// 内存梯度:onChange 已保证 cache 与 metrics 同步;空数组时退回单色 gradient。
    private var cachedMemoryAreaGradient: LinearGradient {
        memoryAreaGradientCache ?? memoryPressureGradient(points: [], opacity: 0.50)
    }

    private var cachedMemoryLineGradient: LinearGradient {
        memoryLineGradientCache ?? memoryPressureGradient(points: [], opacity: 1)
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

            lossChart(points: points, yDomain: yDomain)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .chartOverlay { proxy in
                    ChartHoverOverlay(
                        proxy: proxy,
                        selection: selection,
                        domain: displayedStepDomain,
                        yValue: selectedPoint?.smoothedLoss,
                        yDomain: yDomain,
                        label: selectedPoint.map {
                            String(format: "Raw %.4f · EMA %.4f", $0.rawLoss, $0.smoothedLoss)
                        },
                        labelColor: Color.accentColor,
                        indicatorColor: Color.accentColor
                    )
                }
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

            learningRateChart()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .chartOverlay { proxy in
                    ChartHoverOverlay(
                        proxy: proxy,
                        selection: selection,
                        domain: displayedStepDomain,
                        yValue: selectedPoint?.learningRate,
                        yDomain: learningRateDomain,
                        label: selectedPoint.map { formatLearningRate($0.learningRate) },
                        labelColor: .teal,
                        indicatorColor: .teal
                    )
                }
        }
    }

    private func memorySection(
        points: [SmoothedMemoryPoint],
        areaGradient: LinearGradient,
        lineGradient: LinearGradient,
        selection: Binding<Int?>
    ) -> some View {
        let selectedPoint = nearestMemoryPoint(to: selection.wrappedValue, in: points)

        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("进程内存")
                    .font(.headline)
                Text(L10n.format("上限 %.1f GB", memoryCapacityGiB))
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
                lineGradient: lineGradient
            )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .chartOverlay { proxy in
                    ChartHoverOverlay(
                        proxy: proxy,
                        selection: selection,
                        domain: displayedStepDomain,
                        yValue: selectedPoint?.physicalFootprintGiB,
                        yDomain: 0...memoryCapacityGiB,
                        label: selectedPoint.map {
                            String(format: "%.2f GB", $0.physicalFootprintGiB)
                        },
                        labelColor: selectedPoint?.pressure.chartColor ?? .accentColor,
                        indicatorColor: selectedPoint?.pressure.chartColor ?? .accentColor
                    )
                }
        }
    }

    /// Loss Chart。
    /// 关键性能点:Chart 的 builder 闭包**完全不引用 `selection` / `selectedPoint`**。
    /// 原实现把选中 RuleMark/PointMark/annotation 放在 Chart 内部,导致 hover 时
    /// IndependentChartSelection.selection 变化 → 子树重建 → Chart builder 重新求值 →
    /// ForEach(900+) 重新构造所有 LineMark,每帧数千次构造,明显掉帧。
    /// 现在选中可视化全部移到 `.chartOverlay`(ChartHoverOverlay),Chart 本体只依赖
    /// 数据点,SwiftUI 可稳定复用已构造的 mark。
    private func lossChart(
        points: [SmoothedLossPoint],
        yDomain: ClosedRange<Double>
    ) -> some View {
        Chart {
                ForEach(store.lossPoints) { point in
                    LineMark(
                        x: .value(L10n.string("步"), point.displayedStep),
                        y: .value(L10n.string("Raw loss"), point.loss),
                        series: .value(L10n.string("曲线"), L10n.string("Raw"))
                    )
                    .foregroundStyle(Color.accentColor.opacity(0.22))
                    .lineStyle(StrokeStyle(lineWidth: 0.8))
                    .interpolationMethod(.linear)
                }

                ForEach(points) { point in
                    LineMark(
                        x: .value(L10n.string("步"), point.step + 1),
                        y: .value(L10n.string("EMA loss"), point.smoothedLoss),
                        series: .value(L10n.string("曲线"), "EMA")
                    )
                    .foregroundStyle(Color.accentColor)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    .interpolationMethod(.linear)
                }

                ForEach(store.heldOutPoints) { point in
                    LineMark(
                        x: .value(L10n.string("步"), point.step + 1),
                        y: .value(L10n.string("Held-out loss"), point.loss),
                        series: .value(L10n.string("曲线"), L10n.string("Held-out"))
                    )
                    .foregroundStyle(.secondary)
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
                }

                ForEach(store.epochBoundaries) { boundary in
                    RuleMark(x: .value(L10n.string("Epoch"), boundary.step + 1))
                        .foregroundStyle(.quaternary)
                        .lineStyle(StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
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
            .accessibilityLabel("训练 loss 曲线")
    }

    private func learningRateChart() -> some View {
        Chart {
            ForEach(store.lossPoints) { point in
                LineMark(
                    x: .value(L10n.string("步"), point.displayedStep),
                    y: .value(L10n.string("Learning rate"), point.learningRate),
                    series: .value(L10n.string("曲线"), "LR")
                )
                .foregroundStyle(.teal)
                .lineStyle(StrokeStyle(lineWidth: 2))
                .interpolationMethod(.linear)
            }

            ForEach(store.epochBoundaries) { boundary in
                RuleMark(x: .value(L10n.string("Epoch"), boundary.step + 1))
                    .foregroundStyle(.quaternary)
                    .lineStyle(StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
            }

            if let warmupEndDisplayedStep {
                RuleMark(x: .value(L10n.string("Warmup 完成"), warmupEndDisplayedStep))
                    .foregroundStyle(.teal.opacity(0.55))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
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
        .accessibilityLabel("训练学习率曲线")
    }

    private func memoryChart(
        points: [SmoothedMemoryPoint],
        areaGradient: LinearGradient,
        lineGradient: LinearGradient
    ) -> some View {
        Chart {
            ForEach(points) { point in
                AreaMark(
                    x: .value(L10n.string("步"), point.step + 1),
                    yStart: .value(L10n.string("基线"), 0),
                    yEnd: .value(L10n.string("EMA 进程内存 GB"), point.physicalFootprintGiB),
                    series: .value(L10n.string("内存"), L10n.string("进程内存"))
                )
                .foregroundStyle(areaGradient)
                .interpolationMethod(.linear)
                .alignsMarkStylesWithPlotArea()

                LineMark(
                    x: .value(L10n.string("步"), point.step + 1),
                    y: .value(L10n.string("EMA 进程内存 GB"), point.physicalFootprintGiB),
                    series: .value(L10n.string("内存"), L10n.string("进程内存"))
                )
                .foregroundStyle(lineGradient)
                .lineStyle(StrokeStyle(lineWidth: 1.5))
                .interpolationMethod(.linear)
                .alignsMarkStylesWithPlotArea()
            }

            RuleMark(y: .value(L10n.string("严重阈值"), criticalMemoryGiB))
                .foregroundStyle(MemoryPressureLevel.critical.chartColor.opacity(0.70))
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
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

/// 静态 Chart 的选中可视化 overlay。
///
/// 用法:`Chart { ... }` 内部只画静态线条,不引用 selection;然后 `.chartOverlay { proxy in
/// ChartHoverOverlay(proxy:selection:domain:yValue:yDomain:...) }`。
///
/// 这样 hover 时只有 overlay 重建,Chart 本体保持稳定,避免 ForEach(N) 重新构造所有 mark。
///
/// 三个职责合一:
/// 1. 透明矩形接收 `onContinuousHover` → 更新 `selection`(step,已 clamp 到 domain)
/// 2. 用 `proxy.plotFrame` + `proxy.position(forX:in:)` 把选中 step 转屏幕 x 坐标
/// 3. 画竖线(RuleMark 等价)+ 选中点圆 + 顶部 annotation label
private struct ChartHoverOverlay: View {
    let proxy: ChartProxy
    @Binding var selection: Int?
    let domain: ClosedRange<Int>
    /// 选中步对应的 y 值(nil 时不画选中点,例如 LR 图的值类型不同时也可只画竖线)。
    let yValue: Double?
    let yDomain: ClosedRange<Double>
    let label: String?
    let labelColor: Color
    let indicatorColor: Color

    var body: some View {
        GeometryReader { geometry in
            if let plotFrameAnchor = proxy.plotFrame {
                let plotFrame = geometry[plotFrameAnchor]
                hoverField(plotFrame: plotFrame)
                if let selection,
                   let x = screenX(for: selection, in: plotFrame) {
                    selectionOverlay(
                        x: x, plotFrame: plotFrame, geometry: geometry
                    )
                }
            }
        }
    }

    /// 透明命中区 + hover → selection 转换。
    private func hoverField(plotFrame: CGRect) -> some View {
        Color.clear
            .contentShape(Rectangle())
            .onContinuousHover { phase in
                switch phase {
                case .active(let location):
                    guard plotFrame.contains(location),
                          let step: Int = proxy.value(
                            atX: location.x - plotFrame.origin.x
                          ) else {
                        if selection != nil { selection = nil }
                        return
                    }
                    let newSelection = min(max(step, domain.lowerBound), domain.upperBound)
                    if selection != newSelection {
                        selection = newSelection
                    }
                case .ended:
                    if selection != nil { selection = nil }
                }
            }
    }

    /// 选中步 → 屏幕 X(基于 plotFrame 与 proxy)。
    private func screenX(for step: Int, in plotFrame: CGRect) -> CGFloat? {
        guard let pos = proxy.position(forX: Double(step)) else { return nil }
        return plotFrame.origin.x + pos
    }

    /// 选中步对应 y 值 → 屏幕 Y。
    private func screenY(for value: Double, in plotFrame: CGRect) -> CGFloat? {
        guard let pos = proxy.position(forY: value) else { return nil }
        return plotFrame.origin.y + pos
    }

    /// 竖线 + 圆点 + annotation。三层叠在 plotFrame 上,都用绝对坐标(GeometryReader 本地系)。
    @ViewBuilder
    private func selectionOverlay(
        x: CGFloat, plotFrame: CGRect, geometry: GeometryProxy
    ) -> some View {
        // 竖线(贯穿 plot 区域)。
        Path { path in
            path.move(to: CGPoint(x: x, y: plotFrame.minY))
            path.addLine(to: CGPoint(x: x, y: plotFrame.maxY))
        }
        .stroke(indicatorColor.opacity(0.75), style: StrokeStyle(lineWidth: 1))
        .allowsHitTesting(false)

        // 选中圆点(若提供 y 值且能定位)。
        if let yValue,
           let y = screenY(for: yValue, in: plotFrame) {
            Circle()
                .fill(indicatorColor)
                .frame(width: 9, height: 9)
                .position(x: x, y: y)
                .allowsHitTesting(false)
        }

        // 顶部 annotation label:用 fixedSize 测量后 .position 居中到 (x, plotFrame.minY)。
        // plotFrame.minY 是 plot 顶部边界,label 中心放到这里,半高在 plot 上方、半高在内。
        // 若选中步靠近左右边缘,label 可能溢出 —— 可接受(原 Chart annotation 用 fit-to-chart)。
        if let label {
            ChartHoverValueLabel(text: label, color: labelColor)
                .fixedSize()
                .allowsHitTesting(false)
                .position(x: x, y: plotFrame.minY)
        }
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

            Text(L10n.string(label))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(L10n.format("%@ 曲线", L10n.string(label)))
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
