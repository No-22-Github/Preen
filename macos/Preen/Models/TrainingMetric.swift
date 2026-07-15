import Foundation

struct TrainingMetric: Identifiable, Equatable {
    var id: Int { step }
    let step: Int
    let loss: Double
    let learningRate: Double
    let epoch: Int
    let timestamp: Double

    var displayedStep: Int { step + 1 }
}

struct SmoothedLossPoint: Identifiable, Equatable {
    var id: Int { step }
    let step: Int
    let rawLoss: Double
    let smoothedLoss: Double
    let learningRate: Double
    let epoch: Int
}

struct EpochLossPoint: Identifiable, Equatable {
    var id: Int { epoch }
    let epoch: Int
    let step: Int
    let trainLoss: Double
    let heldOutLoss: Double?
    let stateStd: Double
}

enum TrainingMetricMath {
    static func ema(_ metrics: [TrainingMetric], smoothing: Double) -> [SmoothedLossPoint] {
        let amount = min(max(smoothing, 0), 0.95)
        var previous: Double?
        return metrics.map { metric in
            let value = previous.map { amount * $0 + (1 - amount) * metric.loss } ?? metric.loss
            previous = value
            return SmoothedLossPoint(
                step: metric.step,
                rawLoss: metric.loss,
                smoothedLoss: value,
                learningRate: metric.learningRate,
                epoch: metric.epoch
            )
        }
    }

    /// 训练事件使用零基 step，图表则展示一基步数。刻度从 1 起步，中间落在
    /// 10/20/50/100… 这样的整齐数值，并始终以任务总步数收尾。
    static func displayedStepAxisValues(totalSteps: Int, desiredCount: Int) -> [Int] {
        let upperBound = max(totalSteps, 1)
        guard upperBound > 1 else { return [1] }

        let count = max(desiredCount, 2)
        let roughStride = Double(upperBound) / Double(count - 1)
        let magnitude = pow(10, floor(log10(roughStride)))
        let normalized = roughStride / magnitude
        let niceNormalized: Double
        switch normalized {
        case ..<1.5: niceNormalized = 1
        case ..<3.5: niceNormalized = 2
        case ..<7.5: niceNormalized = 5
        default: niceNormalized = 10
        }
        let stride = max(Int((niceNormalized * magnitude).rounded()), 1)

        var values = [1]
        var value = stride
        while value <= upperBound {
            if value != 1 { values.append(value) }
            value += stride
        }
        if values.last != upperBound { values.append(upperBound) }
        return values
    }

    /// 根据当前实际绘制的 loss 自动收紧纵轴，并留出足够空间避免折线贴边。
    /// 极差很小时额外按数值量级留白，防止近似常数曲线被放大成剧烈波动。
    static func lossYAxisDomain(values: [Double]) -> ClosedRange<Double> {
        let finiteValues = values.filter(\.isFinite)
        guard let minimum = finiteValues.min(),
              let maximum = finiteValues.max() else {
            return 0...1
        }

        let span = maximum - minimum
        let magnitude = max(abs(minimum), abs(maximum), 1e-6)
        let padding: Double
        if span > 0 {
            padding = max(span * 0.10, magnitude * 0.01)
        } else {
            padding = max(magnitude * 0.10, 1e-3)
        }

        let lowerBound = minimum >= 0 ? max(0, minimum - padding) : minimum - padding
        return lowerBound...(maximum + padding)
    }
}
