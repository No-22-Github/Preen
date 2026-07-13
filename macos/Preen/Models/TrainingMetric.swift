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
}
