import XCTest
@testable import Preen

@MainActor
final class TrainingMetricTests: XCTestCase {
    func testEMAAndZeroSmoothing() {
        let metrics = [
            TrainingMetric(step: 0, loss: 10, learningRate: 0.01, epoch: 0, timestamp: 100),
            TrainingMetric(step: 1, loss: 6, learningRate: 0.01, epoch: 0, timestamp: 102),
            TrainingMetric(step: 2, loss: 2, learningRate: 0.01, epoch: 0, timestamp: 104),
        ]
        XCTAssertEqual(TrainingMetricMath.ema(metrics, smoothing: 0).map(\.smoothedLoss), [10, 6, 2])
        let smoothed = TrainingMetricMath.ema(metrics, smoothing: 0.5).map(\.smoothedLoss)
        XCTAssertEqual(smoothed[0], 10, accuracy: 1e-10)
        XCTAssertEqual(smoothed[1], 8, accuracy: 1e-10)
        XCTAssertEqual(smoothed[2], 5, accuracy: 1e-10)
    }

    func testProgressEtaAndEpochEndUseZeroBasedEventsCorrectly() throws {
        let store = TrainStore()
        store.consume(event: .start(config: try snapshot(samples: 5, epochs: 2), timestamp: 10))
        store.consume(event: .step(step: 0, totalSteps: 10, loss: 3, lr: 0.01, epoch: 0, timestamp: 100))
        XCTAssertEqual(store.displayedCurrentStep, 1)
        XCTAssertEqual(store.progress, 0.1, accuracy: 1e-10)
        XCTAssertNil(store.remainingSeconds)

        store.consume(event: .step(step: 1, totalSteps: 10, loss: 2.8, lr: 0.01, epoch: 0, timestamp: 102))
        XCTAssertEqual(store.remainingSeconds ?? -1, 16, accuracy: 1e-10)
        store.consume(event: .epochEnd(
            epoch: 0, loss: 2.9, stateStd: 0.12, lr: 0.01,
            heldOutLoss: 3.1, best: 3.1, patienceLeft: 3, timestamp: 110
        ))
        XCTAssertEqual(store.heldOutPoints.first?.step, 4)
        XCTAssertEqual(store.epochLossPoints.first?.step, 4)

        store.consume(event: .completed(path: "/tmp/state.npz", elapsed: 20, message: nil, timestamp: 120))
        XCTAssertEqual(store.progress, 1, accuracy: 1e-10)
        XCTAssertEqual(store.displayedCurrentStep, 10)
    }

    private func snapshot(samples: Int, epochs: Int) throws -> TrainConfigSnapshot {
        let json = """
        {"lr":0.01,"lr_floor":0.0001,"warmup":10,"ctx_len":512,"epochs":\(epochs),"grad_clip":1,"log_every":1,"early_stop":true,"early_stop_patience":3,"checkpoint_every":2,"seed":42,"n_samples":\(samples)}
        """
        return try JSONDecoder().decode(TrainConfigSnapshot.self, from: Data(json.utf8))
    }
}
