import XCTest
import UserNotifications
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

    func testDisplayedStepAxisUsesRoundOneBasedLabels() {
        XCTAssertEqual(
            TrainingMetricMath.displayedStepAxisValues(totalSteps: 540, desiredCount: 6),
            [1, 100, 200, 300, 400, 500, 540]
        )
        XCTAssertEqual(
            TrainingMetricMath.displayedStepAxisValues(totalSteps: 540, desiredCount: 4),
            [1, 200, 400, 540]
        )
        XCTAssertEqual(
            TrainingMetricMath.displayedStepAxisValues(totalSteps: 400, desiredCount: 6),
            [1, 100, 200, 300, 400]
        )
        XCTAssertEqual(
            TrainingMetricMath.displayedStepAxisValues(totalSteps: 350, desiredCount: 6),
            [1, 50, 100, 150, 200, 250, 300, 350]
        )
        XCTAssertEqual(
            TrainingMetricMath.displayedStepAxisValues(totalSteps: 10, desiredCount: 6),
            [1, 2, 4, 6, 8, 10]
        )
        XCTAssertEqual(
            TrainingMetricMath.displayedStepAxisValues(totalSteps: 1, desiredCount: 6),
            [1]
        )
    }

    func testLossYAxisDomainFitsCurrentValuesWithPadding() {
        let domain = TrainingMetricMath.lossYAxisDomain(values: [2, 3, 4])
        XCTAssertEqual(domain.lowerBound, 1.8, accuracy: 1e-10)
        XCTAssertEqual(domain.upperBound, 4.2, accuracy: 1e-10)
    }

    func testLossYAxisDomainHandlesFlatAndInvalidData() {
        let flatDomain = TrainingMetricMath.lossYAxisDomain(values: [2, 2])
        XCTAssertEqual(flatDomain.lowerBound, 1.8, accuracy: 1e-10)
        XCTAssertEqual(flatDomain.upperBound, 2.2, accuracy: 1e-10)

        let zeroDomain = TrainingMetricMath.lossYAxisDomain(values: [0])
        XCTAssertEqual(zeroDomain.lowerBound, 0, accuracy: 1e-10)
        XCTAssertGreaterThan(zeroDomain.upperBound, 0)

        let emptyDomain = TrainingMetricMath.lossYAxisDomain(values: [.nan, .infinity])
        XCTAssertEqual(emptyDomain, 0...1)
    }

    func testTrainingStepForwardsCalculatedProgressToDock() throws {
        let dock = DockProgressSpy()
        let store = TrainStore(
            repository: RunRepository(),
            backendStore: BackendStore(),
            dockProgress: dock
        )
        store.consume(event: .start(config: try snapshot(samples: 5, epochs: 2), timestamp: 10))
        store.consume(event: .step(
            step: 2,
            totalSteps: 10,
            loss: 2.5,
            lr: 0.001,
            epoch: 0,
            timestamp: 100
        ))

        XCTAssertEqual(dock.updates, [0.3])
        XCTAssertEqual(dock.clearCount, 0)

        store.reset()
        XCTAssertEqual(dock.clearCount, 1)
    }

    func testDockBadgeLabelUsesClampedRoundedPercentage() {
        XCTAssertEqual(DockProgressController.badgeLabel(for: -0.2), "0%")
        XCTAssertEqual(DockProgressController.badgeLabel(for: 0.404), "40%")
        XCTAssertEqual(DockProgressController.badgeLabel(for: 0.406), "41%")
        XCTAssertEqual(DockProgressController.badgeLabel(for: 1.2), "100%")
        XCTAssertEqual(DockProgressController.badgeLabel(for: .nan), "0%")
    }

    func testTrainingNotificationAuthorizationRegistersDockBadges() {
        XCTAssertTrue(TrainingNotificationController.authorizationOptions.contains(.alert))
        XCTAssertTrue(TrainingNotificationController.authorizationOptions.contains(.sound))
        XCTAssertTrue(TrainingNotificationController.authorizationOptions.contains(.badge))
    }

    func testTrainingDefaultsMatchBackendLearningRates() {
        let config = TrainingConfig()
        XCTAssertEqual(config.lr, 0.0001)
        XCTAssertEqual(config.lrFloor, 0.00001)

        let arguments = config.commandLineArguments()
        XCTAssertEqual(arguments[arguments.firstIndex(of: "--lr")! + 1], "0.0001")
        XCTAssertEqual(arguments[arguments.firstIndex(of: "--lr-floor")! + 1], "1e-05")
    }

    func testProgressEtaAndEpochEndUseZeroBasedEventsCorrectly() throws {
        let store = TrainStore()
        store.consume(event: .start(config: try snapshot(samples: 5, epochs: 2), timestamp: 10))
        store.consume(event: .step(step: 0, totalSteps: 10, loss: 3, lr: 0.001, epoch: 0, timestamp: 100))
        XCTAssertEqual(store.displayedCurrentStep, 1)
        XCTAssertEqual(store.progress, 0.1, accuracy: 1e-10)
        XCTAssertEqual(store.currentLr, 0.001, accuracy: 1e-10)
        XCTAssertEqual(store.lossPoints.map(\.learningRate), [0.001])
        XCTAssertNil(store.remainingSeconds)

        store.consume(event: .step(step: 1, totalSteps: 10, loss: 2.8, lr: 0.002, epoch: 0, timestamp: 102))
        XCTAssertEqual(store.currentLr, 0.002, accuracy: 1e-10)
        XCTAssertEqual(store.lossPoints.map(\.learningRate), [0.001, 0.002])
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

@MainActor
private final class DockProgressSpy: DockProgressControlling {
    private(set) var updates: [Double] = []
    private(set) var clearCount = 0

    func update(progress: Double) {
        updates.append(progress)
    }

    func clear() {
        clearCount += 1
    }
}
