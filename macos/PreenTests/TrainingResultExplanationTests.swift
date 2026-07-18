import XCTest
@testable import Preen

final class TrainingResultExplanationTests: XCTestCase {
    @MainActor
    func testDataSummaryEventDecodesAndFeedsLiveStore() throws {
        let json = """
        {"type":"data_summary","timestamp":1,"total_records":200,"valid_samples":198,
         "train_samples":178,"held_out_samples":20,"truncated_samples":4,
         "dropped_samples":0,"target_fully_truncated":1}
        """
        let event = try JSONDecoder().decode(TrainEvent.self, from: Data(json.utf8))
        let store = TrainStore()
        store.replay(events: [event])
        XCTAssertEqual(store.trainSampleCount, 178)
        XCTAssertEqual(store.heldOutSampleCount, 20)
        XCTAssertEqual(store.truncatedSampleCount, 4)
        XCTAssertEqual(store.droppedSampleCount, 0)
        XCTAssertEqual(store.targetFullyTruncatedCount, 1)
    }

    @MainActor
    func testLiveCancellationBeforeEpochStartDoesNotInventStageOrStep() throws {
        let startJSON = """
        {"type":"start","timestamp":1,"config":{"lr":0.0001,"lr_floor":0.00001,
         "warmup":0,"ctx_len":512,"epochs":3,"grad_clip":1,"log_every":10,
         "early_stop":false,"early_stop_patience":3,"checkpoint_every":1,
         "seed":42,"n_samples":20}}
        """
        let decoder = JSONDecoder()
        let store = TrainStore()
        store.replay(events: [
            try decoder.decode(TrainEvent.self, from: Data(startJSON.utf8)),
            .cancelled(message: "cancel", timestamp: 2),
        ])
        let facts = TrainingResultExplanation(store: store)
        XCTAssertNil(facts.lastStartedEpoch)
        XCTAssertEqual(facts.completedSteps, 0)
        XCTAssertNil(facts.firstEpochLoss)
    }

    func testFirstCompletedEpochToFinalRelativeChange() {
        XCTAssertEqual(
            TrainingResultExplanation.relativeChangePercent(first: 4, final: 3) ?? .nan,
            -25,
            accuracy: 0.000_001
        )
        XCTAssertEqual(
            TrainingResultExplanation.relativeChangePercent(first: 2, final: 2.5) ?? .nan,
            25,
            accuracy: 0.000_001
        )
        XCTAssertNil(TrainingResultExplanation.relativeChangePercent(first: 0, final: 1))
    }

    func testCompletedSummaryUsesEpochAveragesAndBestHeldOutEpoch() {
        var run = TrainingRun(status: .completed, config: config)
        run.summary.elapsedSeconds = 12
        let events: [TrainEvent] = [
            .dataSummary(
                totalRecords: 20, validSamples: 20, trainSamples: 18, heldOutSamples: 2,
                truncatedSamples: 3, droppedSamples: 0, targetFullyTruncated: 1,
                timestamp: 1
            ),
            epoch(0, train: 4, held: 1.5, stateStd: 0.10),
            epoch(1, train: 3, held: 1.2, stateStd: 0.12),
            epoch(2, train: 2, held: 1.3, stateStd: 0.13),
            .completed(path: "/tmp/state.npz", elapsed: 12, message: nil, timestamp: 5),
        ]
        let facts = TrainingResultExplanation(run: run, events: events, metadata: nil)
        XCTAssertEqual(facts.termination, .completed)
        XCTAssertEqual(facts.actualEpochs, 3)
        XCTAssertEqual(facts.firstEpochLoss, 4)
        XCTAssertEqual(facts.finalEpochLoss, 2)
        XCTAssertEqual(facts.relativeTrainLossChangePercent ?? 0, -50, accuracy: 0.000_001)
        XCTAssertEqual(facts.firstHeldOutLoss, 1.5)
        XCTAssertEqual(facts.finalHeldOutLoss, 1.3)
        XCTAssertEqual(facts.bestHeldOutLoss, 1.2)
        XCTAssertEqual(facts.bestHeldOutEpoch, 2)
        XCTAssertEqual(facts.stateStd, 0.13)
        XCTAssertEqual(facts.trainSamples, 18)
        XCTAssertEqual(facts.heldOutSamples, 2)
        XCTAssertEqual(facts.truncatedSamples, 3)
        XCTAssertEqual(facts.droppedSamples, 0)
    }

    func testEarlyStopUsesConfiguredPatience() {
        var run = TrainingRun(status: .completed, config: config)
        run.summary.earlyStopped = true
        let facts = TrainingResultExplanation(
            run: run,
            events: [.earlyStop(epoch: 2, best: 1.2, heldOutLoss: 1.3, message: "stop", timestamp: 4)],
            metadata: nil
        )
        XCTAssertEqual(facts.termination, .earlyStopped(patience: 3))
    }

    func testCancelledBeforeFirstEpochDoesNotInventResults() {
        var run = TrainingRun(status: .cancelled, config: config)
        run.startedAt = Date(timeIntervalSince1970: 10)
        run.finishedAt = Date(timeIntervalSince1970: 15)
        let facts = TrainingResultExplanation(
            run: run,
            events: [
                .epochStart(epoch: 0, timestamp: 11),
                .step(step: 0, totalSteps: 100, loss: 5, lr: 0.001, epoch: 0, timestamp: 12),
                .cancelled(message: "cancel", timestamp: 15),
            ],
            metadata: nil
        )
        XCTAssertEqual(facts.termination, .cancelled)
        XCTAssertEqual(facts.actualEpochs, 0)
        XCTAssertNil(facts.firstEpochLoss)
        XCTAssertNil(facts.finalEpochLoss)
        XCTAssertNil(facts.stateStd)
        XCTAssertEqual(facts.elapsedSeconds, 5)
        XCTAssertEqual(facts.lastStartedEpoch, 1)
        XCTAssertEqual(facts.completedSteps, 1)
    }

    func testFailureAndMissingHeldOutRemainExplicitFacts() {
        var run = TrainingRun(status: .failed, config: config)
        run.failureMessage = "boom"
        let facts = TrainingResultExplanation(
            run: run,
            events: [epoch(0, train: 2, held: nil, stateStd: 0.1)],
            metadata: nil
        )
        XCTAssertEqual(facts.termination, .failed)
        XCTAssertNil(facts.bestHeldOutLoss)
        XCTAssertEqual(facts.failureMessage, "boom")
    }

    private func epoch(
        _ epoch: Int,
        train: Double,
        held: Double?,
        stateStd: Double
    ) -> TrainEvent {
        .epochEnd(
            epoch: epoch, loss: train, stateStd: stateStd, lr: 0.001,
            heldOutLoss: held, best: held, patienceLeft: 3,
            timestamp: Double(epoch + 2)
        )
    }

    private var config: PersistedTrainingConfig {
        PersistedTrainingConfig(
            modelPath: "/models/rwkv", dataPath: "/data/train.jsonl",
            outputPath: "/states/state.npz", template: "qa",
            learningRate: 0.0001, learningRateFloor: 0.00001,
            warmup: 50, contextLength: 512, epochs: 5,
            gradientClip: 1, earlyStop: true, earlyStopPatience: 3,
            testRatio: 0.1, seed: 42, cacheLimitGB: "auto"
        )
    }
}
