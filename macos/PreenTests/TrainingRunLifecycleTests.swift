import XCTest
@testable import Preen

final class TrainingRunLifecycleTests: XCTestCase {
    func testCompletedLifecycle() {
        var run = TrainingRun(createdAt: Date(timeIntervalSince1970: 1))
        run.apply(event: .start(config: snapshot, timestamp: 2))
        XCTAssertEqual(run.status, .running)
        run.apply(event: .epochEnd(
            epoch: 0, loss: 2.5, stateStd: 0.12, lr: 0.01,
            heldOutLoss: 2.7, best: 2.7, patienceLeft: 3, timestamp: 3
        ))
        run.apply(event: .final(path: "/tmp/state.npz", elapsed: 10, best: 2.7, timestamp: 4))
        XCTAssertEqual(run.status, .finishing)
        run.apply(event: .completed(path: "/tmp/state.npz", elapsed: 11, message: nil, timestamp: 5))
        XCTAssertEqual(run.status, .completed)
        XCTAssertEqual(run.summary.finalLoss, 2.5)
        XCTAssertEqual(run.summary.actualEpochs, 1)
        XCTAssertEqual(run.artifacts.statePath, "/tmp/state.npz")
    }

    func testFailedLifecycle() {
        var run = TrainingRun()
        run.apply(event: .failed(message: "bad data", path: nil, timestamp: 2))
        XCTAssertEqual(run.status, .failed)
        XCTAssertEqual(run.failureMessage, "bad data")
    }

    func testCancelledLifecycle() {
        var run = TrainingRun()
        run.apply(event: .cancelled(message: "用户取消", timestamp: 2))
        XCTAssertEqual(run.status, .cancelled)
        XCTAssertNotNil(run.finishedAt)
    }

    func testUnfinishedRunBecomesInterruptedOnRestore() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = RunRepository(rootURL: root)
        let run = TrainingRun(status: .running)
        _ = try await repository.create(run)
        _ = try await repository.markUnfinishedRunsInterrupted(at: Date(timeIntervalSince1970: 9))
        let restored = try await repository.load(id: run.id)
        XCTAssertEqual(restored.status, .interrupted)
        XCTAssertEqual(restored.failureMessage, L10n.string("App 上次退出时训练尚未结束"))
    }

    private var snapshot: TrainConfigSnapshot {
        let data = Data("""
        {"lr":0.01,"lr_floor":0.0001,"warmup":10,"ctx_len":512,"epochs":3,"grad_clip":1,"log_every":1,"early_stop":true,"early_stop_patience":3,"checkpoint_every":2,"seed":42,"n_samples":10}
        """.utf8)
        return try! JSONDecoder().decode(TrainConfigSnapshot.self, from: data)
    }
}
