import XCTest
@testable import Preen

/// 锁定 TrainingConfig.projectedCounts 与 Python data.train_test_split +
/// train.py total_steps 的跨语言公式对齐（panel 预估必须 == 实际 total_steps）。
/// 历史 bug：旧预估用 valid * epochs，没扣 held-out，导致面板显示 400 步而实际只跑 360 步。
final class TrainingConfigProjectionTests: XCTestCase {

    /// 用户真实场景：200 条全有效、早停开、test_ratio 0.1、2 epoch。
    /// train_test_split: n_test = max(1, int(200*0.1)) = 20 → train 180。
    /// total_steps = 180 * 2 = 360（旧 buggy 预估会给 400）。
    func testHeldOutDeductedMatchesActualTotalSteps() {
        let r = TrainingConfig.projectedCounts(
            effectiveValid: 200, truncated: 0, dropTruncated: false,
            earlyStop: true, testRatio: 0.1, epochs: 2
        )
        XCTAssertEqual(r.train, 180)
        XCTAssertEqual(r.heldOut, 20)
        XCTAssertEqual(r.steps, 360)
    }

    /// 早停关闭 = 全量训练，不划分验证集。
    func testEarlyStopOffNoSplit() {
        let r = TrainingConfig.projectedCounts(
            effectiveValid: 200, truncated: 0, dropTruncated: false,
            earlyStop: false, testRatio: 0.1, epochs: 3
        )
        XCTAssertEqual(r.train, 200)
        XCTAssertEqual(r.heldOut, 0)
        XCTAssertEqual(r.steps, 600)
    }

    /// dropTruncated：扣掉截断条后再按公式划分。
    /// 200 有效 - 30 截断 = 170；held_out = max(1, int(170*0.1)) = 17；train 153；steps 306。
    func testDropTruncatedDeductsBeforeSplit() {
        let r = TrainingConfig.projectedCounts(
            effectiveValid: 200, truncated: 30, dropTruncated: true,
            earlyStop: true, testRatio: 0.1, epochs: 2
        )
        XCTAssertEqual(r.train, 153)
        XCTAssertEqual(r.heldOut, 17)
        XCTAssertEqual(r.steps, 306)
    }

    /// 大数据集：10k 条、3 epoch。对齐用户后续 batch 实验场景。
    func testLargeDatasetScaling() {
        let r = TrainingConfig.projectedCounts(
            effectiveValid: 10_000, truncated: 0, dropTruncated: false,
            earlyStop: true, testRatio: 0.1, epochs: 3
        )
        XCTAssertEqual(r.heldOut, 1000)
        XCTAssertEqual(r.train, 9000)
        XCTAssertEqual(r.steps, 27_000)
    }

    /// 边界：有效样本极少时 held_out 至少 1 条（对齐 max(1, ...) 下限）。
    func testTinyDatasetHeldOutFloorIsOne() {
        let r = TrainingConfig.projectedCounts(
            effectiveValid: 5, truncated: 0, dropTruncated: false,
            earlyStop: true, testRatio: 0.1, epochs: 3
        )
        XCTAssertEqual(r.heldOut, 1)  // int(5*0.1)=0 → max(1,0)=1
        XCTAssertEqual(r.train, 4)
        XCTAssertEqual(r.steps, 12)
    }
}
