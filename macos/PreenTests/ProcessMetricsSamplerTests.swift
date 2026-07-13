import XCTest
@testable import Preen

final class ProcessMetricsSamplerTests: XCTestCase {
    func testSamplerUsesDecimalGBAndCarriesTrainingTiming() throws {
        var sampledPID: Int32?
        let sampler = ProcessMetricsSampler(
            footprintProvider: { pid in
                sampledPID = pid
                return 12_713_115_648
            },
            swapProvider: { 1_500_000_000 }
        )

        let metric = try XCTUnwrap(sampler.sample(pid: 4321, step: 17, secondsPerStep: 1.25))
        XCTAssertEqual(sampledPID, 4321)
        XCTAssertEqual(metric.step, 17)
        XCTAssertEqual(metric.physicalFootprintGB, 12.713115648, accuracy: 1e-12)
        XCTAssertEqual(metric.swapUsedGB, 1.5, accuracy: 1e-12)
        XCTAssertEqual(metric.secondsPerStep, 1.25)
    }

    func testMissingProcessDoesNotProduceMetric() {
        let sampler = ProcessMetricsSampler(
            footprintProvider: { _ in nil },
            swapProvider: { XCTFail("不应在进程缺失时读取 swap"); return 0 }
        )

        XCTAssertNil(sampler.sample(pid: 9999, step: 0, secondsPerStep: nil))
    }
}
