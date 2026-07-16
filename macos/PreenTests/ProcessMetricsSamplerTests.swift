import XCTest
@testable import Preen

final class ProcessMetricsSamplerTests: XCTestCase {
    func testSamplerUsesGiBForAppDisplayAndCarriesTrainingTiming() throws {
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
        XCTAssertEqual(
            metric.physicalFootprintGiB,
            Double(12_713_115_648) / Double(1024 * 1024 * 1024),
            accuracy: 1e-12
        )
        XCTAssertEqual(metric.swapUsedGiB, 1.3969838619232178, accuracy: 1e-12)
        XCTAssertEqual(metric.secondsPerStep, 1.25)
    }

    func testMissingProcessDoesNotProduceMetric() {
        let sampler = ProcessMetricsSampler(
            footprintProvider: { _ in nil },
            swapProvider: { XCTFail("不应在进程缺失时读取 swap"); return 0 }
        )

        XCTAssertNil(sampler.sample(pid: 9999, step: 0, secondsPerStep: nil))
    }

    func testMemoryEMAUsesPeakPerStepAndConfiguredSmoothing() throws {
        let metrics = [
            metric(step: 0, footprint: 4),
            metric(step: 0, footprint: 6),
            metric(step: 1, footprint: 10),
            metric(step: 2, footprint: 14),
        ]

        let points = MemoryMetricMath.ema(
            metrics, physicalMemoryGiB: 20, smoothing: 0.90
        )
        XCTAssertEqual(points.map(\.step), [0, 1, 2])
        XCTAssertEqual(points[0].physicalFootprintGiB, 6, accuracy: 1e-10)
        XCTAssertEqual(points[1].physicalFootprintGiB, 6.4, accuracy: 1e-10)
        XCTAssertEqual(points[2].physicalFootprintGiB, 7.16, accuracy: 1e-10)
    }

    func testMemoryPressureUsesMachineRatioAndSystemSignal() {
        XCTAssertEqual(
            MemoryMetricMath.pressureLevel(
                footprintGiB: 11, physicalMemoryGiB: 16, systemPressure: .normal
            ),
            .normal
        )
        XCTAssertEqual(
            MemoryMetricMath.pressureLevel(
                footprintGiB: 11.2, physicalMemoryGiB: 16, systemPressure: .normal
            ),
            .warning
        )
        XCTAssertEqual(
            MemoryMetricMath.pressureLevel(
                footprintGiB: 13.6, physicalMemoryGiB: 16, systemPressure: .normal
            ),
            .critical
        )
        XCTAssertEqual(
            MemoryMetricMath.pressureLevel(
                footprintGiB: 4, physicalMemoryGiB: 16, systemPressure: .critical
            ),
            .critical
        )
    }

    private func metric(
        step: Int,
        footprint: Double,
        pressure: MemoryPressureLevel = .normal
    ) -> ProcessMetric {
        ProcessMetric(
            timestamp: Date(timeIntervalSince1970: Double(step)),
            step: step,
            physicalFootprintGiB: footprint,
            swapUsedGiB: 0,
            pressure: pressure,
            secondsPerStep: nil
        )
    }
}
