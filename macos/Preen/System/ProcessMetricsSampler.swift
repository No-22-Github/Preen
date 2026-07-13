import Foundation
import Darwin

enum MemoryPressureLevel: String, Codable, Equatable {
    case normal
    case warning
    case critical
}

struct ProcessMetric: Identifiable, Equatable {
    let id = UUID()
    let timestamp: Date
    let step: Int
    let physicalFootprintGB: Double
    let swapUsedGB: Double
    let pressure: MemoryPressureLevel
    let secondsPerStep: Double?
}

final class MemoryPressureMonitor: @unchecked Sendable {
    private let lock = NSLock()
    private var level: MemoryPressureLevel = .normal
    private let source: DispatchSourceMemoryPressure

    init() {
        source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.normal, .warning, .critical], queue: .global(qos: .utility)
        )
        source.setEventHandler { [weak self, weak source] in
            guard let self, let event = source?.data else { return }
            let newLevel: MemoryPressureLevel
            if event.contains(.critical) { newLevel = .critical }
            else if event.contains(.warning) { newLevel = .warning }
            else { newLevel = .normal }
            self.lock.withLock { self.level = newLevel }
        }
        source.resume()
    }

    var current: MemoryPressureLevel { lock.withLock { level } }
}

final class ProcessMetricsSampler: @unchecked Sendable {
    typealias FootprintProvider = (Int32) -> UInt64?
    typealias SwapProvider = () -> UInt64

    private let footprintProvider: FootprintProvider
    private let swapProvider: SwapProvider
    private let pressureMonitor: MemoryPressureMonitor

    init(
        footprintProvider: @escaping FootprintProvider = ProcessMetricsSampler.physicalFootprint,
        swapProvider: @escaping SwapProvider = ProcessMetricsSampler.swapUsed,
        pressureMonitor: MemoryPressureMonitor = MemoryPressureMonitor()
    ) {
        self.footprintProvider = footprintProvider
        self.swapProvider = swapProvider
        self.pressureMonitor = pressureMonitor
    }

    func sample(pid: Int32, step: Int, secondsPerStep: Double?) -> ProcessMetric? {
        guard let footprint = footprintProvider(pid) else { return nil }
        return ProcessMetric(
            timestamp: Date(),
            step: step,
            physicalFootprintGB: Double(footprint) / 1e9,
            swapUsedGB: Double(swapProvider()) / 1e9,
            pressure: pressureMonitor.current,
            secondsPerStep: secondsPerStep
        )
    }

    private static func physicalFootprint(pid: Int32) -> UInt64? {
        var info = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { rebound in
                proc_pid_rusage(pid, RUSAGE_INFO_V4, rebound)
            }
        }
        return result == 0 ? info.ri_phys_footprint : nil
    }

    private static func swapUsed() -> UInt64 {
        var usage = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        let result = withUnsafeMutablePointer(to: &usage) { pointer in
            sysctlbyname("vm.swapusage", pointer, &size, nil, 0)
        }
        return result == 0 ? usage.xsu_used : 0
    }
}
