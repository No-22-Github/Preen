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

    /// 同一步可能采样多次；全程图只保留该步的最高压力，避免按秒无限累积。
    func mergingPeak(with newer: ProcessMetric) -> ProcessMetric {
        ProcessMetric(
            timestamp: max(timestamp, newer.timestamp),
            step: step,
            physicalFootprintGB: max(physicalFootprintGB, newer.physicalFootprintGB),
            swapUsedGB: max(swapUsedGB, newer.swapUsedGB),
            pressure: MemoryMetricMath.moreSevere(pressure, newer.pressure),
            secondsPerStep: newer.secondsPerStep ?? secondsPerStep
        )
    }
}

struct SmoothedMemoryPoint: Identifiable, Equatable {
    var id: Int { step }
    let step: Int
    let physicalFootprintGB: Double
    let pressure: MemoryPressureLevel
}

enum MemoryMetricMath {
    static let emaSmoothing = 0.90
    static let warningRatio = 0.70
    static let criticalRatio = 0.85

    static func ema(
        _ metrics: [ProcessMetric],
        physicalMemoryGB: Double,
        smoothing: Double = emaSmoothing
    ) -> [SmoothedMemoryPoint] {
        let grouped = Dictionary(grouping: metrics, by: \.step)
        let perStep = grouped.keys
            .sorted()
            .compactMap { step -> ProcessMetric? in
                guard let samples = grouped[step],
                      let first = samples.first else { return nil }
                return samples.dropFirst().reduce(first) { $0.mergingPeak(with: $1) }
            }

        let amount = min(max(smoothing, 0), 0.99)
        var previous: Double?
        return perStep.map { metric in
            let value = previous.map {
                amount * $0 + (1 - amount) * metric.physicalFootprintGB
            } ?? metric.physicalFootprintGB
            previous = value
            return SmoothedMemoryPoint(
                step: metric.step,
                physicalFootprintGB: value,
                pressure: pressureLevel(
                    footprintGB: value,
                    physicalMemoryGB: physicalMemoryGB,
                    systemPressure: metric.pressure
                )
            )
        }
    }

    static func pressureLevel(
        footprintGB: Double,
        physicalMemoryGB: Double,
        systemPressure: MemoryPressureLevel
    ) -> MemoryPressureLevel {
        guard physicalMemoryGB > 0 else { return systemPressure }
        let ratio = max(0, footprintGB) / physicalMemoryGB
        let estimated: MemoryPressureLevel
        if ratio >= criticalRatio {
            estimated = .critical
        } else if ratio >= warningRatio {
            estimated = .warning
        } else {
            estimated = .normal
        }
        return moreSevere(estimated, systemPressure)
    }

    static func moreSevere(
        _ lhs: MemoryPressureLevel,
        _ rhs: MemoryPressureLevel
    ) -> MemoryPressureLevel {
        severity(lhs) >= severity(rhs) ? lhs : rhs
    }

    private static func severity(_ level: MemoryPressureLevel) -> Int {
        switch level {
        case .normal: return 0
        case .warning: return 1
        case .critical: return 2
        }
    }
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
