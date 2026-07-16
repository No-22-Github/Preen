import SwiftUI

extension MemoryPressureLevel {
    var displayLabel: String {
        switch self {
        case .normal: return L10n.string("压力正常")
        case .warning: return L10n.string("压力警告")
        case .critical: return L10n.string("压力严重")
        }
    }

    /// 跟随 macOS 系统色，分别对应活动监视器的绿、黄、红压力状态。
    var chartColor: Color {
        switch self {
        case .normal: return .green
        case .warning: return .yellow
        case .critical: return .red
        }
    }
}
