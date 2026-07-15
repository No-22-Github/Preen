import AppKit

@MainActor
protocol DockProgressControlling: AnyObject {
    func update(progress: Double)
    func clear()
}

@MainActor
final class DockProgressController: DockProgressControlling {
    func update(progress: Double) {
        let tile = NSApp.dockTile
        tile.contentView = nil
        tile.badgeLabel = Self.badgeLabel(for: progress)
        tile.display()
    }

    func clear() {
        let tile = NSApp.dockTile
        tile.contentView = nil
        tile.badgeLabel = nil
        tile.display()
    }

    nonisolated static func badgeLabel(for progress: Double) -> String {
        let normalized = progress.isNaN ? 0 : min(max(progress, 0), 1)
        return "\(Int((normalized * 100).rounded()))%"
    }
}
