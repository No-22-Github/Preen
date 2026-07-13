import AppKit

@MainActor
final class DockProgressController {
    func update(progress: Double) {
        let percent = Int(min(max(progress, 0), 1) * 100)
        NSApp.dockTile.badgeLabel = "\(percent)%"
    }

    func clear() {
        NSApp.dockTile.badgeLabel = nil
    }
}
