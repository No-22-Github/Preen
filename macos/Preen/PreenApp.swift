//
//  PreenApp.swift
//  Preen
//
//  App 入口(@main)。持有 AppState 单例,WindowGroup 挂 ContentView。
//

import SwiftUI

@main
struct PreenApp: App {
    @State private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView(appState: appState)
                .frame(minWidth: 1000, minHeight: 680)  // 兜底最小尺寸(macOS 14)
        }
        .defaultSize(width: 1180, height: 760)  // design.md §3 默认尺寸(macOS 15+)
    }
}
