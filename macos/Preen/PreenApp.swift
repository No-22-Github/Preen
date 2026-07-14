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
    @Environment(\.openWindow) private var openWindow

    var body: some Scene {
        WindowGroup {
            ContentView(appState: appState)
                .frame(minWidth: 1000, minHeight: 680)  // 兜底最小尺寸(macOS 14)
                .task {
                    async let runtime: Void = appState.backendStore.checkRuntime()
                    async let runs: Void = appState.restoreRuns()
                    _ = await (runtime, runs)
                }
        }
        .defaultSize(width: 1180, height: 760)  // design.md §3 默认尺寸(macOS 15+)

        // 关于 Preen 独立窗口(由状态底栏 info 图标 / app 菜单触发)。
        WindowGroup("关于 Preen", id: "about") {
            AboutView()
        }
        .windowResizability(.contentSize)

        // app 菜单「关于」项也指向同一窗口。
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("关于 Preen") {
                    openWindow(id: "about")
                }
            }
            InspectorCommands()
        }
    }
}
