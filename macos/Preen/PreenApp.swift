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

    // 启动时是否显示欢迎窗口(欢迎窗口底部开关写同一个 key)。
    @AppStorage("welcomeShowsAtLaunch") private var welcomeShowsAtLaunch = true

    var body: some Scene {
        WindowGroup {
            ContentView(appState: appState)
                .frame(minWidth: 1000, minHeight: 680)  // 兜底最小尺寸(macOS 14)
                .task {
                    async let runtime: Void = appState.backendStore.checkRuntime()
                    async let runs: Void = appState.restoreRuns()
                    _ = await (runtime, runs)
                }
                .task {
                    // 首启(或用户保留「启动时显示」)自动弹欢迎窗口作为落脚点。
                    if welcomeShowsAtLaunch {
                        appState.isWelcomePresented = true
                    }
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
            // 「窗口」菜单增加重新打开欢迎窗口的入口(对齐 Xcode 的 Welcome to Xcode)。
            CommandGroup(after: .windowList) {
                Button("欢迎使用 Preen") {
                    appState.isWelcomePresented = true
                }
            }
            InspectorCommands()
        }
    }
}
