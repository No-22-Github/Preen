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

        // 诊断日志窗口(Mo9):作为独立窗口承载,让用户能在训练/推理过程中持续 tail,
        // 而不是被 sheet 阻挡父窗口。HIG: repeated input-and-observe workflows 应该
        // 用 panel/window,而不是 sheet。
        Window("诊断日志", id: "backend-logs") {
            BackendLogSheet(store: appState.backendStore)
        }
        .defaultSize(width: 720, height: 480)

        // 设置窗口(⌘,)。macOS 用户期望 App 菜单里有 Settings 项。
        // 当前只暴露「启动时显示欢迎窗口」一项,后续可扩展。
        Settings {
            SettingsView(welcomeShowsAtLaunch: $welcomeShowsAtLaunch)
        }

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

            // 前往菜单(M4):面板切换 ⌘1-4。Mac 用户期望键盘可达。
            CommandMenu("前往") {
                ForEach(Array(SidebarItem.allCases.enumerated()), id: \.element.id) { index, item in
                    Button(item.label) {
                        appState.selection = item
                    }
                    .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: .command)
                }
            }

            // 训练菜单:开始/停止训练、最近训练。
            CommandMenu("训练") {
                Button("开始训练…") {
                    appState.selection = .training
                }
                .keyboardShortcut("n", modifiers: [.command, .shift])
                .disabled(appState.trainStore.hasActiveProcess)

                Button("停止训练") {
                    appState.trainStore.cancel()
                }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(!appState.trainStore.hasActiveProcess)

                Divider()

                Button("训练记录…") {
                    appState.selection = .history
                }
            }

            // 对话菜单:连接 / 断开 / A-B / 加载 State / 清除 State / 会话参数。
            CommandMenu("对话") {
                Button("连接本地模型") {
                    appState.connectInference()
                }
                .disabled(appState.chatStore.hasActiveProcess || appState.modelPath.isEmpty)

                Button("断开") {
                    appState.disconnectInference()
                }
                .disabled(!appState.chatStore.isConnected)

                Divider()

                Button("A/B 对比") {
                    appState.chatStore.isComparisonMode.toggle()
                }
                .disabled(appState.chatStore.statePath == nil)

                Button("加载 State…") {
                    appState.requestLoadStateFromFilePicker()
                }
                .disabled(appState.modelPath.isEmpty)

                Button("卸下 State") {
                    appState.requestSessionReplacement(.clearState)
                }
                .disabled(appState.chatStore.statePath == nil)

                Divider()

                Button("停止生成") {
                    appState.chatStore.abort()
                }
                .keyboardShortcut(.escape, modifiers: [])
                .disabled(!appState.chatStore.isGenerating)
            }

            // 模型菜单:选择/管理最近模型,转换入口。
            CommandMenu("模型") {
                if appState.recentModels.isEmpty {
                    Text("还没有模型记录")
                } else {
                    ForEach(appState.recentModels) { model in
                        Button(model.displayName) {
                            appState.selectModel(path: model.path)
                        }
                    }
                }
                Divider()
                Button("转换模型…") {
                    appState.goToModelConversion()
                }
            }

            // 工具菜单:诊断日志入口(等同于后端状态页的"诊断日志…")。
            CommandMenu("工具") {
                Button("诊断日志…") {
                    openWindow(id: "backend-logs")
                }
            }
        }
    }
}

/// 设置窗口内容。当前只有"启动时显示欢迎窗口"一项。
private struct SettingsView: View {
    @Binding var welcomeShowsAtLaunch: Bool

    var body: some View {
        Form {
            Section("启动") {
                Toggle("启动时显示欢迎窗口", isOn: $welcomeShowsAtLaunch)
            }
        }
        .formStyle(.grouped)
        .padding()
        .frame(width: 380)
    }
}
