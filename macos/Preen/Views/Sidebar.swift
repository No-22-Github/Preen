//
//  Sidebar.swift
//  Preen
//
//  侧边栏。主导航列表(训练/对话/训练记录/工具箱)。
//   - 模型选择器已移至主窗口 toolbar primaryAction(靠近窗口右边缘)。
//   - 后端状态入口已移至 GlobalStatusBar(绿点区可点击)。
//

import SwiftUI

struct Sidebar: View {
    @Bindable var appState: AppState

    var body: some View {
        navList
            .frame(minWidth: 200)
    }

    private var navList: some View {
        List(selection: $appState.selection) {
            ForEach(SidebarItem.allCases) { item in
                NavigationLink(value: item) {
                    Label(item.label, systemImage: item.systemImage)
                }
            }
        }
        .listStyle(.sidebar)
    }
}
