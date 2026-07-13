//
//  AboutView.swift
//  Preen
//
//  关于 Preen 独立窗口:app icon + 名称 + 版本 + 描述 + GitHub / 参考 / 致谢。
//  由状态底栏右侧 info 图标 / app 菜单「关于 Preen」触发(openWindow)。
//

import SwiftUI

struct AboutView: View {
    /// 版本号从 Bundle 读取(构建时由 pbxproj MARKETING_VERSION 注入)。
    private var versionText: String {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return "版本 \(v) (\(build))"
    }

    var body: some View {
        VStack(spacing: 16) {
            Image(nsImage: NSApplication.shared.applicationIconImage)
                .resizable()
                .scaledToFit()
                .frame(width: 96, height: 96)

            Text("Preen")
                .font(.largeTitle.bold())

            Text(versionText)
                .font(.caption)
                .foregroundStyle(.secondary)

            Text("RWKV-7 State Tuning")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Divider()
                .frame(maxWidth: 280)

            // GitHub 卡片(可点开浏览器)。
            Button {
                if let url = URL(string: "https://github.com/No-22-Github/Preen") {
                    NSWorkspace.shared.open(url)
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "globe")
                        .foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("GitHub 仓库")
                        Text("源码、Issue、Roadmap")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Image(systemName: "arrow.up.right")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .padding(10)
                .frame(width: 280)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            .buttonStyle(.plain)

            Spacer(minLength: 0)
        }
        .padding(28)
        .frame(width: 360, height: 420)
    }
}
