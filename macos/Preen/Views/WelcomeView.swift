//
//  WelcomeView.swift
//  Preen
//
//  Xcode 式启动器窗口:首次启动自动弹出,之后可从「窗口 → 欢迎使用 Preen」再开。
//  不是教程闪屏——是没有上下文时的落脚点:说明用途 + 最近模型 + 三个明确入口。
//  对齐 Apple HIG:通过界面在情境中给出明确主操作,而非前置说明书。
//
//  三个入口指向真实的第一步:先有一个转换好的 RWKV-7 模型,再选数据训练。
//

import SwiftUI
import AppKit

struct WelcomeView: View {
    @Bindable var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @AppStorage("welcomeShowsAtLaunch") private var showsAtLaunch = true

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HStack(spacing: 0) {
                actionsColumn
                    .frame(width: 300)
                Divider()
                recentColumn
                    .frame(maxWidth: .infinity)
            }
            Divider()
            footer
        }
        .frame(width: 720, height: 460)
        // 告知主窗口收起侧栏(背景呈空状态);窗口关闭时恢复。
        .onAppear { appState.isWelcomePresented = true }
        .onDisappear { appState.isWelcomePresented = false }
    }

    // MARK: - 头部:用途一句话

    private var header: some View {
        HStack(spacing: 16) {
            Image(nsImage: NSApplication.shared.applicationIconImage)
                .resizable()
                .scaledToFit()
                .frame(width: 64, height: 64)
            VStack(alignment: .leading, spacing: 4) {
                Text("欢迎使用 Preen")
                    .font(.title.bold())
                Text("为 RWKV-7 训练一个行为 / 性格 state，再到对话里验证效果。")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(24)
    }

    // MARK: - 左列:三个有序入口

    private var actionsColumn: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("从这里开始")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)

            WelcomeAction(
                index: 1,
                icon: "wrench.and.screwdriver",
                title: "转换模型",
                subtitle: "把 BlinkDL / HF 权重转成 Preen 可用模型",
                prominent: appState.recentModels.isEmpty
            ) {
                appState.goToModelConversion()
                dismiss()
            }

            WelcomeAction(
                index: 2,
                icon: "shippingbox",
                title: "选择已有模型",
                subtitle: "已转换过?直接选一个 BF16 模型目录",
                prominent: false
            ) {
                pickModel()
            }

            WelcomeAction(
                index: 3,
                icon: "graduationcap",
                title: "选择训练数据",
                subtitle: "JSONL / JSON / CSV，自动探测格式",
                prominent: false
            ) {
                appState.selection = .training
                dismiss()
            }

            Spacer()
        }
        .padding(24)
    }

    private func pickModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "选择模型目录"
        if panel.runModal() == .OK, let url = panel.url {
            appState.selectModel(path: url.path)
            appState.selection = .training
            dismiss()
        }
    }

    // MARK: - 右列:最近模型

    private var recentColumn: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("最近模型")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)

            if appState.recentModels.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "shippingbox")
                        .font(.system(size: 32))
                        .foregroundStyle(.tertiary)
                    Text("还没有模型")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    Text("先「转换模型」得到一个 Preen 可用的 RWKV-7 模型。")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(spacing: 6) {
                        ForEach(appState.recentModels) { model in
                            WelcomeRecentRow(model: model) {
                                appState.selectModel(path: model.path)
                                appState.selection = .training
                                dismiss()
                            }
                        }
                    }
                }
            }
        }
        .padding(24)
    }

    // MARK: - 底部:启动时显示开关

    private var footer: some View {
        HStack {
            Toggle("启动时显示", isOn: $showsAtLaunch)
                .toggleStyle(.checkbox)
                .font(.callout)
            Spacer()
            Button("开始使用") { dismiss() }
                .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 14)
    }
}

// MARK: - 子组件

/// 左列入口卡片:序号 + 图标 + 标题 + 副标题,整卡可点。
private struct WelcomeAction: View {
    let index: Int
    let icon: String
    let title: String
    let subtitle: String
    let prominent: Bool
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.title3)
                    .frame(width: 28)
                    .foregroundStyle(prominent ? Color.accentColor : .secondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.body.weight(.medium))
                        .foregroundStyle(.primary)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(backgroundFill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(prominent ? Color.accentColor.opacity(0.5) : .clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
    }

    private var backgroundFill: Color {
        if prominent { return Color.accentColor.opacity(hovering ? 0.18 : 0.12) }
        return Color.primary.opacity(hovering ? 0.08 : 0.04)
    }
}

/// 右列最近模型行:模型名 + 精度标记,整行可点。
private struct WelcomeRecentRow: View {
    let model: RecentModel
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: "shippingbox")
                    .foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text(model.displayName)
                        .font(.callout)
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(model.path)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 6)
                precisionTag
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(Color.primary.opacity(hovering ? 0.08 : 0.03))
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
    }

    private var precisionTag: some View {
        let badge = ModelConfigProbe.precisionBadge(for: model.path)
        let isQuantized = badge == "int8"
        return Text(badge.uppercased())
            .font(.caption2.weight(.semibold))
            .foregroundStyle(isQuantized ? Color.orange : Color.secondary)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                (isQuantized ? Color.orange : Color.secondary).opacity(0.15),
                in: RoundedRectangle(cornerRadius: 4, style: .continuous)
            )
            .help(isQuantized ? "INT8 · 仅推理，不可训练" : "BF16 标准精度")
    }
}
