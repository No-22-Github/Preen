//
//  SessionReplacementConfirmationSheet.swift
//  Preen
//
//  会话替换确认 sheet(P0-04)。
//
//  原 confirmationDialog 不支持复选框(macOS 限制),改用自定义 sheet:
//   - 标题描述具体动作(切换模型 / 卸下 State / 更改模板 / 断开等)。
//   - 后果文案来自 SessionReplacementIntent.consequence。
//   - 复选框「本次运行内不再提醒」:勾选后本次运行(App 进程生命周期)内
//     所有会话替换动作直接执行,不弹此 sheet;重启 App 自动失效。
//     PRD P0-04 §七「不增加永久不再提醒」由"仅本次运行"维持。
//

import SwiftUI

struct SessionReplacementConfirmationSheet: View {
    let title: String
    let message: String
    let buttonTitle: String
    @Binding var suppressFuture: Bool
    let onConfirm: () -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.title2)
                        .foregroundStyle(.orange)
                    VStack(alignment: .leading, spacing: 8) {
                        Text(title)
                            .font(.headline)
                        Text(message)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer(minLength: 0)
                }

                Divider()

                Toggle(isOn: $suppressFuture) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("本次运行内不再确认会话替换")
                            .font(.callout)
                        Text("勾选后本次运行期间(重启 App 后失效)切换模型、加载/卸下 State、更改模板等会话替换动作将直接执行,不再弹此确认。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .toggleStyle(.checkbox)
            }
            .padding(20)

            Divider()

            HStack {
                Spacer()
                Button("取消", role: .cancel) { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button(buttonTitle, role: .destructive) { onConfirm() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
            .padding(16)
        }
        .frame(width: 460)
    }
}
