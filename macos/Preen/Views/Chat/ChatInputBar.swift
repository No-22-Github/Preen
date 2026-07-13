//
//  ChatInputBar.swift
//  Preen
//
//  输入栏。design.md §6:
//   - TextEditor + 发送按钮(生成中变 abort 红按钮)。
//   - busy 时禁用发送。
//   - 回车发送(Command+Return 也可,Shift+Return 换行)。
//

import SwiftUI

struct ChatInputBar: View {
    @Binding var text: String
    var canSend: Bool
    var isGenerating: Bool
    var onSend: () -> Void
    var onAbort: () -> Void

    @FocusState private var isFocused: Bool

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            // 输入区。
            TextEditor(text: $text)
                .font(.body)
                .frame(minHeight: 36, maxHeight: 120)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.quaternary, in: .rect)
                .focused($isFocused)
                .disabled(isGenerating)  // 生成时不让编辑
                .onChange(of: text) { _, new in
                    // 单行最大高度限制。
                    if new.count > 2000 { text = String(new.prefix(2000)) }
                }
                .onKeyPress(.return, phases: .down) { press in
                    // 普通回车发送(无 modifiers);Shift+Return 换行。
                    if press.modifiers.isEmpty {
                        send()
                        return .handled
                    }
                    return .ignored
                }

            // 发送 / abort 按钮(互斥)。
            if isGenerating {
                Button(role: .destructive, action: onAbort) {
                    Label("停止", systemImage: "stop.fill")
                        .frame(minWidth: 70)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .keyboardShortcut(.escape, modifiers: [])
            } else {
                Button(action: send) {
                    Label("发送", systemImage: "arrow.up")
                        .frame(minWidth: 70)
                }
                .preenGlassButton(prominent: true)
                .controlSize(.large)
                .disabled(!canSend || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .padding(12)
    }

    private func send() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, canSend else { return }
        onSend()
        text = ""
    }
}
