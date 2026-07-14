//
//  ChatInputBar.swift
//  Preen
//
//  输入栏。design.md §6:
//   - TextEditor + 发送按钮(生成中变 abort 红按钮)。
//   - busy 时仍可编辑,但禁用发送。
//   - 回车发送(Command+Return 也可,Shift+Return 换行)。
//

import SwiftUI

struct ChatInputBar: View {
    @Binding var text: String
    var canSend: Bool
    var isGenerating: Bool
    var canClear: Bool
    var onSend: () -> Void
    var onAbort: () -> Void
    var onClearSession: () -> Void

    @FocusState private var isFocused: Bool

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            // 输入区。
            TextEditor(text: $text)
                .font(.body)
                .scrollContentBackground(.hidden)
                .frame(minHeight: 36, maxHeight: 120)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.quaternary.opacity(0.7), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                .focused($isFocused)
                .onKeyPress(.return, phases: .down) { press in
                    // 普通回车发送(无 modifiers);Shift+Return 换行。
                    if press.modifiers.isEmpty && canSend {
                        send()
                        return .handled
                    }
                    return .ignored
                }

            // 右侧按钮列:清除会话(上) + 发送/停止(下)。
            VStack(spacing: 8) {
                Button(action: onClearSession) {
                    Image(systemName: "trash")
                        .font(.body)
                        .frame(width: 30, height: 30)
                }
                .buttonStyle(.bordered)
                .disabled(!canClear || isGenerating)
                .help("清除当前会话,开始新一轮")

                if isGenerating {
                    Button(role: .destructive, action: onAbort) {
                        Image(systemName: "stop.fill")
                            .font(.body)
                            .frame(width: 30, height: 30)
                    }
                    .keyboardShortcut(.escape, modifiers: [])
                    .help("停止生成 (Esc)")
                } else {
                    Button(action: send) {
                        Image(systemName: "arrow.up")
                            .font(.body.weight(.semibold))
                            .frame(width: 30, height: 30)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canSend || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .keyboardShortcut(.return, modifiers: .command)
                    .help("发送 (⌘⏎)")
                }
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
