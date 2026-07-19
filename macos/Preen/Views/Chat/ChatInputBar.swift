//
//  ChatInputBar.swift
//  Preen
//
//  输入栏。design.md §6:
//   - 多行文本编辑器 + 发送按钮(生成中变 abort 红按钮)。
//   - busy 时仍可编辑,但禁用发送。
//   - 普通 Return 换行(macOS 多行编辑惯例),⌘⏎ 发送(发送按钮快捷键)。
//

import AppKit
import SwiftUI

struct ChatInputBar: View {
    @Binding var text: String
    var canSend: Bool
    var isGenerating: Bool
    var canClear: Bool
    var onSend: () -> Void
    var onAbort: () -> Void
    var onClearSession: () -> Void

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            // 输入区:自定义 PlaceholderTextEditor(NSViewRepresentable)。
            // SwiftUI TextEditor 不原生支持 placeholder,手写 ZStack overlay
            // 与 NSTextView 内部 textContainerInset + lineFragmentPadding 的对齐
            // 在不同 macOS 版本上偏 1-2pt。直接包装 NSTextView,placeholder
            // 用系统 placeholderTextColor 在与输入文字完全相同的位置渲染,
            // 保证像素级对齐。普通 Return 换行,⌘⏎ 发送(下方按钮已绑定)。
            PlaceholderTextEditor(text: $text, placeholder: "输入消息…")
                .frame(minHeight: 36, maxHeight: 120)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.quaternary.opacity(0.7), in: RoundedRectangle(cornerRadius: 10, style: .continuous))

            // 右侧按钮列:清除会话(上) + 发送/停止(下)。
            VStack(spacing: 8) {
                Button(action: onClearSession) {
                    Image(systemName: "trash")
                        .font(.body)
                        .frame(width: 30, height: 30)
                }
                .buttonStyle(.bordered)
                .disabled(!canClear || isGenerating)
                .help("清除当前会话，开始新一轮")
                .accessibilityLabel("清除当前会话")

                if isGenerating {
                    Button(role: .destructive, action: onAbort) {
                        Image(systemName: "stop.fill")
                            .font(.body)
                            .frame(width: 30, height: 30)
                    }
                    .keyboardShortcut(.escape, modifiers: [])
                    .help("停止生成 (Esc)")
                    .accessibilityLabel("停止生成")
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
                    .accessibilityLabel("发送消息")
                }
            }
        }
        .padding(12)
        .accessibilityElement(children: .contain)
    }

    private func send() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, canSend else { return }
        onSend()
        text = ""
    }
}

// MARK: - PlaceholderTextEditor

/// 带 placeholder 的多行文本编辑器。
///
/// SwiftUI 的 TextEditor 不原生支持 placeholder。这里直接包装 NSScrollView +
/// NSTextView,placeholder 在 `PreenTextView.draw(_:)` 里用与输入文字相同的
/// textContainer 位置(textContainerInset + lineFragmentPadding)渲染,
/// 因此与输入文字像素级对齐,不再依赖任何手调偏移量。
struct PlaceholderTextEditor: NSViewRepresentable {
    @Binding var text: String
    var placeholder: String

    func makeNSView(context: Context) -> NSScrollView {
        let textView = PreenTextView()
        textView.placeholderString = placeholder
        textView.font = .systemFont(ofSize: NSFont.systemFontSize)
        textView.delegate = context.coordinator
        textView.allowsUndo = true
        textView.isRichText = false
        textView.drawsBackground = false
        // 与 placeholder 共享的内边距 —— 这就是两者对齐的单一事实源。
        textView.textContainerInset = NSSize(width: 4, height: 4)
        textView.textContainer?.lineFragmentPadding = 4
        textView.isHorizontallyResizable = false
        textView.isVerticallyResizable = true
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.string = text

        let scrollView = NSScrollView()
        scrollView.documentView = textView
        scrollView.hasVerticalScroller = false
        scrollView.hasHorizontalScroller = false
        scrollView.drawsBackground = false
        scrollView.autohidesScrollers = true
        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        guard let textView = nsView.documentView as? PreenTextView else { return }
        // 仅在外部值与当前值不同时写入,避免用户正在输入时光标被重置。
        if textView.string != text {
            textView.string = text
        }
        if textView.placeholderString != placeholder {
            textView.placeholderString = placeholder
            textView.needsDisplay = true
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator($text) }

    final class Coordinator: NSObject, NSTextViewDelegate {
        let text: Binding<String>
        init(_ text: Binding<String>) { self.text = text }

        func textDidChange(_ notification: Notification) {
            guard let tv = notification.object as? NSTextView else { return }
            text.wrappedValue = tv.string
        }
    }
}

/// 自定义 NSTextView:空文本时在与输入文字相同的位置渲染 placeholder。
final class PreenTextView: NSTextView {
    var placeholderString: String = ""

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard string.isEmpty, !placeholderString.isEmpty else { return }
        // 关键:placeholder 用与输入文字完全一致的 origin(textContainerInset +
        // lineFragmentPadding),所以两者必然像素级对齐。
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font ?? .systemFont(ofSize: NSFont.systemFontSize),
            .foregroundColor: NSColor.placeholderTextColor
        ]
        let linePadding = textContainer?.lineFragmentPadding ?? 0
        let origin = NSPoint(
            x: textContainerInset.width + linePadding,
            y: textContainerInset.height
        )
        (placeholderString as NSString).draw(at: origin, withAttributes: attributes)
    }

    override func didChangeText() {
        super.didChangeText()
        // 文本变化时重绘,让 placeholder 显/隐切换。
        needsDisplay = true
    }
}
