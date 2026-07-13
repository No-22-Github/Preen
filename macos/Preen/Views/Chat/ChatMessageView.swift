//
//  ChatMessageView.swift
//  Preen
//
//  单条消息气泡。design.md §6:
//   - user:右对齐,accent。
//   - assistant:左对齐;think 段 dim + hairline 分隔,answer 段正常。
//   - 每条 assistant 底部:summary_line 技术摘要 dim(stop_reason/token 数/t/s)。
//   - abort 后保留已生成部分,UI 加「(已中断)」。
//

import SwiftUI

struct ChatMessageView: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user {
                Spacer(minLength: 60)
                userBubble
            } else {
                assistantBubble
                Spacer(minLength: 60)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
    }

    // MARK: - user

    private var userBubble: some View {
        Text(message.fullText)
            .textSelection(.enabled)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.accentColor.opacity(0.15), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .frame(maxWidth: 480, alignment: .trailing)
    }

    // MARK: - assistant

    private var assistantBubble: some View {
        VStack(alignment: .leading, spacing: 6) {
            // 各段。
            ForEach(message.segments) { seg in
                segmentView(seg)
            }

            // 中断标记。
            if message.isAborted {
                Text("(已中断)")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            // 错误。
            if let err = message.errorText {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }

            // 技术摘要(dim 显示)。
            if let summary = message.summary {
                Text(summary)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: 480, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    @ViewBuilder
    private func segmentView(_ seg: ChatSegment) -> some View {
        if seg.phase == .think {
            // think 段:dim + hairline 分隔(简化:始终展开,#8 会改折叠)。
            VStack(alignment: .leading, spacing: 4) {
                Text("思考")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                Text(seg.text)
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .italic()
                Divider()
            }
        } else {
            // answer 段。
            if !seg.text.isEmpty {
                Text(seg.text)
                    .font(.body)
                    .textSelection(.enabled)
            }
        }
    }
}
