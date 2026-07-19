//
//  ChatRawContinuationView.swift
//  Preen
//
//  RAW 模板纯续写视图(P0-01)。
//
//  RAW 模板对应"模型从给定前缀往后直接续写",没有 User/Assistant 包装。
//  对应 RWKV 这类因果语言模型最基础的用法。
//
//  布局:
//   - 上半:大 TextEditor,任意前缀文本(占满主体)。
//   - 下半:模型续写只读区,实时显示生成内容(流式追加)。
//  底部:`续写` / `停止` / `清空`。
//
//  复用 store.send / store.abort / store.isGenerating;store 把续写内容作为
//  一条 assistant message 放进 messages 列表,这里只显示最后一条 assistant。
//

import SwiftUI

struct ChatRawContinuationView: View {
    @Bindable var store: ChatStore
    @Binding var inputText: String
    var onAbort: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            // 上半:前缀输入(TextEditor 占满剩余空间)。
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("前缀文本")
                        .font(.headline)
                    Spacer()
                    Text("模型将从这个文本末尾往后续写")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                TextEditor(text: $inputText)
                    .font(.body.monospaced())
                    .scrollContentBackground(.hidden)
                    .background(.quaternary.opacity(0.2), in: RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(.quaternary, lineWidth: 0.5)
                    )
                    .disabled(store.isGenerating)
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .padding(.bottom, 8)

            Divider().padding(.vertical, 4)

            // 下半:模型续写结果(只读,实时流式)。
            continuationOutput
                .padding(.horizontal, 16)
                .padding(.bottom, 12)

            Divider()

            actionBar
                .background(.regularMaterial)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - 续写输出

    @ViewBuilder
    private var continuationOutput: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("模型续写")
                    .font(.headline)
                Spacer()
                if store.isGenerating {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("生成中…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else if let last = lastAssistantMessage, !last.segments.isEmpty,
                          last.fullText.isEmpty == false {
                    Label("完成", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
            }
            if let last = lastAssistantMessage, last.fullText.isEmpty == false {
                ScrollView {
                    Text(last.fullText)
                        .font(.body.monospaced())
                        .foregroundStyle(.primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(10)
                }
                .background(Color.accentColor.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
                .overlay(alignment: .leading) {
                    Rectangle()
                        .fill(Color.accentColor.opacity(0.6))
                        .frame(width: 2)
                }
            } else if let err = store.lastError {
                Label(err, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.caption)
                    .textSelection(.enabled)
            } else {
                ContentUnavailableView(
                    "等待续写",
                    systemImage: "text.append",
                    description: Text("在前缀文本中输入任意内容,点击「续写」让模型往后生成。")
                )
                .frame(maxWidth: .infinity, minHeight: 140)
            }
        }
    }

    // MARK: - 底部操作栏

    private var actionBar: some View {
        HStack(spacing: 10) {
            if store.isGenerating {
                Button(role: .destructive, action: onAbort) {
                    Label("停止", systemImage: "stop.fill")
                        .frame(minWidth: 120)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.escape, modifiers: [])
            } else {
                Button(action: continueGeneration) {
                    Label("续写", systemImage: "play.fill")
                        .frame(minWidth: 120)
                }
                .buttonStyle(.borderedProminent)
                .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          || !store.isConnected)
                .help(store.isConnected
                      ? "从输入文本末尾继续生成"
                      : "未连接后端")
            }

            Spacer()

            Button("清空续写") {
                clearContinuationOnly()
            }
            .buttonStyle(.bordered)
            .disabled(store.isGenerating || lastAssistantMessage == nil)
            .help("仅清空模型续写结果,保留前缀文本")

            Button("采纳续写") {
                adoptContinuationToInput()
            }
            .buttonStyle(.bordered)
            .disabled(store.isGenerating || lastAssistantMessage?.fullText.isEmpty ?? true)
            .help("把模型生成结果拼到前缀文本末尾,可继续往后续写")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    // MARK: - 动作

    private func continueGeneration() {
        let text = inputText
        guard !text.isEmpty else { return }
        store.send(text: text)
    }

    private func clearContinuationOnly() {
        // 清空 messages 列表里所有 assistant 续写(保留空会话)。
        // RAW 模板下 messages 不区分多轮,只保留当前前缀输入框的内容。
        store.newSession()
    }

    private func adoptContinuationToInput() {
        guard let last = lastAssistantMessage, !last.fullText.isEmpty else { return }
        inputText = inputText + last.fullText
        store.newSession()
    }

    private var lastAssistantMessage: ChatMessage? {
        store.messages.last(where: { $0.role == .assistant })
    }
}
