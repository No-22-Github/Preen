//
//  ChatPanel.swift
//  Preen
//
//  对话面板最小版(单栏)。design.md §6 但本期只做单栏:
//   - 顶部:state 选择器 + 连接 / 断开按钮 + 采样配置(简化版,折叠)。
//   - 主体:消息列表(自动滚到底)。
//   - 底部:输入栏。
//
//  本期不做(留 #8):
//   - A/B 双栏(默认开双栏是 design.md §6.1 的核心,但本期最小版单栏)。
//   - Inspector(⌥⌘I)完整会话区。
//   - 崩溃恢复重放(Spec §5 验收 c)。
//   - /rewind 按钮(design.md §6.3:消息 hover 出「回到这里」)。
//

import SwiftUI

struct ChatPanel: View {
    @Bindable var store: ChatStore
    /// 模型路径(侧边栏选的,从 app-wide 注入)。
    var modelPath: String
    /// 外部「去对话」入口注入的 state 路径(训练完成 → 跳对话,自动设上)。
    @Binding var injectedStatePath: String?

    @State private var inputText: String = ""
    @State private var showSampler: Bool = false
    /// 启动日志弹窗:点「连接」后显示,ready 自动关 / 失败保留排查。
    @State private var isShowingStartupLog: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Divider()
            messageList
            Divider()
            ChatInputBar(
                text: $inputText,
                canSend: store.canSend,
                isGenerating: store.isGenerating,
                onSend: { store.send(text: inputText); inputText = "" },
                onAbort: { store.abort() }
            )
        }
        .sheet(isPresented: $isShowingStartupLog) {
            StartupLogSheet(store: store) {
                isShowingStartupLog = false
            }
        }
        .onChange(of: injectedStatePath) { _, newPath in
            // 训练完成跳来:自动连 + 切 state。
            if let path = newPath, store.isConnected {
                store.setState(path: path)
            }
        }
    }

    // MARK: - 顶栏

    private var topBar: some View {
        HStack(spacing: 12) {
            // 连接状态指示。
            Circle()
                .fill(store.isConnected ? Color.green : Color.secondary)
                .frame(width: 8, height: 8)
            Text(store.isConnected ? "已连接" : "未连接")
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()

            // state 显示。
            if let path = store.statePath {
                Label(URL(fileURLWithPath: path).lastPathComponent, systemImage: "cpu")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.quaternary, in: .rect)
            }

            // 选择 state。
            Button {
                pickState()
            } label: {
                Label("选 state", systemImage: "folder")
            }
            .disabled(!store.isConnected)

            // 采样配置(折叠 DisclosureGroup)。
            DisclosureGroup("采样", isExpanded: $showSampler) {
                samplerControls
            }
            .font(.caption)

            // 连接 / 断开。
            if store.isConnected {
                Button("断开") { store.disconnect() }
            } else {
                Button {
                    store.connect(model: URL(fileURLWithPath: modelPath))
                    // 弹出启动日志窗口,实时看后端输出,ready 自动关。
                    isShowingStartupLog = true
                } label: {
                    Label("连接", systemImage: "link")
                }
                .disabled(modelPath.isEmpty)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(height: 44)
    }

    private var samplerControls: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                LabeledDoubleField(label: "temp", value: $store.genConfig.temperature, default: 1.2)
                LabeledDoubleField(label: "top_p", value: $store.genConfig.topP, default: 0.5)
                LabeledIntField(label: "max_tokens", value: $store.genConfig.maxTokens, default: 300)
            }
            HStack {
                LabeledIntField(label: "seed", value: $store.genConfig.seed, default: 42)
                LabeledDoubleField(label: "presence", value: $store.genConfig.presencePenalty, default: 0.4)
                LabeledDoubleField(label: "frequency", value: $store.genConfig.frequencyPenalty, default: 0.4)
            }
            HStack {
                LabeledDoubleField(label: "decay", value: $store.genConfig.penaltyDecay, default: 0.996)
                Button("应用(下轮生效)") { store.applyConfig() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
        }
        .padding(.vertical, 4)
    }

    // MARK: - 消息列表

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 4) {
                    if store.messages.isEmpty {
                        emptyState
                            .padding(.top, 80)
                    }
                    ForEach(store.messages) { msg in
                        ChatMessageView(message: msg)
                            .id(msg.id)
                    }
                }
                .padding(.vertical, 8)
            }
            .onChange(of: store.messages.count) { _, _ in
                // 自动滚到底(新消息到达时)。
                if let last = store.messages.last {
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
            .onChange(of: store.messages.last?.segments.last?.text) { _, _ in
                // 流式追加时也滚(同一消息的段文本变长)。
                if let last = store.messages.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - 空态

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text(store.isConnected ? "开始对话" : "点击「连接」开始对话")
                .font(.title3)
                .foregroundStyle(.secondary)
            if store.isConnected && store.statePath == nil {
                Text("未选 state —— 当前是基线(无 state)模式")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            if let err = store.lastError {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal, 16)
                    .frame(maxWidth: 400)
            }
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - 内部

    private func pickState() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.data]  // .npz 走 UTI data
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            store.setState(path: url.path)
        }
    }
}
