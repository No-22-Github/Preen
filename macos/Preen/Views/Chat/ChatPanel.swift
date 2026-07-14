//
//  ChatPanel.swift
//  Preen
//
//  对话面板最小版(单栏)。design.md §6 但本期只做单栏:
//   - 对话操作位于系统 toolbar,连接按钮位于未连接空态。
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
import AppKit

struct ChatPanel: View {
    @Bindable var store: ChatStore
    /// 模型路径(顶部中央 toolbar 选的,从 app-wide 注入)。
    var modelPath: String
    /// 外部「去对话」入口注入的 state 路径(训练完成 → 跳对话,自动设上)。
    @Binding var injectedStatePath: String?
    /// 由窗口 toolbar 打开的生成参数 sheet。
    @Binding var isShowingGenerationParameters: Bool
    var onConnect: () -> Void

    @State private var inputText: String = ""
    /// 启动日志弹窗:点「连接」后显示,ready 自动关 / 失败保留排查。
    @State private var isShowingStartupLog: Bool = false
    /// 清除会话确认弹窗。
    @State private var showClearConfirm: Bool = false
    /// 仅当用户仍位于底部附近时，才跟随流式输出。
    @State private var isFollowingLatest: Bool = true

    var body: some View {
        VStack(spacing: 0) {
            messageList
            Divider()
            if let error = store.lastError {
                chatErrorBanner(error)
                Divider()
            }
            ChatInputBar(
                text: $inputText,
                canSend: store.canSend,
                isGenerating: store.isGenerating,
                canClear: store.isConnected && !store.messages.isEmpty,
                onSend: {
                    isFollowingLatest = true
                    store.send(text: inputText)
                    inputText = ""
                },
                onAbort: { store.abort() },
                onClearSession: { showClearConfirm = true }
            )
        }
        .confirmationDialog(
            "清除当前会话?",
            isPresented: $showClearConfirm,
            titleVisibility: .visible
        ) {
            Button("清除", role: .destructive) {
                store.newSession()
            }
            Button("取消", role: .cancel) {}
        } message: {
            Text("将清空所有对话消息并开始新一轮。此操作无法撤销。")
        }
        .sheet(isPresented: $isShowingStartupLog) {
            StartupLogSheet(
                store: store,
                onDismiss: { isShowingStartupLog = false },
                onRetry: { startConnection() }
            )
        }
        .sheet(isPresented: $isShowingGenerationParameters) {
            samplerSheet
        }
        .onChange(of: injectedStatePath) { _, newPath in
            // 训练完成跳来:自动连 + 切 state。
            if let path = newPath, store.isConnected {
                store.setState(path: path)
            }
        }
    }

    private var samplerSheet: some View {
        VStack(spacing: 0) {
            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("生成参数")
                        .font(.headline)
                    Text("修改将从下一轮回复开始生效")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button("恢复默认") {
                    store.genConfig = .defaultConfig
                }
                .disabled(store.genConfig == .defaultConfig)
            }
            .padding(20)

            Divider()

            ScrollView {
                VStack(spacing: 14) {
                    parameterSection(title: "采样") {
                        GenerationDoubleField(
                            title: "温度",
                            detail: "控制回答的随机性",
                            value: $store.genConfig.temperature,
                            defaultValue: 1.2
                        )
                        Divider()
                        GenerationDoubleField(
                            title: "核采样",
                            detail: "限制候选 token 范围",
                            value: $store.genConfig.topP,
                            defaultValue: 0.5
                        )
                        Divider()
                        GenerationIntField(
                            title: "最大长度",
                            detail: "单次回复的 token 上限",
                            value: $store.genConfig.maxTokens,
                            defaultValue: 300
                        )
                    }

                    parameterSection(title: "重复惩罚") {
                        GenerationDoubleField(
                            title: "出现惩罚",
                            detail: "降低已出现内容再次生成的概率",
                            value: $store.genConfig.presencePenalty,
                            defaultValue: 0.4
                        )
                        Divider()
                        GenerationDoubleField(
                            title: "频率惩罚",
                            detail: "按出现次数增加惩罚",
                            value: $store.genConfig.frequencyPenalty,
                            defaultValue: 0.4
                        )
                        Divider()
                        GenerationDoubleField(
                            title: "惩罚衰减",
                            detail: "控制重复惩罚随时间衰减",
                            value: $store.genConfig.penaltyDecay,
                            defaultValue: 0.996
                        )
                    }

                    parameterSection(title: "可复现性") {
                        GenerationIntField(
                            title: "随机种子",
                            detail: "相同参数下复现采样结果",
                            value: $store.genConfig.seed,
                            defaultValue: 42
                        )
                    }
                }
                .padding(20)
            }

            Divider()

            HStack {
                Button("取消", role: .cancel) { isShowingGenerationParameters = false }
                Spacer()
                Button("应用") {
                    store.applyConfig()
                    isShowingGenerationParameters = false
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
            .padding(20)
        }
        .frame(width: 520, height: 600)
    }

    private func parameterSection<Content: View>(
        title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 4)

            VStack(spacing: 10) {
                content()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
    }

    // MARK: - 消息列表

    private var messageList: some View {
        Group {
            if store.isConnected {
                connectedMessageList
            } else {
                emptyState
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var connectedMessageList: some View {
        ScrollViewReader { proxy in
            ZStack(alignment: .bottomTrailing) {
                ScrollView {
                    LazyVStack(spacing: 4) {
                        ForEach(store.messages) { msg in
                            ChatMessageView(message: msg)
                                .id(msg.id)
                        }
                        Color.clear
                            .frame(height: 1)
                            .id("chat-bottom")
                    }
                    .padding(.vertical, 8)
                }
                .background(
                    ChatScrollPositionObserver(isNearBottom: $isFollowingLatest, tolerance: 80)
                        .frame(width: 0, height: 0)
                )
                .onAppear {
                    guard !store.messages.isEmpty else { return }
                    DispatchQueue.main.async {
                        proxy.scrollTo("chat-bottom", anchor: .bottom)
                    }
                }
                .onChange(of: store.messages.count) { _, _ in
                    guard isFollowingLatest else { return }
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo("chat-bottom", anchor: .bottom)
                    }
                }
                .onChange(of: store.messages.last?.segments.last?.text) { _, _ in
                    guard isFollowingLatest else { return }
                    proxy.scrollTo("chat-bottom", anchor: .bottom)
                }

                if store.messages.isEmpty {
                    emptyState
                }

                if !isFollowingLatest && !store.messages.isEmpty {
                    Button {
                        isFollowingLatest = true
                        withAnimation(.easeOut(duration: 0.15)) {
                            proxy.scrollTo("chat-bottom", anchor: .bottom)
                        }
                    } label: {
                        Label("回到最新消息", systemImage: "arrow.down")
                    }
                    .buttonStyle(.bordered)
                    .padding(14)
                    .help("恢复跟随流式输出")
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
            Text(store.isConnected ? "开始对话" : "连接本地模型")
                .font(.title3)
                .foregroundStyle(.secondary)
            if !store.isConnected {
                Button {
                    startConnection()
                } label: {
                    if store.hasActiveProcess {
                        HStack(spacing: 8) {
                            ProgressView()
                                .controlSize(.small)
                            Text("正在连接…")
                        }
                    } else {
                        Label("连接", systemImage: "link")
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(modelPath.isEmpty || store.hasActiveProcess)
            }
            if store.isConnected && store.statePath == nil {
                Text("未选 state —— 当前是基线(无 state)模式")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            Text("回答由本地 AI 模型生成，可能不准确，请核实重要信息。")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.top, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func chatErrorBanner(_ error: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            Text(error)
                .font(.caption)
                .foregroundStyle(.primary)
                .textSelection(.enabled)
            Spacer()
            Button {
                store.clearLastError()
            } label: {
                Image(systemName: "xmark")
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.plain)
            .help("关闭错误提示")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.red.opacity(0.08))
    }

    // MARK: - 内部

    private func startConnection() {
        onConnect()
        // 弹出启动日志窗口,实时看后端输出,ready 自动关。
        isShowingStartupLog = true
    }

}

/// 生成参数面板专用的小数输入行。恢复按钮始终占位，避免值变化时布局跳动。
private struct GenerationDoubleField: View {
    let title: String
    let detail: String
    @Binding var value: Double
    let defaultValue: Double

    var body: some View {
        GenerationParameterRow(title: title, detail: detail) {
            TextField(title, value: $value, format: .number)
                .labelsHidden()
                .multilineTextAlignment(.trailing)
                .textFieldStyle(.roundedBorder)
                .frame(width: 92)

            resetButton
        }
    }

    private var resetButton: some View {
        Button {
            value = defaultValue
        } label: {
            Image(systemName: "arrow.counterclockwise")
                .frame(width: 18, height: 18)
        }
        .buttonStyle(.borderless)
        .disabled(value == defaultValue)
        .opacity(value == defaultValue ? 0 : 1)
        .help("恢复默认值 \(defaultValue)")
        .frame(width: 24)
    }
}

/// 生成参数面板专用的整数输入行。
private struct GenerationIntField: View {
    let title: String
    let detail: String
    @Binding var value: Int
    let defaultValue: Int

    var body: some View {
        GenerationParameterRow(title: title, detail: detail) {
            TextField(title, value: $value, format: .number.grouping(.never))
                .labelsHidden()
                .multilineTextAlignment(.trailing)
                .textFieldStyle(.roundedBorder)
                .frame(width: 92)

            resetButton
        }
    }

    private var resetButton: some View {
        Button {
            value = defaultValue
        } label: {
            Image(systemName: "arrow.counterclockwise")
                .frame(width: 18, height: 18)
        }
        .buttonStyle(.borderless)
        .disabled(value == defaultValue)
        .opacity(value == defaultValue ? 0 : 1)
        .help("恢复默认值 \(defaultValue)")
        .frame(width: 24)
    }
}

private struct GenerationParameterRow<Controls: View>: View {
    let title: String
    let detail: String
    @ViewBuilder let controls: () -> Controls

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.body)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 16)

            HStack(spacing: 4) {
                controls()
            }
        }
        .frame(minHeight: 38)
    }
}

/// macOS 14 没有 SwiftUI 的 scroll geometry API，通过 NSScrollView 仅观察用户滚动。
/// 流式文本使 document 变长时不会误判为用户离开底部。
private struct ChatScrollPositionObserver: NSViewRepresentable {
    @Binding var isNearBottom: Bool
    let tolerance: CGFloat

    func makeCoordinator() -> Coordinator {
        Coordinator(isNearBottom: $isNearBottom, tolerance: tolerance)
    }

    func makeNSView(context: Context) -> NSView {
        let view = NSView(frame: .zero)
        DispatchQueue.main.async {
            context.coordinator.attach(from: view)
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.isNearBottom = $isNearBottom
        if context.coordinator.scrollView == nil {
            DispatchQueue.main.async {
                context.coordinator.attach(from: nsView)
            }
        }
    }

    static func dismantleNSView(_ nsView: NSView, coordinator: Coordinator) {
        coordinator.detach()
    }

    final class Coordinator {
        var isNearBottom: Binding<Bool>
        let tolerance: CGFloat
        weak var scrollView: NSScrollView?
        private var boundsObserver: NSObjectProtocol?

        init(isNearBottom: Binding<Bool>, tolerance: CGFloat) {
            self.isNearBottom = isNearBottom
            self.tolerance = tolerance
        }

        func attach(from view: NSView) {
            guard scrollView == nil else { return }
            var ancestor = view.superview
            while let current = ancestor, !(current is NSScrollView) {
                ancestor = current.superview
            }
            guard let scrollView = ancestor as? NSScrollView else { return }
            self.scrollView = scrollView
            scrollView.contentView.postsBoundsChangedNotifications = true
            boundsObserver = NotificationCenter.default.addObserver(
                forName: NSView.boundsDidChangeNotification,
                object: scrollView.contentView,
                queue: .main
            ) { [weak self] _ in
                self?.updatePosition()
            }
            updatePosition()
        }

        func detach() {
            if let boundsObserver {
                NotificationCenter.default.removeObserver(boundsObserver)
            }
            boundsObserver = nil
            scrollView = nil
        }

        private func updatePosition() {
            guard let scrollView, let documentView = scrollView.documentView else { return }
            let visible = scrollView.documentVisibleRect
            let distance: CGFloat
            if documentView.isFlipped {
                distance = documentView.bounds.maxY - visible.maxY
            } else {
                distance = visible.minY - documentView.bounds.minY
            }
            let nearBottom = distance <= tolerance
            if isNearBottom.wrappedValue != nearBottom {
                isNearBottom.wrappedValue = nearBottom
            }
        }
    }
}
