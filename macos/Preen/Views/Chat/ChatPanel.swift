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
import AppKit

struct ChatPanel: View {
    @Bindable var store: ChatStore
    /// 模型路径(右上角 toolbar 选的,从 app-wide 注入)。
    var modelPath: String
    /// 外部「去对话」入口注入的 state 路径(训练完成 → 跳对话,自动设上)。
    @Binding var injectedStatePath: String?
    var onConnect: () -> Void
    var onDisconnect: () -> Void

    @State private var inputText: String = ""
    /// 采样配置 sheet(独立弹窗,不再塞顶栏)。
    @State private var showSamplerSheet: Bool = false
    /// 启动日志弹窗:点「连接」后显示,ready 自动关 / 失败保留排查。
    @State private var isShowingStartupLog: Bool = false
    /// 清除会话确认弹窗。
    @State private var showClearConfirm: Bool = false
    /// 仅当用户仍位于底部附近时，才跟随流式输出。
    @State private var isFollowingLatest: Bool = true

    var body: some View {
        VStack(spacing: 0) {
            topBar
                .padding(.horizontal, 8)
                .padding(.top, 6)
                .padding(.bottom, 4)
            Divider()
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
                onRetry: { onConnect() }
            )
        }
        .sheet(isPresented: $showSamplerSheet) {
            samplerSheet
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
                let name: String = URL(fileURLWithPath: path).lastPathComponent
                Label(name, systemImage: "cpu")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            }

            // 选择 state。
            Button {
                pickState()
            } label: {
                Label("选 state", systemImage: "folder")
            }
            .disabled(!store.isConnected)

            // 采样配置(独立 sheet,省略号表示打开新界面)。
            Button {
                showSamplerSheet = true
            } label: {
                Label("采样…", systemImage: "slider.horizontal.3")
            }
            .disabled(!store.isConnected)
            .help("温度、top_p、惩罚等生成参数")

            // 连接 / 断开。
            if store.isConnected {
                Button("断开", action: onDisconnect)
            } else {
                Button {
                    onConnect()
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
        .preenGlassSurface(cornerRadius: 14)
    }

    private var samplerSheet: some View {
        VStack(spacing: 0) {
            Text("采样配置")
                .font(.headline)
                .padding(.top, 20)
                .padding(.bottom, 4)
            Text("修改后点「应用」,下一轮回复生效")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.bottom, 16)

            Form {
                Section("采样") {
                    HStack {
                        LabeledDoubleField(label: "temp", value: $store.genConfig.temperature, default: 1.2)
                        LabeledDoubleField(label: "top_p", value: $store.genConfig.topP, default: 0.5)
                        LabeledIntField(label: "max_tokens", value: $store.genConfig.maxTokens, default: 300)
                    }
                }
                Section("惩罚") {
                    HStack {
                        LabeledDoubleField(label: "presence", value: $store.genConfig.presencePenalty, default: 0.4)
                        LabeledDoubleField(label: "frequency", value: $store.genConfig.frequencyPenalty, default: 0.4)
                        LabeledDoubleField(label: "decay", value: $store.genConfig.penaltyDecay, default: 0.996)
                    }
                }
                Section("可复现性") {
                    LabeledIntField(label: "seed", value: $store.genConfig.seed, default: 42)
                }
            }
            .formStyle(.grouped)

            HStack {
                Button("取消", role: .cancel) { showSamplerSheet = false }
                Spacer()
                Button("应用") {
                    store.applyConfig()
                    showSamplerSheet = false
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
            .padding(16)
        }
        .frame(width: 460)
    }

    // MARK: - 消息列表

    private var messageList: some View {
        ScrollViewReader { proxy in
            ZStack(alignment: .bottomTrailing) {
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
            Text(store.isConnected ? "开始对话" : "点击「连接」开始对话")
                .font(.title3)
                .foregroundStyle(.secondary)
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
        .frame(maxWidth: .infinity)
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
