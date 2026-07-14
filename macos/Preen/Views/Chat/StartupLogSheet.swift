//
//  StartupLogSheet.swift
//  Preen
//
//  后端启动日志弹窗:点「连接」后弹出,实时展示 serve 进程 stderr,
//  收到 ready(connected)自动关闭;启动失败则保留日志 + 失败提示供排查。
//
//  使用系统日志面板背景 + 等宽字体，自动滚到底,
//  让用户直观看到后端在干什么(加载模型 / 校验 cache / 发 ready)。
//

import SwiftUI

struct StartupLogSheet: View {
    @Bindable var store: ChatStore
    /// 关闭弹窗(成功或用户手动关)。
    var onDismiss: () -> Void
    /// 重试:重新发起连接(失败态下可用)。
    var onRetry: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            header

            Divider()

            logView
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()
            footer
        }
        .frame(width: 640, height: 380)
        .onChange(of: store.isConnected) { _, connected in
            // 收到 ready = 启动成功,自动关闭弹窗。
            if connected {
                onDismiss()
            }
        }
    }

    // MARK: - 头部

    private var header: some View {
        HStack(spacing: 10) {
            if store.isConnected {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            } else if store.startupError != nil {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            } else {
                ProgressView()
                    .controlSize(.small)
            }

            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.headline)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var title: String {
        if store.isConnected { return "后端已就绪" }
        if store.startupError != nil { return "启动失败" }
        return "正在启动后端…"
    }

    private var subtitle: String {
        if store.isConnected { return "ready 事件已收到,窗口即将关闭" }
        if store.startupError != nil { return "请查看下方日志排查原因" }
        return "加载模型可能需要数秒,请稍候"
    }

    // MARK: - 日志区

    private var logView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                // monospaced 系统日志面板。
                Text(store.startupLog.isEmpty ? "(等待输出…)" : store.startupLog)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(store.startupLog.isEmpty ? .secondary : .primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(10)
                    .id("logEnd")
            }
            .background(Color(nsColor: .textBackgroundColor))
            .overlay {
                Rectangle()
                    .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
            }
            .onChange(of: store.startupLog) { _, _ in
                withAnimation(.easeOut(duration: 0.1)) {
                    proxy.scrollTo("logEnd", anchor: .bottom)
                }
            }
        }
    }

    // MARK: - 底部

    private var footer: some View {
        HStack {
            if let err = store.startupError {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .lineLimit(2)
                    .truncationMode(.tail)
            }
            Spacer()
            if store.startupError != nil {
                Button("重试") {
                    onRetry()
                }
                .keyboardShortcut(.defaultAction)
            }
            Button(store.isConnected ? "完成" : "关闭") {
                onDismiss()
            }
            .keyboardShortcut(.cancelAction)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }
}
