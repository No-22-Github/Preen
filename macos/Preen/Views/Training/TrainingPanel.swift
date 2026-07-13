//
//  TrainingPanel.swift
//  Preen
//
//  训练面板四状态路由器(design.md §4):
//   [空:选数据] → [配置] → [运行中] → [完成/失败/已停止]
//
//  TrainStore.state 状态机:
//   - idle:空或配置(看 config 是否填齐决定)
//   - running / finishing:运行中(finishing 显示"收尾中")
//   - completed:完成
//   - failed:失败
//   - cancelled:已停止(保留曲线 + 可重训)
//

import SwiftUI

struct TrainingPanel: View {
    @Bindable var store: TrainStore
    @State private var config: TrainingConfig = .defaultConfig
    @State private var phase: Phase = .empty  // idle 态下的子阶段

    /// 「去对话」回调(把产物 state 路径传给对话面板)。
    var onGoToChat: (URL) -> Void

    /// idle 态子阶段。
    private enum Phase {
        case empty
        case configuring
    }

    var body: some View {
        Group {
            switch store.state {
            case .idle:
                switch phase {
                case .empty:
                    TrainingEmptyView(config: $config) {
                        phase = .configuring
                    }
                case .configuring:
                    TrainingConfigView(config: $config) {
                        store.start(config: config)
                    }
                    .toolbar {
                        ToolbarItem(placement: .navigation) {
                            Button("返回") { phase = .empty }
                        }
                    }
                }

            case .preparing:
                VStack(spacing: 12) {
                    ProgressView()
                    Text("正在创建训练记录")
                        .font(.headline)
                    Text("events 与日志目录就绪后将启动训练进程")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button("取消") { store.cancel() }
                        .buttonStyle(.bordered)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            case .running:
                TrainingRunningView(store: store)

            case .finishing:
                // final 已到,completed 未到 —— 显示"收尾中"。
                finishingView

            case .completed:
                TrainingDoneView(store: store, onGoToChat: onGoToChat)

            case .failed:
                failedView

            case .cancelled:
                cancelledView
            }
        }
        .animation(.default, value: store.state)
        .animation(.default, value: phase)
    }

    // MARK: - finishing

    private var finishingView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .controlSize(.large)
            Text("收尾中…")
                .font(.headline)
            Text("训练循环已完成,正在落盘 state 产物")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - failed

    private var failedView: some View {
        VStack(spacing: 16) {
            Image(systemName: "xmark.octagon.fill")
                .font(.system(size: 48))
                .foregroundStyle(.red)
            Text("训练失败")
                .font(.title)

            if let msg = store.errorMessage {
                ScrollView { Text(msg).textSelection(.enabled).frame(maxWidth: 500) }
                    .frame(maxHeight: 200)
                    .padding(8)
                    .background(.red.opacity(0.1), in: .rect)
            }

            HStack {
                Button("返回配置") { store.reset(); phase = .configuring }
                Button("查看曲线") { /* 留 #7:失败后保留曲线可看 */ }
                    .disabled(store.lossPoints.isEmpty)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - cancelled

    private var cancelledView: some View {
        VStack(spacing: 16) {
            Image(systemName: "stop.circle.fill")
                .font(.system(size: 48))
                .foregroundStyle(.orange)
            Text("已停止")
                .font(.title)

            if let msg = store.cancelledMessage {
                Text(msg)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text("训练已被取消,曲线已保留")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack {
                Button("返回配置") { store.reset(); phase = .configuring }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
            }
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
