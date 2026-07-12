//
//  TrainingEmptyView.swift
//  Preen
//
//  训练空态。design.md §4:拖拽区(本期简化为按钮 picker)。
//  选完数据 → 进配置态。
//

import SwiftUI

struct TrainingEmptyView: View {
    @Binding var config: TrainingConfig
    var onConfigured: () -> Void  // 选完数据,进配置态

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "square.and.arrow.down")
                .font(.system(size: 56))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)

            Text("选择训练数据开始")
                .font(.title2)

            Text("支持 JSONL / JSON / CSV;Alpaca / ShareGPT / ChatML / 裸 QA 自动探测")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            VStack(spacing: 12) {
                PathRow(label: "训练数据",
                        path: $config.dataPath,
                        isDirectory: false)
                PathRow(label: "模型目录",
                        path: $config.modelPath,
                        isDirectory: true)
            }
            .frame(maxWidth: 480)

            if !config.dataPath.isEmpty && !config.modelPath.isEmpty {
                Button {
                    onConfigured()
                } label: {
                    Label("继续配置", systemImage: "arrow.right")
                        .frame(minWidth: 140)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .transition(.opacity)
            }

            Spacer()
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
