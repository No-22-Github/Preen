//
//  TrainingEmptyView.swift
//  Preen
//
//  训练空态。design.md §4:拖拽区(本期简化为按钮 picker)。
//  选完数据 → 进配置态。
//

import SwiftUI
import UniformTypeIdentifiers

struct TrainingEmptyView: View {
    @Binding var config: TrainingConfig
    var recentRuns: [TrainingRun]
    var onSelectRun: (TrainingRun) -> Void
    var onConfigured: () -> Void  // 选完数据,进配置态

    @State private var isDropTargeted = false
    @SceneStorage("trainingRecentRunsInspectorPresented") private var isInspectorPresented = true

    @ViewBuilder
    var body: some View {
        if recentRuns.isEmpty {
            newRunSection
        } else {
            newRunSection
                .toolbar {
                    ToolbarItem(
                        id: "training-recent-runs-inspector",
                        placement: .primaryAction,
                        showsByDefault: true
                    ) {
                        Button {
                            isInspectorPresented.toggle()
                        } label: {
                            Label("最近训练", systemImage: "sidebar.trailing")
                        }
                        .labelStyle(.iconOnly)
                        .help(isInspectorPresented ? "隐藏最近训练" : "显示最近训练")
                        .accessibilityValue(isInspectorPresented ? "已显示" : "已隐藏")
                    }
                }
                .inspector(isPresented: $isInspectorPresented) {
                    RecentRunsView(runs: recentRuns, onSelect: onSelectRun)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 16)
                        .inspectorColumnWidth(min: 250, ideal: 280, max: 340)
                }
        }
    }

    private var newRunSection: some View {
        VStack(spacing: 16) {
            Image("PreenTitle")
                .resizable()
                .scaledToFit()
                .frame(width: 240)
                .padding(.bottom, 24)
                .accessibilityHidden(true)

            Text("选择训练数据开始")
                .font(.title2)

            Text("支持 JSONL / JSON / CSV;Alpaca / ShareGPT / ChatML / 裸 QA 自动探测")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            dropZone
                .frame(maxWidth: 440)

            VStack(spacing: 12) {
                PathRow(label: "训练数据",
                        path: $config.dataPath,
                        isDirectory: false)
                HStack {
                    Text("当前模型")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 200, alignment: .leading)
                    Text(config.modelPath.isEmpty ? "请在窗口顶部选择模型" : URL(fileURLWithPath: config.modelPath).lastPathComponent)
                        .lineLimit(1)
                        .foregroundStyle(config.modelPath.isEmpty ? .secondary : .primary)
                    Spacer()
                }
            }
            .frame(maxWidth: 440)

            if !config.dataPath.isEmpty && !config.modelPath.isEmpty {
                Button {
                    onConfigured()
                } label: {
                    Label("继续配置", systemImage: "arrow.right")
                        .frame(minWidth: 140)
                }
                .preenGlassButton(prominent: true)
                .controlSize(.large)
                .transition(.opacity)
            }
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// 拖拽接受区:把文件拖进来直接设为训练数据,也保留点击走 Open Panel。
    private var dropZone: some View {
        VStack(spacing: 4) {
            if config.dataPath.isEmpty {
                Text("拖入数据文件，或点下方「选择…」")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text(URL(fileURLWithPath: config.dataPath).lastPathComponent)
                    .font(.caption)
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .frame(maxWidth: 480, minHeight: 56)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(isDropTargeted ? Color.accentColor.opacity(0.12) : Color.clear)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(
                    isDropTargeted ? Color.accentColor : Color.secondary.opacity(0.4),
                    style: StrokeStyle(lineWidth: 1.5, dash: [6, 4])
                )
        )
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
            handleDrop(providers)
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        provider.loadItem(forTypeIdentifier: "public.file-url", options: nil) { item, _ in
            guard let data = item as? Data,
                  let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
            let ext = url.pathExtension.lowercased()
            let accepted = ["json", "jsonl", "csv"]
            DispatchQueue.main.async {
                if accepted.contains(ext) {
                    config.dataPath = url.path
                }
            }
        }
        return true
    }
}
