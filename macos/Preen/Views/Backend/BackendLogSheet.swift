import SwiftUI

/// 诊断日志视图。
///
/// 同时用于「诊断日志」窗口(Mo9 改造)和未来其它承载场景。窗口化让用户可以
/// 在训练/推理过程中持续 tail 日志(HIG: "repeated input-and-observe workflows
/// should use a panel, not a sheet")。窗口标题统一为「诊断日志」。
struct BackendLogSheet: View {
    @Bindable var store: BackendStore
    @Environment(\.dismiss) private var dismiss
    @State private var source = 0

    var body: some View {
        VStack(spacing: 0) {
            Picker("来源", selection: $source) {
                Text("Runtime").tag(0)
                Text("Serve").tag(1)
                Text("Train").tag(2)
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 440)
            .padding(.horizontal, 18)
            .padding(.top, 18)

            Group {
                if logText.isEmpty {
                    ContentUnavailableView {
                        Label(emptyTitle, systemImage: "doc.text")
                    } description: {
                        Text(emptyDescription)
                    }
                } else {
                    ScrollView {
                        Text(logText)
                            .font(.body.monospaced())
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .topLeading)
                            .padding(18)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()
            HStack {
                Spacer()
                Button("完成") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 12)
            .background(.bar)
        }
        .frame(width: 700, height: 460)
        // 窗口化场景(由 PreenApp 的 Window 场景承载)下,dismiss() 关闭窗口;
        // 作为 sheet 呈现时(向后兼容),dismiss() 收起 sheet。
        .navigationTitle("诊断日志")
    }

    private var logText: String {
        switch source {
        case 0: return store.runtimeLog
        case 1: return store.inferenceLog
        default: return store.trainingLog
        }
    }

    private var emptyTitle: String {
        switch source {
        case 0: return L10n.string("暂无运行时日志")
        case 1: return L10n.string("暂无推理日志")
        default: return L10n.string("暂无训练日志")
        }
    }

    private var emptyDescription: String {
        switch source {
        case 0: return L10n.string("环境检查正常时不会产生运行时日志。")
        case 1: return L10n.string("启动推理服务后，进程输出会显示在这里。")
        default: return L10n.string("开始训练后，进程输出会显示在这里。")
        }
    }
}
