import SwiftUI

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
        case 0: return "暂无运行时日志"
        case 1: return "暂无推理日志"
        default: return "暂无训练日志"
        }
    }

    private var emptyDescription: String {
        switch source {
        case 0: return "环境检查正常时不会产生运行时日志。"
        case 1: return "启动推理服务后，进程输出会显示在这里。"
        default: return "开始训练后，进程输出会显示在这里。"
        }
    }
}
