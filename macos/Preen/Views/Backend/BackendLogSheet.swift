import SwiftUI

struct BackendLogSheet: View {
    @Bindable var store: BackendStore
    @Environment(\.dismiss) private var dismiss
    @State private var source = 0

    var body: some View {
        VStack(spacing: 12) {
            Picker("来源", selection: $source) {
                Text("Runtime").tag(0)
                Text("Serve").tag(1)
                Text("Train").tag(2)
            }
            .pickerStyle(.segmented)
            TextEditor(text: .constant(logText.isEmpty ? "暂无日志" : logText))
                .font(.body.monospaced())
                .disabled(true)
            HStack { Spacer(); Button("完成") { dismiss() } }
        }
        .padding(18)
        .frame(width: 700, height: 460)
    }

    private var logText: String {
        switch source {
        case 0: return store.runtimeLog
        case 1: return store.inferenceLog
        default: return store.trainingLog
        }
    }
}
