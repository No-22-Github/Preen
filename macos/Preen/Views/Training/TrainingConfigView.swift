//
//  TrainingConfigView.swift
//  Preen
//
//  训练配置表单。design.md §4:
//   - 折叠成一行摘要(lr 0.01 · ctx_len 512 · 3 轮 · 早停 patience 3 · seed 42)。
//   - 展开完整 Form,**必须包含 CLI 全部参数**。
//   - 偏离默认值时显示「恢复默认」。
//   - lr > 0.1 给 inline warning(实测 lr=1.0 会爆炸)。
//
//  本期不做导入预览 token 着色(留 #7);只放超参表单 + 开始按钮。
//

import SwiftUI

struct TrainingConfigView: View {
    @Binding var config: TrainingConfig
    @State private var expanded = false
    var onStart: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // 数据 & 模型选择(选完才有配置)。
                pathsSection

                Divider()

                // 超参摘要(折叠态)。
                summaryRow
                    .onTapGesture { withAnimation { expanded.toggle() } }

                if expanded {
                    hyperparamsForm
                        .transition(.opacity)
                }

                // lr 警告。
                if config.lrWarnsExplosion {
                    Label("lr > 0.1 可能导致 state 爆炸(实测 lr=1.0 会发散),建议 lr=0.01",
                          systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                        .font(.caption)
                }

                Divider()

                // 开始按钮。
                HStack {
                    Spacer()
                    Button(action: onStart) {
                        Label("开始训练", systemImage: "play.fill")
                            .frame(minWidth: 140)
                    }
                    .preenGlassButton(prominent: true)
                    .controlSize(.large)
                    .disabled(!config.canStart)
                    .keyboardShortcut(.return, modifiers: .command)
                }
            }
            .padding(24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - 路径区

    private var pathsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("数据 & 模型").font(.headline)

            HStack {
                Text("模型目录(HF 转换产物)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 200, alignment: .leading)
                Text(config.modelPath.isEmpty ? "请在侧边栏选择模型" : config.modelPath)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
            }

            // 训练数据。
            PathRow(label: "训练数据(JSON / JSONL)",
                    path: $config.dataPath,
                    isDirectory: false)

            // 输出 state。
            PathRow(label: "输出 state(.npz)",
                    path: $config.outPath,
                    isDirectory: false,
                    saveMode: true,
                    defaultFilename: "state.npz")
        }
    }

    // MARK: - 摘要行

    private var summaryRow: some View {
        HStack {
            Image(systemName: expanded ? "chevron.down" : "chevron.right")
                .foregroundStyle(.secondary)
                .frame(width: 16)
            Text(config.summaryLine)
                .font(.subheadline)
                .foregroundStyle(.primary)
            Spacer()
        }
        .contentShape(Rectangle())
    }

    // MARK: - 超参 Form

    private var hyperparamsForm: some View {
        Form {
            Section("学习率") {
                HStack {
                    LabeledDoubleField(label: "lr", value: $config.lr, default: 0.01)
                    LabeledDoubleField(label: "lr_floor", value: $config.lrFloor, default: 1e-4)
                    LabeledIntField(label: "warmup", value: $config.warmup, default: 10)
                }
            }

            Section("训练长度") {
                LabeledIntField(label: "epochs(配早停后是上限)", value: $config.epochs, default: 20)
                LabeledIntField(label: "ctx_len(单条样本最长 token)", value: $config.ctxLen, default: 512)
                LabeledIntField(label: "log_every(每 N 步发一条 step 事件)", value: $config.logEvery, default: 1)
            }

            Section("早停") {
                Toggle("early_stop", isOn: $config.earlyStop)
                if config.earlyStop {
                    LabeledIntField(label: "patience", value: $config.earlyStopPatience, default: 3)
                    LabeledDoubleField(label: "test_ratio(无 test_data 时从 train 划分)",
                                       value: $config.testRatio, default: 0.1)
                }
            }

            Section("梯度 & checkpoint") {
                LabeledDoubleField(label: "grad_clip", value: $config.gradClip, default: 1.0)
                LabeledIntField(label: "checkpoint_every(每 N epoch 存)", value: $config.checkpointEvery, default: 2)
                TextField("checkpoint_dir(空 = 不存)", text: $config.checkpointDir)
                    .font(.body.monospaced())
                TextField("resume(空 = 不恢复)", text: $config.resumePath)
                    .font(.body.monospaced())
            }

            Section("可复现性") {
                LabeledIntField(label: "seed", value: $config.seed, default: 42)
                Picker("template", selection: $config.template) {
                    ForEach(TrainingTemplate.allCases) { t in
                        Text(t.label).tag(t)
                    }
                }
                TextField("cache_limit_gb('auto' 或 GB 数)", text: $config.cacheLimitGb)
            }

            Section("导出") {
                Toggle("训完顺手导出 .pth", isOn: $config.exportPth)
                if config.exportPth {
                    TextField("pth_out(空 = 默认)", text: $config.pthOutPath)
                        .font(.body.monospaced())
                }
                TextField("events_file(诊断用,可选)", text: $config.eventsFilePath)
                    .font(.body.monospaced())
            }
        }
        .formStyle(.grouped)
    }
}

// MARK: - 复用小组件

/// 路径选择行(支持选目录 / 选文件 / 存模式)。
struct PathRow: View {
    let label: String
    @Binding var path: String
    var isDirectory: Bool
    var saveMode: Bool = false
    var defaultFilename: String = ""

    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 200, alignment: .leading)
            Text(path.isEmpty ? "(未选)" : URL(fileURLWithPath: path).lastPathComponent)
                .font(.body)
                .lineLimit(1)
                .truncationMode(.middle)
                .foregroundStyle(path.isEmpty ? .secondary : .primary)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button("选择…") { pick() }
        }
    }

    private func pick() {
        if saveMode {
            pickSave()
        } else {
            pickOpen()
        }
    }

    private func pickOpen() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = isDirectory
        panel.canChooseFiles = !isDirectory
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }

    private func pickSave() {
        let panel = NSSavePanel()
        if !defaultFilename.isEmpty {
            panel.nameFieldStringValue = defaultFilename
        }
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }
}

/// 带默认值的 Double 输入框 + 恢复默认。
struct LabeledDoubleField: View {
    let label: String
    @Binding var value: Double
    let `default`: Double

    var body: some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
            TextField("", value: $value, format: .number)
                .textFieldStyle(.roundedBorder)
                .frame(width: 100)
            if value != `default` {
                Button {
                    value = `default`
                } label: {
                    Image(systemName: "arrow.counterclockwise")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .help("恢复默认 \(`default`)")
            }
        }
    }
}

/// 带默认值的 Int 输入框 + 恢复默认。
struct LabeledIntField: View {
    let label: String
    @Binding var value: Int
    let `default`: Int

    var body: some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
            TextField("", value: $value, format: .number)
                .textFieldStyle(.roundedBorder)
                .frame(width: 100)
            if value != `default` {
                Button {
                    value = `default`
                } label: {
                    Image(systemName: "arrow.counterclockwise")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .help("恢复默认 \(`default`)")
            }
        }
    }
}
