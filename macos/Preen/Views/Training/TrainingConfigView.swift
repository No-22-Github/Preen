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
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // 数据 & 模型选择(选完才有配置)。
                    pathsSection

                    Divider()

                    // 超参摘要：Button 让整行成为热区，同时保留键盘和 VoiceOver 语义。
                    Button {
                        withAnimation(.easeInOut(duration: 0.15)) {
                            expanded.toggle()
                        }
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: expanded ? "chevron.down" : "chevron.right")
                                .foregroundStyle(.secondary)
                                .frame(width: 16)
                                .accessibilityHidden(true)
                            Text("训练参数")
                                .font(.headline)
                            Text(config.summaryLine)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                            Spacer()
                        }
                        .padding(.vertical, 7)
                        .frame(maxWidth: .infinity, minHeight: 36, alignment: .leading)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("训练参数")
                    .accessibilityValue(expanded ? "已展开，\(config.summaryLine)" : "已折叠，\(config.summaryLine)")
                    .accessibilityHint(expanded ? "折叠详细训练参数" : "展开详细训练参数")

                    if expanded {
                        hyperparamsForm
                            .padding(.top, 8)
                            .transition(.opacity)
                    }

                    // lr 警告。
                    if config.lrWarnsExplosion {
                        Label("lr > 0.1 可能导致 state 爆炸(实测 lr=1.0 会发散),建议 lr=0.01",
                              systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .font(.caption)
                    }
                }
                .padding(24)
            }

            Divider()
            trainingActionBar
                .background(.regularMaterial)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var trainingActionBar: some View {
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
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
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
                Text(config.modelPath.isEmpty ? "请在窗口顶部选择模型" : config.modelPath)
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

    // MARK: - 超参 Form

    private var hyperparamsForm: some View {
        Form {
            Section("学习率") {
                TrainingDoubleParameterRow(
                    title: "学习率", key: "lr", detail: "State 更新步长",
                    value: $config.lr, default: 0.01
                )
                TrainingDoubleParameterRow(
                    title: "最低学习率", key: "lr_floor", detail: "余弦衰减下限",
                    value: $config.lrFloor, default: 1e-4
                )
                TrainingIntParameterRow(
                    title: "预热步数", key: "warmup", detail: "学习率预热时长",
                    value: $config.warmup, default: 10
                )
            }

            Section("训练长度") {
                TrainingIntParameterRow(
                    title: "训练轮数", key: "epochs", detail: "启用早停时为上限",
                    value: $config.epochs, default: 20
                )
                TrainingIntParameterRow(
                    title: "上下文长度", key: "ctx_len", detail: "单条样本最长 token",
                    value: $config.ctxLen, default: 512
                )
                TrainingIntParameterRow(
                    title: "日志间隔", key: "log_every", detail: "每 N 步记录一次指标",
                    value: $config.logEvery, default: 1
                )
            }

            Section("早停") {
                TrainingToggleParameterRow(
                    title: "启用早停", key: "early_stop", detail: "验证 loss 不再改善时提前停止",
                    value: $config.earlyStop
                )
                if config.earlyStop {
                    TrainingIntParameterRow(
                        title: "耐心轮数", key: "patience", detail: "允许连续无改善的轮数",
                        value: $config.earlyStopPatience, default: 3
                    )
                    TrainingDoubleParameterRow(
                        title: "验证集比例", key: "test_ratio", detail: "无 test_data 时从训练集划分",
                        value: $config.testRatio, default: 0.1
                    )
                }
            }

            Section("梯度 & checkpoint") {
                TrainingDoubleParameterRow(
                    title: "梯度裁剪", key: "grad_clip", detail: "限制梯度范数",
                    value: $config.gradClip, default: 1.0
                )
                TrainingIntParameterRow(
                    title: "Checkpoint 间隔", key: "checkpoint_every", detail: "每 N 轮保存一次",
                    value: $config.checkpointEvery, default: 2
                )
                TrainingTextParameterRow(
                    title: "Checkpoint 目录", key: "checkpoint_dir", detail: "留空则不保存",
                    prompt: "可选目录", text: $config.checkpointDir, monospaced: true
                )
                TrainingTextParameterRow(
                    title: "恢复训练", key: "resume", detail: "留空则从头开始",
                    prompt: "可选 checkpoint", text: $config.resumePath, monospaced: true
                )
            }

            Section("可复现性") {
                TrainingIntParameterRow(
                    title: "随机种子", key: "seed", detail: "控制数据划分与采样",
                    value: $config.seed, default: 42
                )
                LabeledContent {
                    Picker("任务模板", selection: $config.template) {
                        ForEach(TrainingTemplate.allCases) { template in
                            Text(template.label).tag(template)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 220)
                } label: {
                    TrainingParameterLabel(title: "任务模板", key: "template", detail: "训练与推理必须一致")
                }
                TrainingTextParameterRow(
                    title: "缓存上限", key: "cache_limit_gb", detail: "auto 或 GB 数值",
                    prompt: "auto", text: $config.cacheLimitGb
                )
            }

            Section("导出") {
                TrainingToggleParameterRow(
                    title: "导出 PTH", key: "export_pth", detail: "训练完成后同时导出",
                    value: $config.exportPth
                )
                if config.exportPth {
                    TrainingTextParameterRow(
                        title: "PTH 输出", key: "pth_out", detail: "留空则使用默认路径",
                        prompt: "默认路径", text: $config.pthOutPath, monospaced: true
                    )
                }
                TrainingTextParameterRow(
                    title: "事件日志", key: "events_file", detail: "诊断用，可选",
                    prompt: "可选文件", text: $config.eventsFilePath, monospaced: true
                )
            }
        }
        .formStyle(.grouped)
    }
}

// MARK: - 训练参数行

private struct TrainingParameterLabel: View {
    let title: String
    let key: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
            Text("\(key) · \(detail)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

private struct TrainingDoubleParameterRow: View {
    let title: String
    let key: String
    let detail: String
    @Binding var value: Double
    let `default`: Double

    var body: some View {
        LabeledContent {
            HStack(spacing: 8) {
                TextField(title, value: $value, format: .number)
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 140)
                resetButton
            }
        } label: {
            TrainingParameterLabel(title: title, key: key, detail: detail)
        }
    }

    @ViewBuilder
    private var resetButton: some View {
        if value != `default` {
            Button {
                value = `default`
            } label: {
                Image(systemName: "arrow.counterclockwise")
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.borderless)
            .help("恢复默认 \(`default`)")
        } else {
            Color.clear.frame(width: 28, height: 28)
        }
    }
}

private struct TrainingIntParameterRow: View {
    let title: String
    let key: String
    let detail: String
    @Binding var value: Int
    let `default`: Int

    var body: some View {
        LabeledContent {
            HStack(spacing: 8) {
                TextField(title, value: $value, format: .number)
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 140)
                resetButton
            }
        } label: {
            TrainingParameterLabel(title: title, key: key, detail: detail)
        }
    }

    @ViewBuilder
    private var resetButton: some View {
        if value != `default` {
            Button {
                value = `default`
            } label: {
                Image(systemName: "arrow.counterclockwise")
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.borderless)
            .help("恢复默认 \(`default`)")
        } else {
            Color.clear.frame(width: 28, height: 28)
        }
    }
}

private struct TrainingToggleParameterRow: View {
    let title: String
    let key: String
    let detail: String
    @Binding var value: Bool

    var body: some View {
        LabeledContent {
            Toggle(title, isOn: $value)
                .labelsHidden()
                .frame(width: 176, alignment: .leading)
        } label: {
            TrainingParameterLabel(title: title, key: key, detail: detail)
        }
    }
}

private struct TrainingTextParameterRow: View {
    let title: String
    let key: String
    let detail: String
    let prompt: String
    @Binding var text: String
    var monospaced = false

    var body: some View {
        LabeledContent {
            TextField(prompt, text: $text)
                .font(monospaced ? .body.monospaced() : .body)
                .textFieldStyle(.roundedBorder)
                .frame(width: 220)
        } label: {
            TrainingParameterLabel(title: title, key: key, detail: detail)
        }
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
