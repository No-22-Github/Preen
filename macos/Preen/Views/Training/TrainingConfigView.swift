//
//  TrainingConfigView.swift
//  Preen
//
//  训练配置表单。design.md §4:
//   - 折叠成一行摘要(lr 0.0001 · ctx_len 512 · 3 轮 · 早停 patience 3 · seed 42)。
//   - 展开完整 Form,**必须包含 CLI 全部参数**。
//   - 偏离默认值时显示「恢复默认」。
//   - lr > 0.1 给 inline warning(实测 lr=1.0 会爆炸)。
//
//  本期不做导入预览 token 着色(留 #7);只放超参表单 + 开始按钮。
//

import SwiftUI
import UniformTypeIdentifiers

struct TrainingConfigView: View {
    @Binding var config: TrainingConfig
    @State private var expanded = false
    @State private var dataExpanded = false
    @State private var dataPreview: TrainingDataPreview = .empty

    // 训练前数据检查(tokenizer 统计有效数/截断/步数)。
    @State private var recordCount: Int?
    @State private var inspection: DataInspectionResult?
    @State private var inspectionError: String?
    @State private var isInspecting = false
    @State private var inspectTask: Task<Void, Never>?
    @State private var outputValidationError: String?
    private let inspector = DataInspectionRunner()
    /// 超过此条数不在改动时即时检查(避免大数据集卡顿),改由手动「检查数据」触发。
    private let autoCheckCap = 30_000

    var onStart: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // 数据 & 模型选择(选完才有配置)。
                    pathsSection

                    Divider()

                    // 轻量训练集预览(读前几条原始记录,看训练集里到底有啥)。
                    dataPreviewSection

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
                    .accessibilityValue(
                        expanded
                            ? L10n.format("已展开，%@", config.summaryLine)
                            : L10n.format("已折叠，%@", config.summaryLine)
                    )
                    .accessibilityHint(
                        L10n.string(expanded ? "折叠详细训练参数" : "展开详细训练参数")
                    )

                    if expanded {
                        hyperparamsForm
                            .padding(.top, 8)
                            .transition(.opacity)
                    }

                    // lr 警告。
                    if config.lrWarnsExplosion {
                        Label("lr > 0.1 可能导致 state 爆炸（实测 lr=1.0 会发散），建议从默认 0.0001 起步",
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
        .onAppear {
            config.refreshAutomaticOutputPath()
            onDataChanged()
        }
        .onChange(of: config.dataPath) { _, _ in
            outputValidationError = nil
            config.refreshAutomaticOutputPath()
            onDataChanged()
        }
        .onChange(of: config.modelPath) { _, _ in
            outputValidationError = nil
            config.refreshAutomaticOutputPath()
            onDataChanged()
        }
        .onChange(of: config.outPath) { _, _ in outputValidationError = nil }
        .onChange(of: config.ctxLen) { _, _ in onCtxChanged() }
        .onDisappear { inspectTask?.cancel() }
    }

    /// 读文件前几条记录填预览(纯 Swift,瞬时;失败降级为 error 文案不阻塞)。
    private func reloadDataPreview() {
        dataPreview = TrainingDataPreview.load(path: config.dataPath)
    }

    // MARK: - 训练前数据检查

    /// 数据/模型变化:刷新预览 + 数条数;≤30K 即时自动检查,>30K 等手动触发。
    private func onDataChanged() {
        reloadDataPreview()
        recordCount = TrainingDataPreview.countRecords(path: config.dataPath)
        inspection = nil
        inspectionError = nil
        if let count = recordCount, count <= autoCheckCap {
            runInspection(debounceMs: 0)
        }
    }

    /// ctx_len 变化会改变截断结果:≤30K 防抖后重查;>30K 使旧结果失效,等手动重查。
    private func onCtxChanged() {
        if let count = recordCount, count <= autoCheckCap {
            runInspection(debounceMs: 400)
        } else {
            inspection = nil
        }
    }

    /// 起一次 data-info 检查(debounce 用于 ctx_len 连续输入)。model/数据缺失或 int8 时跳过。
    private func runInspection(debounceMs: Int) {
        inspectTask?.cancel()
        guard !config.modelPath.isEmpty, !config.dataPath.isEmpty, isModelTrainable else { return }
        let model = config.modelPath
        let data = config.dataPath
        let ctx = config.ctxLen
        inspectTask = Task {
            if debounceMs > 0 {
                try? await Task.sleep(for: .milliseconds(debounceMs))
                if Task.isCancelled { return }
            }
            await MainActor.run { isInspecting = true; inspectionError = nil }
            let outcome = await inspector.inspect(modelPath: model, dataPath: data, ctxLen: ctx)
            if Task.isCancelled { return }
            await MainActor.run {
                isInspecting = false
                switch outcome {
                case .success(let result): inspection = result
                case .failure(let message): inspection = nil; inspectionError = message
                }
            }
        }
    }

    /// 模型是否可训练(int8 → false)。顶部 toolbar 可在配置态中途换模型,故这里兜底。
    private var isModelTrainable: Bool {
        ModelConfigProbe.isTrainable(modelPath: config.modelPath)
    }

    /// 不能开始训练的原因(按优先级取第一个)。nil 表示可以开始。
    /// 直接展示在按钮左侧,省得用户从上往下扫找缺哪项。
    private var blockingReason: String? {
        if config.modelPath.isEmpty { return L10n.string("请在窗口顶部选择模型") }
        if !isModelTrainable {
            return L10n.string("当前模型为 INT8，仅支持推理，请另选 BF16 模型")
        }
        if config.dataPath.isEmpty { return L10n.string("请选择训练数据") }
        if config.outPath.isEmpty { return L10n.string("无法生成输出 State 路径") }
        if let outputValidationError { return outputValidationError }
        // 截断(含完全截断)只警告不阻断,见 dataSummary。这里不再拦截。
        return nil
    }

    private var trainingActionBar: some View {
        HStack(spacing: 10) {
            statusArea
            Spacer(minLength: 8)
            Button(action: validateAndStart) {
                Label("开始训练", systemImage: "play.fill")
                    .frame(minWidth: 140)
            }
            .preenGlassButton(prominent: true)
            .controlSize(.large)
            .disabled(blockingReason != nil)
            .help(blockingReason ?? L10n.string("开始训练"))
            .keyboardShortcut(.return, modifiers: .command)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
    }

    private func validateAndStart() {
        do {
            _ = try TrainingOutputPath.validate(config: config)
            outputValidationError = nil
            onStart()
        } catch TrainingOutputPathError.destinationExists where config.outputPathMode == .automatic {
            // A directory may have appeared after the suggestion was rendered.
            // Automatic mode resolves the race to a fresh non-conflicting path.
            config.regenerateAutomaticOutputPath()
            validateAndStart()
        } catch {
            outputValidationError = error.localizedDescription
        }
    }

    /// 按钮左侧状态区:阻断原因 > 检查中 > 数据摘要 > 大数据集手动检查 > 检查失败。
    @ViewBuilder
    private var statusArea: some View {
        if let reason = blockingReason {
            Label(reason, systemImage: "exclamationmark.circle.fill")
                .foregroundStyle(.orange)
                .font(.callout)
                .lineLimit(1)
                .truncationMode(.middle)
        } else if isInspecting {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("检查数据中…").foregroundStyle(.secondary)
            }
            .font(.callout)
        } else if let insp = inspection {
            dataSummary(insp)
        } else if let count = recordCount, count > autoCheckCap {
            HStack(spacing: 8) {
                Text("约 \(count) 条 · 超过 \(autoCheckCap / 1000)K 未自动检查")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Button("检查数据") { runInspection(debounceMs: 0) }
                    .controlSize(.small)
            }
        } else if let error = inspectionError {
            // 检查失败不阻断训练(启动时 Python 侧会再兜底一次),仅提示。
            Label("数据检查未完成：\(error)", systemImage: "exclamationmark.triangle")
                .foregroundStyle(.secondary)
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    /// 数据摘要:训练/验证条数 · 预计步数 · 截断处理(只警告不阻断)。
    /// 丢弃模式:有效数扣掉截断条,步数据此重算,显示丢弃数。
    /// 保留模式:完全截断橙色警告(target 前段丢失),仅部分截断黄色提示(截头保尾)。
    private func dataSummary(_ insp: DataInspectionResult) -> some View {
        let projection = projectedCounts(insp)
        let trainCount = projection.train
        let heldOutCount = projection.heldOut
        let steps = projection.steps
        // 严重度:丢弃模式无警告(已处理) > 完全截断(橙) > 仅部分截断(黄) > 无(绿)
        let hasFullyTruncated = !config.dropTruncated && insp.targetFullyTruncated > 0
        let hasPartialOnly = !config.dropTruncated && insp.truncated > 0 && !hasFullyTruncated
        let icon: String
        let tint: Color
        if hasFullyTruncated { icon = "exclamationmark.triangle.fill"; tint = .orange }
        else if hasPartialOnly { icon = "info.circle.fill"; tint = .yellow }
        else { icon = "checkmark.seal.fill"; tint = .green }

        return HStack(spacing: 6) {
            Image(systemName: icon).foregroundStyle(tint)
            if config.earlyStop {
                Text("\(trainCount) 条训练 · \(heldOutCount) 条验证 · 预计 ~\(steps) 步")
            } else {
                Text("\(trainCount) 条训练 · 预计 ~\(steps) 步")
            }
            if config.dropTruncated, insp.truncated > 0 {
                Text("· 丢弃 \(insp.truncated) 条超长").foregroundStyle(.secondary)
            } else if hasFullyTruncated {
                Text("· \(insp.targetFullyTruncated) 条 target 完全截断（可训练，建议增大 ctx_len）")
                    .foregroundStyle(.secondary)
            } else if hasPartialOnly {
                Text("· \(insp.truncated) 条部分截断（截头保尾）").foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .lineLimit(1)
        .truncationMode(.tail)
        .help(summaryTooltip(insp))
    }

    private func summaryTooltip(_ insp: DataInspectionResult) -> String {
        var lines = [
            L10n.format("总记录 %lld · 有效 %lld", insp.total, insp.valid),
            L10n.format(
                "token: 均值 %lld · p95 %lld · max %lld",
                Int(insp.meanTokens), Int(insp.p95Tokens), insp.maxTokens
            ),
            "ctx_len \(insp.ctxLen)",
        ]
        if insp.truncated > 0 {
            lines.append(
                L10n.format(
                    "%lld 条超长,截头部保尾部 stop(target 保留,可训练)",
                    insp.truncated
                )
            )
        }
        return lines.joined(separator: "\n")
    }

    /// 按 service.run_training 的口径预估训练/验证条数与步数。
    /// 早停开启时（面板无独立 test_data 选项）从有效样本划 test_ratio 做验证，
    /// 对齐 data.train_test_split 的 max(1, int(n*ratio)) 公式，与实际 total_steps 一致；
    /// 早停关闭 = 全量训练，不划分。委托给可测的 TrainingConfig.projectedCounts。
    private func projectedCounts(_ insp: DataInspectionResult) -> (train: Int, heldOut: Int, steps: Int) {
        TrainingConfig.projectedCounts(
            effectiveValid: insp.valid,
            truncated: insp.truncated,
            dropTruncated: config.dropTruncated,
            earlyStop: config.earlyStop,
            testRatio: config.testRatio,
            epochs: config.epochs
        )
    }

    // MARK: - 路径区

    private var pathsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("数据 & 模型").font(.headline)

            HStack {
                Text("模型目录（HF 转换产物）")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 200, alignment: .leading)
                Text(
                    config.modelPath.isEmpty
                        ? L10n.string("请在窗口顶部选择模型")
                        : config.modelPath
                )
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
            }

            // 训练数据。内置资源显示只读归属；自选数据保留原 PathRow。
            if config.datasetSource == "builtin:nekoqa_200" {
                builtinDatasetRow
            } else {
                PathRow(label: "训练数据（JSON / JSONL）",
                        path: $config.dataPath,
                        isDirectory: false)
            }

            outputPathRow
        }
    }

    private var outputPathRow: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("输出 State（.npz）")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 200, alignment: .leading)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(config.outPath.isEmpty ? L10n.string("正在生成…") : config.outPath)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(config.outPath)
                    if config.outputPathMode == .automatic {
                        Text("自动")
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.quaternary, in: Capsule())
                    }
                }
                Text("State、metadata 与可选 PTH 将保存在同一目录；已有文件绝不会被覆盖。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("更改…") { pickOutputPath() }
        }
    }

    private func pickOutputPath() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "state.npz"
        panel.allowedContentTypes = [UTType(filenameExtension: "npz") ?? .data]
        if !config.outPath.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: config.outPath).deletingLastPathComponent()
        }
        guard panel.runModal() == .OK, var url = panel.url else { return }
        if url.pathExtension.lowercased() != "npz" {
            url.appendPathExtension("npz")
        }
        config.markOutputPathManual(url.path)
        do {
            _ = try TrainingOutputPath.validate(config: config)
            outputValidationError = nil
        } catch {
            outputValidationError = error.localizedDescription
        }
    }

    private var builtinDatasetRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("训练数据")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 200, alignment: .leading)
                VStack(alignment: .leading, spacing: 3) {
                    Text("内置示例 · NekoQA 200")
                        .font(.body.weight(.medium))
                    Text("200 条 · 角色风格 QA · 模板 QA · 版本 \(config.datasetVersion ?? "—")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("用于体验角色与表达风格迁移，不用于学习新知识。")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("选择自己的数据…") { pickOwnData() }
            }
            HStack(spacing: 12) {
                Color.clear.frame(width: 200, height: 1)
                Link("NekoQA-10K 来源", destination: URL(string: "https://huggingface.co/datasets/liumindmind/NekoQA-10K")!)
                if let licenseURL = try? BuiltinDataset.nekoQA200().directoryURL.appendingPathComponent("LICENSE") {
                    Link("Apache-2.0 许可证", destination: licenseURL)
                }
            }
            .font(.caption)
        }
    }

    private func pickOwnData() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            config.markDataAsUserSelected(path: url.path)
        }
    }

    // MARK: - 训练数据预览(轻量)

    @ViewBuilder
    private var dataPreviewSection: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.15)) { dataExpanded.toggle() }
        } label: {
            HStack(spacing: 10) {
                Image(systemName: dataExpanded ? "chevron.down" : "chevron.right")
                    .foregroundStyle(.secondary)
                    .frame(width: 16)
                    .accessibilityHidden(true)
                Text("训练数据预览")
                    .font(.headline)
                Text(dataPreviewSummary)
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
        .accessibilityLabel("训练数据预览")
        .accessibilityHint(
            L10n.string(dataExpanded ? "折叠训练数据预览" : "展开训练数据预览")
        )

        if dataExpanded {
            dataPreviewBody
                .padding(.top, 4)
                .transition(.opacity)
        }
    }

    /// 折叠行的一句话摘要:错误 / 前 N 条 / 空。
    private var dataPreviewSummary: String {
        if let error = dataPreview.error { return error }
        if dataPreview.samples.isEmpty { return L10n.string("无可预览记录") }
        if dataPreview.hasMore {
            return L10n.format("前 %lld 条原始记录（还有更多）", dataPreview.samples.count)
        }
        return L10n.format("前 %lld 条原始记录", dataPreview.samples.count)
    }

    @ViewBuilder
    private var dataPreviewBody: some View {
        if let error = dataPreview.error {
            Label(error, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
        } else if dataPreview.samples.isEmpty {
            Text("这个文件里读不到记录，训练前请确认数据内容。")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(dataPreview.samples) { sample in
                    VStack(alignment: .leading, spacing: 6) {
                        Text("样本 \(sample.id + 1)")
                            .font(.caption.bold())
                            .foregroundStyle(.secondary)
                        ForEach(Array(sample.fields.enumerated()), id: \.offset) { _, field in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(field.key)
                                    .font(.caption2.weight(.medium))
                                    .foregroundStyle(.secondary)
                                Text(field.value.isEmpty ? L10n.string("（空）") : field.value)
                                    .font(.callout)
                                    .foregroundStyle(field.value.isEmpty ? .tertiary : .primary)
                                    .textSelection(.enabled)
                                    .lineLimit(6)
                                    .truncationMode(.tail)
                            }
                        }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                }
                Text("原始字段直读，不含模板渲染；token 长度与截断风险见工具箱 · 数据集预览。")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - 学习率调度器行(只读,写死 cosine)

    /// 点明学习率调度策略:固定 cosine + 线性 warmup。右侧胶囊只读,与其他行右边缘对齐。
    private var schedulerRow: some View {
        LabeledContent {
            HStack(spacing: RowLayout.spacing) {
                Text("Cosine + Warmup")
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(.quaternary.opacity(0.5), in: Capsule())
                    .frame(width: RowLayout.controlWidth, alignment: .trailing)
                Color.clear.frame(width: RowLayout.resetSlot, height: RowLayout.resetSlot)
            }
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text("学习率调度")
                Text("warmup 线性升到峰值 → cosine 衰减到下限")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - 超参 Form

    private var hyperparamsForm: some View {
        Form {
            Section("学习率") {
                // 调度器写死 cosine(state tuning 实测最稳),这一行点明下面三个参数构成一条调度曲线。
                schedulerRow
                TrainingDoubleParameterRow(
                    title: "学习率", key: "lr", detail: "调度峰值(warmup 升到此)",
                    value: $config.lr, default: 1e-4
                )
                TrainingDoubleParameterRow(
                    title: "最低学习率", key: "lr_floor", detail: "cosine 衰减终点(下限)",
                    value: $config.lrFloor, default: 1e-5
                )
                TrainingIntParameterRow(
                    title: "预热步数", key: "warmup", detail: "前 N 步从 0 线性升到峰值",
                    value: $config.warmup, default: 50
                )
            }

            Section("训练长度") {
                TrainingIntParameterRow(
                    title: "训练轮数", key: "epochs", detail: "启用早停时为上限",
                    value: $config.epochs, default: 5
                )
                TrainingIntParameterRow(
                    title: "上下文长度", key: "ctx_len", detail: "单条样本最长 token",
                    value: $config.ctxLen, default: 512
                )
                TrainingIntParameterRow(
                    title: "日志间隔", key: "log_every", detail: "每 N 步记录一次指标",
                    value: $config.logEvery, default: 1
                )
                TrainingToggleParameterRow(
                    title: "丢弃超长样本", key: "drop_truncated",
                    detail: "关=截头保尾继续训练(默认) · 开=直接丢弃",
                    value: $config.dropTruncated
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
                    HStack(spacing: RowLayout.spacing) {
                        Picker("任务模板", selection: $config.template) {
                            ForEach(TrainingTemplate.allCases) { template in
                                Text(template.label).tag(template)
                            }
                        }
                        .labelsHidden()
                        .frame(width: RowLayout.controlWidth)
                        Color.clear.frame(width: RowLayout.resetSlot, height: RowLayout.resetSlot)
                    }
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

/// 所有参数行共享的右侧布局:控件等宽 + 固定复位槽,保证左右边缘对齐。
private enum RowLayout {
    static let controlWidth: CGFloat = 200   // 输入框 / 选择器 / 开关的统一宽度
    static let resetSlot: CGFloat = 28        // 复位按钮槽(无按钮时留等宽透明占位)
    static let spacing: CGFloat = 8
}

/// 复位按钮槽:偏离默认值时显示复位按钮,否则等宽透明占位(保证右边缘不跳)。
private struct ResetSlot: View {
    let show: Bool
    let help: String
    let action: () -> Void

    var body: some View {
        if show {
            Button(action: action) {
                Image(systemName: "arrow.counterclockwise")
                    .frame(width: RowLayout.resetSlot, height: RowLayout.resetSlot)
            }
            .buttonStyle(.borderless)
            .help(help)
        } else {
            Color.clear.frame(width: RowLayout.resetSlot, height: RowLayout.resetSlot)
        }
    }
}

private struct TrainingParameterLabel: View {
    let title: String
    let key: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(L10n.string(title))
            Text("\(key) · \(L10n.string(detail))")
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
            HStack(spacing: RowLayout.spacing) {
                TextField(L10n.string(title), value: $value, format: .number)
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(width: RowLayout.controlWidth)
                ResetSlot(show: value != `default`, help: L10n.format("恢复默认 %g", `default`)) {
                    value = `default`
                }
            }
        } label: {
            TrainingParameterLabel(title: title, key: key, detail: detail)
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
            HStack(spacing: RowLayout.spacing) {
                TextField(L10n.string(title), value: $value, format: .number)
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(width: RowLayout.controlWidth)
                ResetSlot(show: value != `default`, help: L10n.format("恢复默认 %lld", `default`)) {
                    value = `default`
                }
            }
        } label: {
            TrainingParameterLabel(title: title, key: key, detail: detail)
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
            HStack(spacing: RowLayout.spacing) {
                Toggle(L10n.string(title), isOn: $value)
                    .labelsHidden()
                    .frame(width: RowLayout.controlWidth, alignment: .leading)
                // 开关行无复位,留等宽透明槽保持右边缘对齐。
                Color.clear.frame(width: RowLayout.resetSlot, height: RowLayout.resetSlot)
            }
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
            HStack(spacing: RowLayout.spacing) {
                TextField(L10n.string(prompt), text: $text)
                    .labelsHidden()  // 否则 prompt 会作为标签渲染到框外(macOS)
                    .font(monospaced ? .body.monospaced() : .body)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: RowLayout.controlWidth)
                // 文本行无复位,留等宽透明槽保持右边缘对齐。
                Color.clear.frame(width: RowLayout.resetSlot, height: RowLayout.resetSlot)
            }
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
            Text(L10n.string(label))
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 200, alignment: .leading)
            Text(path.isEmpty ? L10n.string("（未选）") : path)
                .font(.body)
                .lineLimit(1)
                .truncationMode(.middle)
                .foregroundStyle(path.isEmpty ? .secondary : .primary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .help(path.isEmpty ? L10n.string("尚未选择路径") : path)
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
            Text(L10n.string(label)).font(.caption).foregroundStyle(.secondary)
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
                .help(L10n.format("恢复默认 %g", `default`))
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
            Text(L10n.string(label)).font(.caption).foregroundStyle(.secondary)
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
                .help(L10n.format("恢复默认 %lld", `default`))
            }
        }
    }
}
