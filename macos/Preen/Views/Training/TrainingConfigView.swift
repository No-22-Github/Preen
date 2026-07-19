//
//  TrainingConfigView.swift
//  Preen
//
//  训练配置表单。design.md §4:
//   - 折叠成一行摘要(lr 0.0001 · ctx_len 512 · 3 轮 · 早停 patience 3 · seed 42)。
//   - 展开完整 Form,**必须包含 CLI 全部参数**。
//   - 偏离默认值时显示「恢复默认」。
//   - lr > 0.1 给 inline warning(实测 lr=1.0 会爆炸)。
//  P0-07:训练入口复用工具箱同源预检，展示最终 prefix / target、token 与截断口径。

import SwiftUI
import UniformTypeIdentifiers

struct TrainingConfigView: View {
    @Binding var config: TrainingConfig
    @State private var expanded = false
    @State private var dataExpanded = false

    // 训练前数据检查：探测、最终模板渲染与 tokenizer 统计来自同一个 Python 命令。
    @State private var recordCount: Int?
    @State private var preflight: DatasetPreviewResult?
    @State private var preflightWasCached = false
    @State private var inspectionError: String?
    @State private var isInspecting = false
    @State private var inspectTask: Task<Void, Never>?
    @State private var outputValidationError: String?
    private let inspector = TrainingDataPreflightRunner()
    /// 2026-07-18 macOS 26 / M4 实测：10,066 条冷启动全量渲染与 tokenize 为 2.94s；
    /// 锁定 10K 自动阈值，规模更大时由用户显式触发，且结果完成前不允许训练。
    private let autoCheckCap = 10_000

    var onStart: () -> Void
    var onConfigureImport: (String) -> Void

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // 数据 & 模型选择(选完才有配置)。
                    pathsSection

                    Divider()

                    // 与训练同源的 schema、token 全量统计与最终模板文本。
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
        // Mo1:开始训练上移到 detail toolbar primaryAction,
        // 避免被窗口底部遮挡(macOS 用户常把窗口拖到屏幕底部之下)。
        // 底部 ActionBar 保留 statusArea(阻断原因/数据摘要)。
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(action: validateAndStart) {
                    Label("开始训练", systemImage: "play.fill")
                }
                .preenGlassButton(prominent: true)
                .disabled(blockingReason != nil || !preflightReady || isInspecting)
                .help(blockingReason ?? (preflightReady
                    ? L10n.string("开始训练")
                    : L10n.string("必须先完成与当前配置同口径的数据检查")))
                .keyboardShortcut(.return, modifiers: .command)
                .accessibilityLabel("开始训练")
            }
        }
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
        .onChange(of: config.template) { _, _ in onTemplateChanged() }
        .onDisappear { inspectTask?.cancel() }
    }

    // MARK: - 训练前数据检查

    /// 数据/模型变化：刷新条数；≤10K 即时自动检查，更大数据等手动触发。
    private func onDataChanged() {
        cancelInspection()
        recordCount = TrainingDataPreview.countRecords(path: config.dataPath)
        preflight = nil
        preflightWasCached = false
        inspectionError = nil
        if let count = recordCount, count <= autoCheckCap {
            runInspection(debounceMs: 0)
        }
    }

    /// ctx_len 变化会改变截断结果：≤10K 防抖后重查，更大数据使旧结果失效。
    private func onCtxChanged() {
        if let count = recordCount, count <= autoCheckCap {
            runInspection(debounceMs: 400)
        } else {
            cancelInspection()
            preflight = nil
        }
    }

    private func onTemplateChanged() {
        cancelInspection()
        preflight = nil
        inspectionError = nil
        if let count = recordCount, count <= autoCheckCap {
            runInspection(debounceMs: 0)
        }
    }

    /// 起一次完整训练同源预检（debounce 用于 ctx_len 连续输入）。
    private func runInspection(debounceMs: Int) {
        cancelInspection()
        guard !config.modelPath.isEmpty, !config.dataPath.isEmpty, isModelTrainable else { return }
        let model = config.modelPath
        let data = config.dataPath
        let ctx = config.ctxLen
        let template = config.template.rawValue
        inspectTask = Task {
            if debounceMs > 0 {
                try? await Task.sleep(for: .milliseconds(debounceMs))
                if Task.isCancelled { return }
            }
            await MainActor.run { isInspecting = true; inspectionError = nil }
            let outcome = await inspector.inspect(
                modelPath: model,
                dataPath: data,
                ctxLen: ctx,
                template: template
            )
            if Task.isCancelled { return }
            await MainActor.run {
                guard config.modelPath == model,
                      config.dataPath == data,
                      config.ctxLen == ctx,
                      config.template.rawValue == template
                else { return }
                isInspecting = false
                switch outcome {
                case .success(let result, let cached):
                    preflight = result
                    preflightWasCached = cached
                case .failure(let message):
                    preflight = nil
                    inspectionError = message
                }
            }
        }
    }

    private func cancelInspection() {
        inspectTask?.cancel()
        inspectTask = nil
        inspector.cancel()
        isInspecting = false
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
        if let inspection = preflight?.inspection {
            let usable = inspection.usableCount(dropTruncated: config.dropTruncated)
            if usable <= 0 { return L10n.string("预检后没有可训练样本") }
        }
        return nil
    }

    private var preflightReady: Bool {
        guard let preflight, preflight.detection.schema != "unknown",
              let inspection = preflight.inspection,
              inspection.template == config.template.rawValue
        else { return false }
        return inspection.usableCount(dropTruncated: config.dropTruncated) > 0
    }

    private var trainingActionBar: some View {
        HStack(spacing: 10) {
            statusArea
            Spacer(minLength: 8)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
    }

    private func validateAndStart() {
        guard preflightReady else {
            inspectionError = L10n.string("必须先完成与当前配置同口径的数据检查")
            return
        }
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
        } else if let preflight, preflight.detection.schema == "unknown" {
            HStack(spacing: 8) {
                Label("无法直接训练此数据格式", systemImage: "questionmark.circle.fill")
                    .foregroundStyle(.orange)
                Button("在数据导入器中配置") { onConfigureImport(config.dataPath) }
            }
            .font(.callout)
        } else if let insp = preflight?.inspection {
            dataSummary(insp)
        } else if let count = recordCount, count > autoCheckCap {
            HStack(spacing: 8) {
                Text("约 \(count) 条 · 超过 \(autoCheckCap / 1000)K 未自动检查")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Button("运行完整检查") { runInspection(debounceMs: 0) }
                    .controlSize(.small)
            }
        } else if let error = inspectionError {
            HStack(spacing: 8) {
                Label("数据检查失败：\(error)", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Button("重试") { runInspection(debounceMs: 0) }
            }
            .font(.caption)
        } else {
            Label("必须先完成与当前配置同口径的数据检查", systemImage: "hourglass")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    /// 数据摘要:训练/验证条数 · 预计步数 · 截断处理(只警告不阻断)。
    /// 丢弃模式:有效数扣掉截断条,步数据此重算,显示丢弃数。
    /// 保留模式:完全截断橙色警告(target 前段丢失),仅部分截断黄色提示(截头保尾)。
    private func dataSummary(_ insp: DatasetInspectionResult) -> some View {
        let projection = projectedCounts(insp)
        let trainCount = projection.train
        let heldOutCount = projection.heldOut
        let steps = projection.steps
        // 严重度:丢弃模式无警告(已处理) > 完全截断(橙) > 仅部分截断(黄) > 无(绿)
        let hasFullyTruncated = !config.dropTruncated && insp.targetFullyTruncated > 0
        let partialTruncated = insp.partialTruncated
        let hasPartialOnly = !config.dropTruncated && partialTruncated > 0 && !hasFullyTruncated
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
                Text("· \(partialTruncated) 条部分截断（截头保尾）").foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .lineLimit(1)
        .truncationMode(.tail)
        .help(summaryTooltip(insp))
    }

    private func summaryTooltip(_ insp: DatasetInspectionResult) -> String {
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
    private func projectedCounts(_ insp: DatasetInspectionResult) -> (train: Int, heldOut: Int, steps: Int) {
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
                Text(config.outPath.isEmpty ? L10n.string("正在生成…") : config.outPath)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(config.outPath)
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

    // MARK: - 训练前数据预检

    @ViewBuilder
    private var dataPreviewSection: some View {
        // HIG materials §macOS:内容层使用语义目的的 Group Box/Section,
        // 不靠自定义背景块表达分组;HIG color:不靠颜色区分信息,提供文字与 glyph 替代。
        GroupBox {
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 8) {
                    Text("训练前数据检查")
                        .font(.headline)
                    if preflightWasCached, preflight != nil {
                        Label("已复用缓存", systemImage: "bolt.fill")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    if !isInspecting {
                        Button("重新检查") { runInspection(debounceMs: 0) }
                            .controlSize(.small)
                    }
                }
                .padding(.bottom, 10)

                if isInspecting {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("正在按最终模板检查全部样本…")
                            .foregroundStyle(.secondary)
                    }
                    .font(.callout)
                } else if let preflight {
                    if preflight.detection.schema == "unknown" {
                        VStack(alignment: .leading, spacing: 8) {
                            Label("无法识别为训练可直接读取的数据", systemImage: "questionmark.circle")
                                .font(.callout)
                            Text("请在数据导入器中选择字段映射并转换为标准 JSONL；完成后会自动回到这里。")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                            Button("在数据导入器中配置") { onConfigureImport(config.dataPath) }
                                .buttonStyle(.borderedProminent)
                        }
                        .padding(.top, 4)
                    } else if let inspection = preflight.inspection {
                        preflightSummary(preflight, inspection: inspection)
                        renderedPreview(preflight.preview)
                    }
                } else if let inspectionError {
                    VStack(alignment: .leading, spacing: 8) {
                        Label("检查失败", systemImage: "exclamationmark.triangle")
                            .font(.callout)
                        Text(inspectionError)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                        Button("重试") { runInspection(debounceMs: 0) }
                    }
                    .padding(.top, 4)
                } else if let count = recordCount, count > autoCheckCap {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("约 \(count) 条；为避免每次改参数都重新 tokenize，大数据需手动完成一次最终口径检查。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Button("运行完整检查") { runInspection(debounceMs: 0) }
                            .buttonStyle(.borderedProminent)
                    }
                    .padding(.top, 4)
                } else {
                    HStack(spacing: 8) {
                        Text("尚未完成检查。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        if !config.modelPath.isEmpty && !config.dataPath.isEmpty {
                            Button("运行完整检查") { runInspection(debounceMs: 0) }
                        }
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    private func preflightSummary(
        _ result: DatasetPreviewResult,
        inspection: DatasetInspectionResult
    ) -> some View {
        let partial = inspection.partialTruncated
        let projection = projectedCounts(inspection)
        return VStack(alignment: .leading, spacing: 8) {
            // 概览行:格式 + 置信度 + 模板,全 secondary 文案 + SF Symbol,无填充色。
            HStack(spacing: 6) {
                Text(result.detection.schema.uppercased())
                    .font(.subheadline.weight(.semibold))
                Text("\(result.detection.confidence.formatted(.percent.precision(.fractionLength(0)))) 置信度")
                    .foregroundStyle(.secondary)
                Text("·")
                    .foregroundStyle(.secondary)
                Text(inspection.template.uppercased())
                    .foregroundStyle(.secondary)
                Spacer()
            }
            Divider()
            // 指标行:用原生 LabeledContent(右对齐),保持 macOS Form 风格。
            // 截断数 > 0 时在值后追加文字标记,不靠颜色区分(HIG color)。
            VStack(spacing: 6) {
                preflightRow("有效样本", "\(inspection.valid) / \(inspection.total)")
                preflightRow("平均 / P95 / 最大 token",
                             "\(inspection.meanTokens.formatted(.number.precision(.fractionLength(1)))) · \(inspection.p95Tokens.formatted(.number.precision(.fractionLength(1)))) · \(inspection.maxTokens)")
                preflightRow(
                    "部分前缀截断",
                    partial > 0 ? "\(partial) 条" : "无"
                )
                preflightRow(
                    "Target 完全截断",
                    inspection.targetFullyTruncated > 0 ? "\(inspection.targetFullyTruncated) 条" : "无"
                )
                preflightRow("训练 / 验证", "\(projection.train) / \(projection.heldOut)")
                preflightRow("预计步数", "~\(projection.steps)")
            }
            // 单一状态说明行(无背景填充,只用 SF Symbol + 文字)。
            if config.dropTruncated, inspection.truncated > 0 {
                Label("将丢弃 \(inspection.truncated) 条截断样本，步数已按剩余样本重算。", systemImage: "trash")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 4)
            } else if inspection.targetFullyTruncated > 0 {
                Label(
                    "有 \(inspection.targetFullyTruncated) 条 target 完全截断；建议增加 ctx_len 或启用丢弃超长样本。",
                    systemImage: "exclamationmark.triangle"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.top, 4)
            } else if partial > 0 {
                Label("有 \(partial) 条只截去前缀头部；训练会保留 target 与 stop token。", systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 4)
            }
        }
    }

    /// 原生右对齐的「标签 → 值」行,与 macOS Form/LabeledContent 一致。
    private func preflightRow(_ title: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(L10n.string(title))
                .foregroundStyle(.secondary)
            Spacer(minLength: 12)
            Text(value)
                .font(.callout.monospacedDigit())
                .lineLimit(1)
                .truncationMode(.head)
        }
        .font(.callout)
    }

    private func renderedPreview(_ samples: [DatasetRenderedSample]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            DisclosureGroup(isExpanded: $dataExpanded) {
                VStack(spacing: 10) {
                    ForEach(Array(samples.prefix(3).enumerated()), id: \.offset) { index, sample in
                        renderedSample(sample, index: index + 1)
                    }
                }
                .padding(.top, 8)
            } label: {
                Text(L10n.format("最终模板预览（%lld 条）", min(3, samples.count)))
                    .font(.callout)
            }

            if !dataExpanded {
                EmptyView()
            }
        }
    }

    private func renderedSample(_ sample: DatasetRenderedSample, index: Int) -> some View {
        // 单条样本:原生 GroupBox 包裹,内部用 Divider 分段,不用色块背景。
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(L10n.format("样本 %lld", index))
                        .font(.callout.weight(.semibold))
                    Spacer()
                    Text(L10n.format("%lld tokens · prefix_len %lld", sample.tokenCount, sample.prefixLen))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                renderedSegment(
                    title: "输入前缀",
                    systemImage: "arrow.right.to.line.compact",
                    text: sample.prefixText
                )
                Divider()
                renderedSegment(
                    title: "训练目标",
                    systemImage: "target",
                    text: sample.targetText
                )
                Divider()
                HStack(spacing: 12) {
                    Label(
                        sample.stopTokenAppended == false ? "未追加 stop token" : "已追加 stop token",
                        systemImage: sample.stopTokenAppended == false ? "xmark.circle" : "stop.circle"
                    )
                    if sample.truncated {
                        let targetRemoved = sample.truncatedTargetTokens ?? 0
                        Label(
                            targetRemoved > 0
                                ? "截断进入 target（\(targetRemoved) tokens）"
                                : "仅截去前缀头部（\(sample.truncatedPrefixTokens ?? 0) tokens）",
                            systemImage: "scissors"
                        )
                    } else {
                        Label("未截断", systemImage: "checkmark.circle")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
        }
    }

    private func renderedSegment(
        title: String,
        systemImage: String,
        text: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(L10n.string(title), systemImage: systemImage)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(text)
                .font(.callout.monospaced())
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
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
                    value: $config.warmup, default: 50, range: 0...10000
                )
            }

            Section("训练长度") {
                TrainingIntParameterRow(
                    title: "训练轮数", key: "epochs", detail: "启用早停时为上限",
                    value: $config.epochs, default: 5, range: 1...10000
                )
                TrainingIntParameterRow(
                    title: "上下文长度", key: "ctx_len", detail: "单条样本最长 token",
                    value: $config.ctxLen, default: 512, range: 64...32768
                )
                TrainingIntParameterRow(
                    title: "日志间隔", key: "log_every", detail: "每 N 步记录一次指标",
                    value: $config.logEvery, default: 1, range: 1...1000
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
                        value: $config.earlyStopPatience, default: 3, range: 1...100
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
                    value: $config.checkpointEvery, default: 2, range: 1...1000
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
    /// 可选范围:提供时同时显示 Stepper(macOS 数值输入惯例:HIG "Stepper for
    /// bounded numerics");nil 时仅 TextField(用于 seed 这种范围不固定的字段)。
    var range: ClosedRange<Int>? = nil

    var body: some View {
        LabeledContent {
            HStack(spacing: RowLayout.spacing) {
                TextField(L10n.string(title), value: $value, format: .number)
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 88)
                if range != nil {
                    Stepper(
                        L10n.string(title),
                        value: $value,
                        in: range ?? (0...Int.max)
                    )
                    .labelsHidden()
                }
                Spacer(minLength: 0)
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
