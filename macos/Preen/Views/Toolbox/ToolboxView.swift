import SwiftUI
import UniformTypeIdentifiers

struct ToolboxView: View {
    /// 与首页 820pt grouped Form 的可见 Section 对齐（系统每侧约 58pt inset）。
    private static let formWidth: CGFloat = 820
    private static let groupedFormInset: CGFloat = 58
    private static let contentWidth = formWidth - groupedFormInset * 2

    private enum Destination: Hashable {
        case modelConversion
        case modelQuantization
        case datasetPreview
        case datasetConversion

        var title: String {
            switch self {
            case .modelConversion: return "模型转换"
            case .modelQuantization: return "模型量化"
            case .datasetPreview: return "数据集预览"
            case .datasetConversion: return "数据集转换"
            }
        }

        var subtitle: String {
            switch self {
            case .modelConversion: return "把 BlinkDL 原生 RWKV-7 权重转换为 Preen 可用模型"
            case .modelQuantization: return "把 BF16 模型量化为 int8，推理更快、内存更省近一半"
            case .datasetPreview: return "查看真实模板文本、token 长度与截断风险"
            case .datasetConversion: return "把外部数据集转换为训练可直接读取的标准 JSONL"
            }
        }
    }

    @Bindable var store: ToolboxStore
    let modelPath: String
    var onSelectModel: (String) -> Void

    @State private var path: [Destination] = []
    @State private var showingOverwriteConfirmation = false
    @State private var showingQuantizeOverwriteConfirmation = false
    @State private var advancedModelOptions = false

    var body: some View {
        NavigationStack(path: $path) {
            toolboxHome
                .navigationTitle("工具箱")
                .navigationSubtitle("选择一个工具开始，不会启动常驻推理进程")
                .navigationDestination(for: Destination.self) { destination in
                    detail(for: destination)
                        .navigationTitle(destination.title)
                        .navigationSubtitle(destination.subtitle)
                        .navigationBarBackButtonHidden(store.isRunning)
                }
        }
        .onChange(of: path) { _, _ in
            store.clearPresentationForNavigation()
        }
        .onAppear {
            if !store.isRunning { store.clearPresentationForNavigation() }
        }
        .onDisappear {
            store.clearPresentationForNavigation()
        }
        .onChange(of: modelPath) { _, newModelPath in
            guard path.last == .datasetPreview,
                  !newModelPath.isEmpty,
                  !store.datasetSourcePath.isEmpty,
                  store.datasetAnalysis == nil,
                  store.datasetState == .idle,
                  !store.isRunning
            else { return }
            store.previewDataset(modelPath: newModelPath)
        }
        .alert("工具任务失败", isPresented: Binding(
            get: { store.errorMessage != nil },
            set: { if !$0 { store.clearError() } }
        )) {
            Button("关闭") { store.clearError() }
        } message: {
            Text(store.errorMessage ?? "未知错误")
        }
        .confirmationDialog(
            "输出目录已有内容",
            isPresented: $showingOverwriteConfirmation,
            titleVisibility: .visible
        ) {
            Button("覆盖并继续", role: .destructive) {
                store.convertModel(overwrite: true)
            }
            Button("取消", role: .cancel) {}
        } message: {
            Text("将替换目录中的模型权重、配置和 tokenizer 文件；目录中的其他文件会保留。")
        }
        .confirmationDialog(
            "输出目录已有内容",
            isPresented: $showingQuantizeOverwriteConfirmation,
            titleVisibility: .visible
        ) {
            Button("覆盖并继续", role: .destructive) {
                store.quantizeModel(overwrite: true)
            }
            Button("取消", role: .cancel) {}
        } message: {
            Text("将替换目录中的量化模型权重、配置和 tokenizer 文件；目录中的其他文件会保留。")
        }
    }

    @ViewBuilder
    private func detail(for destination: Destination) -> some View {
        switch destination {
        case .modelConversion:
            modelConversionView
        case .modelQuantization:
            modelQuantizationView
        case .datasetPreview:
            datasetPreviewView
        case .datasetConversion:
            datasetConversionView
        }
    }

    // MARK: - 工具列表首页(系统设置风格)

    private var toolboxHome: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Form {
                    Section {
                        toolRow(
                            destination: .modelConversion,
                            title: "模型转换",
                            description: "将原生 RWKV-7 .pth 转为 HF safetensors 模型目录。",
                            icon: "shippingbox.and.arrow.backward"
                        )

                        toolRow(
                            destination: .modelQuantization,
                            title: "模型量化",
                            description: "把 BF16 模型转为 int8，推理更快、内存更省近一半。",
                            icon: "speedometer"
                        )

                        toolRow(
                            destination: .datasetPreview,
                            title: "数据集预览",
                            description: "探测格式，查看最终训练文本和 token 截断情况。",
                            icon: "doc.text.magnifyingglass"
                        )

                        toolRow(
                            destination: .datasetConversion,
                            title: "数据集转换",
                            description: "把 Alpaca、ShareGPT、ChatML 或裸 QA 转成标准 JSONL。",
                            icon: "arrow.triangle.2.circlepath.doc.on.clipboard"
                        )
                    }
                }
                .formStyle(.grouped)
            }
            .padding(.vertical, 20)
            .frame(maxWidth: 820, alignment: .center)
            .frame(maxWidth: .infinity, alignment: .top)
        }
    }

    private func toolRow(
        destination: Destination,
        title: String,
        description: String,
        icon: String
    ) -> some View {
        NavigationLink(value: destination) {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(width: 28, height: 28)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                    Text(description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - 模型转换

    private var modelConversionView: some View {
        VStack(spacing: 0) {
            toolScroll {
                surface {
                    toolPathRow("原生模型", detail: ".pth", path: store.modelSourcePath,
                                acceptsDrop: true,
                                onDropPath: { url in
                            selectModelSource(url)
                        }) {
                        if let url = pickFile(allowedContentTypes: [Self.pthContentType]) {
                            selectModelSource(url)
                        }
                    }

                    Divider()

                    toolPathRow("输出目录", detail: nil, path: store.modelOutputPath) {
                        if let url = pickSave(defaultName: modelDefaultName) {
                            store.modelOutputPath = url.path
                        }
                    }

                    Divider()

                    VStack(alignment: .leading, spacing: 0) {
                        Button {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                advancedModelOptions.toggle()
                            }
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "chevron.right")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.tertiary)
                                    .rotationEffect(.degrees(advancedModelOptions ? 90 : 0))
                                Text("高级选项")
                                Spacer(minLength: 0)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)

                        if advancedModelOptions {
                            HStack {
                                Text("权重精度")
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Picker("权重精度", selection: $store.modelPrecision) {
                                    Text("BF16（推荐）").tag("bf16")
                                    Text("FP16").tag("fp16")
                                    Text("FP32").tag("fp32")
                                }
                                .labelsHidden()
                                .frame(width: 180)
                                .disabled(store.isRunning)
                            }
                            .padding(.top, 10)

                            Text("仅决定转换后模型权重格式。State 训练仍保持 fp32 累加。")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                jobStatus(for: "model")

                if let result = store.modelResult {
                    successSurface(title: "转换完成") {
                        LabeledContent("模型目录", value: result.outputPath)
                        LabeledContent("权重", value: "\(result.tensorCount) 个张量 · \(result.precision.uppercased())")
                        if let layers = result.numHiddenLayers {
                            LabeledContent("结构", value: "\(layers) 层 · hidden \(result.hiddenSize ?? 0)")
                        }
                        HStack {
                            Button("设为当前模型") { onSelectModel(result.outputPath) }
                                .buttonStyle(.borderedProminent)
                            Button("在 Finder 中显示") {
                                reveal(path: result.outputPath)
                            }
                        }
                        .padding(.top, 4)
                    }
                }
            }

            modelConversionFooter
        }
    }

    private var modelConversionFooter: some View {
        HStack {
            if store.isRunning {
                Button("取消") { store.cancel() }
                    .buttonStyle(.bordered)
            }
            Spacer()
            Button {
                if store.modelOutputRequiresConfirmation {
                    showingOverwriteConfirmation = true
                } else {
                    store.convertModel()
                }
            } label: {
                Label("开始转换", systemImage: "arrow.right.circle.fill")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!store.canConvertModel)
        }
        .frame(maxWidth: Self.contentWidth)
        .padding(.horizontal, Self.groupedFormInset)
        .padding(.bottom, 24)
        .frame(maxWidth: .infinity)
    }

    // MARK: - 模型量化

    private var modelQuantizationView: some View {
        toolScroll {
            surface {
                toolPathRow("源模型", detail: "BF16 目录", path: store.quantizeSourcePath,
                            acceptsDrop: true,
                            onDropPath: { url in
                        selectQuantizeSource(url)
                    }) {
                    if let url = pickDirectory() {
                        selectQuantizeSource(url)
                    }
                }

                Divider()

                toolPathRow("输出目录", detail: nil, path: store.quantizeOutputPath) {
                    if let url = pickSave(defaultName: quantizeDefaultName) {
                        store.quantizeOutputPath = url.path
                    }
                }
            }

            Text("int8 量化仅用于推理加速（实测 decode 提速约 1.7 倍、内存减半）。量化模型不可训练——state tuning 需要 BF16 权重。")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            HStack {
                if store.isRunning {
                    Button("取消") { store.cancel() }
                        .buttonStyle(.bordered)
                }
                Spacer()
                Button {
                    if store.quantizeOutputRequiresConfirmation {
                        showingQuantizeOverwriteConfirmation = true
                    } else {
                        store.quantizeModel()
                    }
                } label: {
                    Label("开始量化", systemImage: "speedometer")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!store.canQuantize)
            }

            jobStatus(for: "quantize")

            if let result = store.quantizeResult {
                successSurface(title: "量化完成") {
                    LabeledContent("模型目录", value: result.out)
                    LabeledContent("量化", value: "int\(result.bits) · \(result.quantizedLayers) 个量化层")
                    if let elapsed = result.elapsed {
                        LabeledContent("耗时", value: String(format: "%.1f 秒", elapsed))
                    }
                    HStack {
                        Button("设为当前模型") { onSelectModel(result.out) }
                            .buttonStyle(.borderedProminent)
                        Button("在 Finder 中显示") {
                            reveal(path: result.out)
                        }
                    }
                    .padding(.top, 4)
                }
            }
        }
    }

    // MARK: - 数据集预览

    private var datasetPreviewView: some View {
        toolScroll {
            surface {
                datasetSourceRow
                Divider()
                HStack {
                    Label(
                        modelPath.isEmpty
                            ? "请先在窗口右上角选择模型"
                            : URL(fileURLWithPath: modelPath).lastPathComponent,
                        systemImage: "textformat.abc"
                    )
                    .foregroundStyle(modelPath.isEmpty ? .orange : .secondary)
                    Spacer()
                    Picker("多轮", selection: $store.datasetTurnPolicy) {
                        Text("只取首轮").tag("first")
                        Text("拆分全部轮次").tag("all")
                    }
                    .frame(width: 220)
                    .disabled(store.isRunning)
                    .onChange(of: store.datasetTurnPolicy) { _, _ in
                        store.invalidateDatasetAnalysis()
                    }
                    Stepper(
                        "ctx \(store.datasetContextLength)",
                        value: $store.datasetContextLength,
                        in: 32...8192,
                        step: 32
                    )
                    .disabled(store.isRunning)
                    .onChange(of: store.datasetContextLength) { _, _ in
                        store.invalidateDatasetAnalysis()
                    }
                }
            }

            HStack {
                if store.isRunning && store.presentationTool == "dataset" {
                    Button("取消检查") { store.cancel() }
                        .buttonStyle(.bordered)
                }
                if store.datasetNeedsRefresh {
                    Label("设置已更改，需要重新检查", systemImage: "arrow.clockwise")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    store.previewDataset(modelPath: modelPath)
                } label: {
                    Label(
                        store.datasetAnalysis == nil && !store.datasetNeedsRefresh
                            ? "检查数据集" : "重新检查",
                        systemImage: "doc.text.magnifyingglass"
                    )
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!store.canPreviewDataset || modelPath.isEmpty)
            }

            jobStatus(for: "dataset")

            if let analysis = store.datasetAnalysis {
                if analysis.detection.schema == "unknown" {
                    manualMappingSurface(analysis)
                } else {
                    analysisSummary(analysis)
                    samplePreview(store.datasetPreviewSamples)
                }
            }
        }
    }

    private var datasetSourceRow: some View {
        toolPathRow("源数据", detail: "JSON / JSONL / CSV", path: store.datasetSourcePath,
                    acceptsDrop: true,
                    onDropPath: { url in
                selectDatasetSource(url)
            }) {
            if let url = pickFile() {
                selectDatasetSource(url)
            }
        }
    }

    private func selectDatasetSource(_ url: URL) {
        store.selectDatasetSource(path: url.path)
        store.datasetOutputPath = PythonResolver.datasetsDirectory
            .appendingPathComponent(url.deletingPathExtension().lastPathComponent + ".standard.jsonl")
            .path
        if path.last == .datasetPreview, !modelPath.isEmpty {
            store.previewDataset(modelPath: modelPath)
        }
    }

    private func selectModelSource(_ url: URL) {
        guard store.selectModelSource(path: url.path) else { return }
        store.modelOutputPath = PythonResolver.modelsDirectory
            .appendingPathComponent(url.deletingPathExtension().lastPathComponent)
            .path
    }

    private func selectQuantizeSource(_ url: URL) {
        guard store.selectQuantizeSource(path: url.path) else { return }
        store.quantizeOutputPath = PythonResolver.modelsDirectory
            .appendingPathComponent(url.lastPathComponent + "-int8")
            .path
    }

    private func analysisSummary(_ analysis: DatasetPreviewResult) -> some View {
        surface {
            HStack(spacing: 10) {
                Label(analysis.detection.schema, systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .font(.headline)
                Text(analysis.detection.confidence.formatted(.percent.precision(.fractionLength(0))))
                    .foregroundStyle(.secondary)
                if let result = analysis.result {
                    Text("·") .foregroundStyle(.tertiary)
                    Text("\(result.recordCount) 条 · \(result.template)")
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            if let inspection = analysis.inspection {
                Divider()
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 130), alignment: .leading)],
                    alignment: .leading,
                    spacing: 14
                ) {
                    metric("有效样本", "\(inspection.valid)/\(inspection.total)")
                    metric("平均 token", inspection.meanTokens.formatted(.number.precision(.fractionLength(1))))
                    metric("P95", inspection.p95Tokens.formatted(.number.precision(.fractionLength(1))))
                    metric("最大 token", "\(inspection.maxTokens)")
                    metric("将被截断", "\(inspection.truncated)", warning: inspection.truncated > 0)
                    metric(
                        "Target 全丢失",
                        "\(inspection.targetFullyTruncated)",
                        warning: inspection.targetFullyTruncated > 0
                    )
                }
            }
        }
    }

    private func manualMappingSurface(_ analysis: DatasetPreviewResult) -> some View {
        surface {
            Label("无法自动识别字段", systemImage: "questionmark.circle")
                .font(.headline)
            Text("可用字段：\(analysis.availableKeys.joined(separator: " · "))")
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
            HStack {
                TextField("Prompt 字段", text: $store.manualPromptKey)
                TextField("Response 字段", text: $store.manualResponseKey)
                Button("重新检查") {
                    store.previewDataset(modelPath: modelPath)
                }
                .disabled(store.manualPromptKey.isEmpty || store.manualResponseKey.isEmpty)
            }
        }
    }

    private func samplePreview(_ samples: [DatasetRenderedSample]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("完整预览")
                        .font(.headline)
                    Text(datasetPreviewRangeText)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            .id("dataset-preview-page-top")

            if store.datasetPreviewPageCount > 1 {
                datasetPreviewPagination
            }

            ForEach(Array(samples.enumerated()), id: \.offset) { index, sample in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("样本 \(datasetPreviewGlobalIndex(index))").font(.caption.bold())
                        Spacer()
                        Text("\(sample.tokenCount) tokens")
                            .font(.caption.monospacedDigit())
                        if sample.truncated {
                            Label("会截断", systemImage: "exclamationmark.triangle.fill")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                    }

                    datasetPreviewTextSection(
                        title: "输入前缀",
                        text: sample.prefixText
                    )

                    Divider()

                    datasetPreviewTextSection(
                        title: "训练目标",
                        text: sample.targetText
                    )
                }
                .font(.body.monospaced())
                .padding(14)
                .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
            }

            if store.datasetPreviewPageCount > 1 {
                datasetPreviewPagination
                    .padding(.top, 2)
            }
        }
    }

    private func datasetPreviewTextSection(title: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)

            Text(text)
                .foregroundStyle(.primary)
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(title)
        .accessibilityValue(text)
    }

    private var datasetPreviewPagination: some View {
        HStack(spacing: 8) {
            Button {
                store.loadDatasetPreviewPage(1)
            } label: {
                Image(systemName: "backward.end.fill")
            }
            .help("第一页")
            .disabled(store.isRunning || store.datasetPreviewPage <= 1)

            Button {
                store.loadDatasetPreviewPage(store.datasetPreviewPage - 1)
            } label: {
                Image(systemName: "chevron.left")
            }
            .help("上一页")
            .disabled(store.isRunning || store.datasetPreviewPage <= 1)

            Text("第 \(store.datasetPreviewPage) / \(store.datasetPreviewPageCount) 页")
                .font(.caption.monospacedDigit())
                .frame(minWidth: 92)

            Button {
                store.loadDatasetPreviewPage(store.datasetPreviewPage + 1)
            } label: {
                Image(systemName: "chevron.right")
            }
            .help("下一页")
            .disabled(store.isRunning || store.datasetPreviewPage >= store.datasetPreviewPageCount)

            Button {
                store.loadDatasetPreviewPage(store.datasetPreviewPageCount)
            } label: {
                Image(systemName: "forward.end.fill")
            }
            .help("最后一页")
            .disabled(store.isRunning || store.datasetPreviewPage >= store.datasetPreviewPageCount)

            Spacer()
            Text("每页 \(store.datasetPreviewPageSize) 条")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.bordered)
    }

    private var datasetPreviewRangeText: String {
        guard store.datasetPreviewTotal > 0 else { return "没有可预览样本" }
        let first = (store.datasetPreviewPage - 1) * store.datasetPreviewPageSize + 1
        let last = min(
            first + store.datasetPreviewSamples.count - 1,
            store.datasetPreviewTotal
        )
        return "显示 \(first)–\(last)，共 \(store.datasetPreviewTotal) 条"
    }

    private func datasetPreviewGlobalIndex(_ localIndex: Int) -> Int {
        (store.datasetPreviewPage - 1) * store.datasetPreviewPageSize + localIndex + 1
    }

    // MARK: - 数据集转换

    private var datasetConversionView: some View {
        toolScroll {
            surface {
                datasetSourceRow
                Divider()
                toolPathRow("输出文件", detail: ".jsonl", path: store.datasetOutputPath) {
                    if let url = pickSave(defaultName: datasetDefaultName) {
                        store.datasetOutputPath = url.path
                    }
                }
                Divider()
                HStack {
                    Picker("多轮数据", selection: $store.datasetTurnPolicy) {
                        Text("每条只保留首轮").tag("first")
                        Text("每轮拆成独立样本").tag("all")
                    }
                    .frame(width: 300)
                    .disabled(store.isRunning)
                    Spacer()
                    Text("自动识别 Alpaca / ShareGPT / ChatML / 裸 QA")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if !store.manualPromptKey.isEmpty && !store.manualResponseKey.isEmpty {
                    Label(
                        "使用字段映射：\(store.manualPromptKey) → \(store.manualResponseKey)",
                        systemImage: "arrow.left.arrow.right"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }

            HStack {
                if store.isRunning {
                    Button("取消") { store.cancel() }
                        .buttonStyle(.bordered)
                }
                Spacer()
                Button {
                    store.importDataset()
                } label: {
                    Label("转换并保存", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!store.canImportDataset)
            }

            jobStatus(for: "dataset")

            if let path = store.importedDatasetPath {
                successSurface(title: "转换完成") {
                    LabeledContent("标准数据集", value: path)
                    Text("同目录已生成 .import.json；训练时会自动选择正确的数据 loader。")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button("在 Finder 中显示") { reveal(path: path) }
                        .padding(.top, 4)
                }
            }
        }
    }

    // MARK: - 共用组件

    private func toolScroll<Content: View>(
        @ViewBuilder content: @escaping () -> Content
    ) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    content()
                }
                .frame(maxWidth: Self.contentWidth, alignment: .leading)
                .padding(.horizontal, Self.groupedFormInset)
                .padding(.top, 39)
                .padding(.bottom, 24)
                .frame(maxWidth: .infinity, alignment: .top)
            }
            .onChange(of: store.datasetPreviewPage) { _, _ in
                proxy.scrollTo("dataset-preview-page-top", anchor: .top)
            }
        }
    }

    private func surface<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            content()
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.38), in: RoundedRectangle(cornerRadius: 12))
    }

    private func successSurface<Content: View>(
        title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        surface {
            Label(title, systemImage: "checkmark.circle.fill")
                .font(.headline)
                .foregroundStyle(.green)
            content()
        }
    }

    private func toolPathRow(
        _ title: String,
        detail: String?,
        path: String,
        acceptsDrop: Bool = false,
        onDropPath: ((URL) -> Void)? = nil,
        action: @escaping () -> Void
    ) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                if let detail {
                    Text(detail)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(width: 130, alignment: .leading)
            Text(path.isEmpty ? "未选择" : path)
                .lineLimit(1)
                .truncationMode(.middle)
                .foregroundStyle(path.isEmpty ? .secondary : .primary)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button("选择…", action: action)
                .disabled(store.isRunning)
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            guard acceptsDrop, let onDropPath else { return false }
            return Self.handleFileDrop(providers, handler: onDropPath)
        }
    }

    /// 从拖拽数据提取第一个文件 URL。
    private static func handleFileDrop(_ providers: [NSItemProvider], handler: @escaping (URL) -> Void) -> Bool {
        guard let provider = providers.first else { return false }
        provider.loadItem(forTypeIdentifier: "public.file-url", options: nil) { item, _ in
            guard let data = item as? Data,
                  let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
            DispatchQueue.main.async { handler(url) }
        }
        return true
    }

    @ViewBuilder
    private func jobStatus(for tool: String) -> some View {
        if store.presentationTool == tool {
            if store.isRunning {
                VStack(alignment: .leading, spacing: 7) {
                    if let progress = store.progress {
                        ProgressView(value: progress)
                            .progressViewStyle(.linear)
                            .accessibilityLabel(store.statusMessage)
                            .accessibilityValue(progressAccessibilityValue)
                    } else {
                        ProgressView()
                            .progressViewStyle(.linear)
                            .accessibilityLabel(store.statusMessage)
                            .accessibilityValue("正在准备")
                    }
                    HStack {
                        Text(store.statusMessage)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        if let current = store.progressCurrent,
                           let total = store.progressTotal,
                           total > 0 {
                            Text("\(current.formatted()) / \(total.formatted())")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            ForEach(store.warnings, id: \.self) { warning in
                Label(warning, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    private var progressAccessibilityValue: String {
        guard let current = store.progressCurrent,
              let total = store.progressTotal,
              total > 0
        else {
            return store.progress.map { $0.formatted(.percent.precision(.fractionLength(0))) } ?? ""
        }
        return "\(current) / \(total)"
    }

    private func metric(_ title: String, _ value: String, warning: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.headline.monospacedDigit())
                .foregroundStyle(warning ? .orange : .primary)
        }
    }

    private var modelDefaultName: String {
        guard !store.modelSourcePath.isEmpty else { return "rwkv7-converted" }
        return URL(fileURLWithPath: store.modelSourcePath)
            .deletingPathExtension()
            .lastPathComponent
    }

    private var quantizeDefaultName: String {
        guard !store.quantizeSourcePath.isEmpty else { return "rwkv7-int8" }
        return URL(fileURLWithPath: store.quantizeSourcePath)
            .lastPathComponent + "-int8"
    }

    private var datasetDefaultName: String {
        guard !store.datasetSourcePath.isEmpty else { return "dataset.standard.jsonl" }
        return URL(fileURLWithPath: store.datasetSourcePath)
            .deletingPathExtension()
            .lastPathComponent + ".standard.jsonl"
    }

    private static let pthContentType = UTType(filenameExtension: "pth")!

    private func pickFile(allowedContentTypes: [UTType] = []) -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if !allowedContentTypes.isEmpty {
            panel.allowedContentTypes = allowedContentTypes
        }
        return panel.runModal() == .OK ? panel.url : nil
    }

    /// 选择目录(量化源是模型目录,不是文件)。参考 ContentView.pickModel 范式。
    private func pickDirectory() -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        return panel.runModal() == .OK ? panel.url : nil
    }

    private func pickSave(defaultName: String) -> URL? {
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = defaultName
        return panel.runModal() == .OK ? panel.url : nil
    }

    private func reveal(path: String) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }
}
