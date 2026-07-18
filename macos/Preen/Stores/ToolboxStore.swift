import Foundation
import Observation

enum ToolJobState: Equatable {
    case idle, running, completed, failed, cancelled
}

struct ModelConversionResult: Decodable, Equatable {
    let outputPath: String
    let tensorCount: Int
    let precision: String
    let numHiddenLayers: Int?
    let hiddenSize: Int?
    let vocabSize: Int?
    let elapsed: Double?

    enum CodingKeys: String, CodingKey {
        case outputPath = "output_path"
        case tensorCount = "tensor_count"
        case precision
        case numHiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case vocabSize = "vocab_size"
        case elapsed
    }
}

struct QuantizationResult: Decodable, Equatable {
    let bits: Int
    let groupSize: Int
    let quantizedLayers: Int
    let src: String
    let out: String
    let elapsed: Double?

    enum CodingKeys: String, CodingKey {
        case bits, src, out, elapsed
        case groupSize = "group_size"
        case quantizedLayers = "quantized_layers"
    }
}

struct DatasetDetectionResult: Codable, Equatable {
    let schema: String
    let promptKeys: [String]
    let responseKeys: [String]
    let confidence: Double
    let totalSampled: Int

    enum CodingKeys: String, CodingKey {
        case schema, confidence
        case promptKeys = "prompt_keys"
        case responseKeys = "response_keys"
        case totalSampled = "total_sampled"
    }
}

struct DatasetConversionSummary: Codable, Equatable {
    let template: String
    let turnPolicy: String
    let droppedSystem: Int
    let droppedOther: Int
    let qaDegradationHint: Bool
    let recordCount: Int

    enum CodingKeys: String, CodingKey {
        case template
        case turnPolicy = "turn_policy"
        case droppedSystem = "dropped_system"
        case droppedOther = "dropped_other"
        case qaDegradationHint = "qa_degradation_hint"
        case recordCount = "record_count"
    }
}

struct DatasetInspectionResult: Codable, Equatable {
    let total: Int
    let valid: Int
    let truncated: Int
    let targetFullyTruncated: Int
    let minTokens: Int
    let meanTokens: Double
    let p95Tokens: Double
    let maxTokens: Int
    let ctxLen: Int
    let template: String

    enum CodingKeys: String, CodingKey {
        case total, valid, truncated, template
        case targetFullyTruncated = "target_fully_truncated"
        case minTokens = "min_tokens"
        case meanTokens = "mean_tokens"
        case p95Tokens = "p95_tokens"
        case maxTokens = "max_tokens"
        case ctxLen = "ctx_len"
    }
}

extension DatasetInspectionResult {
    /// `truncated` includes fully-truncated targets in the backend aggregate.
    /// Keep the two user-facing severities mutually exclusive.
    var partialTruncated: Int {
        max(0, truncated - targetFullyTruncated)
    }

    func usableCount(dropTruncated: Bool) -> Int {
        max(0, valid - (dropTruncated ? truncated : 0))
    }
}

struct DatasetRenderedSample: Codable, Equatable, Identifiable {
    var id: String { "\(prefixLen)-\(promptText)-\(responseText)" }
    let fullText: String
    let prefixText: String
    let targetText: String
    let prefixLen: Int
    let tokenCount: Int
    let promptText: String
    let responseText: String
    let truncated: Bool
    let stopTokenAppended: Bool?
    let truncatedPrefixTokens: Int?
    let truncatedTargetTokens: Int?

    enum CodingKeys: String, CodingKey {
        case fullText = "full_text"
        case prefixText = "prefix_text"
        case targetText = "target_text"
        case prefixLen = "prefix_len"
        case tokenCount = "token_count"
        case promptText = "prompt_text"
        case responseText = "response_text"
        case truncated
        case stopTokenAppended = "stop_token_appended"
        case truncatedPrefixTokens = "truncated_prefix_tokens"
        case truncatedTargetTokens = "truncated_target_tokens"
    }
}

struct DatasetPreviewResult: Codable, Equatable {
    let detection: DatasetDetectionResult
    let result: DatasetConversionSummary?
    let preview: [DatasetRenderedSample]
    let inspection: DatasetInspectionResult?
    let pagination: DatasetPreviewPagination?
    let availableKeys: [String]
    let turnPolicy: String

    enum CodingKeys: String, CodingKey {
        case detection, result, preview, inspection, pagination
        case availableKeys = "available_keys"
        case turnPolicy = "turn_policy"
    }
}

struct DatasetPreviewPagination: Codable, Equatable {
    let cachePath: String
    let total: Int
    let pageSize: Int
    let pageCount: Int

    enum CodingKeys: String, CodingKey {
        case cachePath = "cache_path"
        case total
        case pageSize = "page_size"
        case pageCount = "page_count"
    }
}

struct DatasetPreviewPageResult: Codable, Equatable {
    let preview: [DatasetRenderedSample]
    let page: Int
    let pageSize: Int
    let pageCount: Int
    let total: Int

    enum CodingKeys: String, CodingKey {
        case preview, page, total
        case pageSize = "page_size"
        case pageCount = "page_count"
    }
}

@Observable
@MainActor
final class ToolboxStore {
    private(set) var modelSourcePath = ""
    var modelOutputPath = ""
    var modelPrecision = "bf16"
    private(set) var modelState: ToolJobState = .idle
    private(set) var modelResult: ModelConversionResult?

    var quantizeSourcePath = ""
    var quantizeOutputPath = ""
    private(set) var quantizeState: ToolJobState = .idle
    private(set) var quantizeResult: QuantizationResult?

    var datasetSourcePath = ""
    var datasetOutputPath = ""
    var datasetTurnPolicy = "first"
    var datasetContextLength = 512
    var manualPromptKey = ""
    var manualResponseKey = ""
    private(set) var datasetState: ToolJobState = .idle
    private(set) var datasetAnalysis: DatasetPreviewResult?
    private(set) var importedDatasetPath: String?
    private(set) var datasetPreviewSamples: [DatasetRenderedSample] = []
    private(set) var datasetPreviewPage = 1
    private(set) var datasetPreviewPageSize = 20
    private(set) var datasetPreviewPageCount = 0
    private(set) var datasetPreviewTotal = 0

    private(set) var progress: Double?
    private(set) var progressCurrent: Int?
    private(set) var progressTotal: Int?
    private(set) var statusMessage = ""
    private(set) var warnings: [String] = []
    private(set) var errorMessage: String?
    private(set) var presentationTool: String?
    private(set) var datasetNeedsRefresh = false

    /// 深链请求:外部(欢迎窗口 / 训练空态)要求打开某个工具页。
    /// ToolboxView 消费后清空,把对应工具推入导航栈。取值对齐 Destination 的 rawValue。
    var pendingTool: String?

    private var runner: ToolJobRunner?
    private var datasetPreviewCachePath: String?
    private var pendingDatasetCacheKey: DatasetPreflightCacheKey?

    var isRunning: Bool { runner?.isRunning == true }
    var canConvertModel: Bool {
        Self.modelSourceValidationError(for: modelSourcePath) == nil
            && !modelOutputPath.isEmpty
            && !isRunning
    }
    var modelOutputRequiresConfirmation: Bool {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: modelOutputPath, isDirectory: &isDirectory
        ), isDirectory.boolValue else { return false }
        let contents = try? FileManager.default.contentsOfDirectory(atPath: modelOutputPath)
        return contents?.isEmpty == false
    }
    var canQuantize: Bool {
        Self.quantizeSourceValidationError(for: quantizeSourcePath) == nil
            && !quantizeOutputPath.isEmpty
            && !isRunning
    }
    var quantizeOutputRequiresConfirmation: Bool {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: quantizeOutputPath, isDirectory: &isDirectory
        ), isDirectory.boolValue else { return false }
        let contents = try? FileManager.default.contentsOfDirectory(atPath: quantizeOutputPath)
        return contents?.isEmpty == false
    }
    var canPreviewDataset: Bool {
        !datasetSourcePath.isEmpty && !isRunning
    }
    var canImportDataset: Bool {
        !datasetSourcePath.isEmpty && !datasetOutputPath.isEmpty && !isRunning
    }

    func convertModel(overwrite: Bool = false) {
        if let validationError = Self.modelSourceValidationError(for: modelSourcePath) {
            errorMessage = validationError
            return
        }
        guard canConvertModel else { return }
        begin(tool: "model")
        modelState = .running
        var argv = [
            "-m", "statetuner.cli", "convert-model",
            "--rwkv7", modelSourcePath,
            "--out", modelOutputPath,
            "--precision", modelPrecision,
        ]
        if overwrite {
            argv.append("--overwrite")
        }
        consume(runner: runner!, stream: runner!.start(argv: argv, currentDirectory: PythonResolver.repoRoot))
    }

    func quantizeModel(overwrite: Bool = false) {
        if let validationError = Self.quantizeSourceValidationError(for: quantizeSourcePath) {
            errorMessage = validationError
            return
        }
        guard canQuantize else { return }
        begin(tool: "quantize")
        quantizeState = .running
        var argv = [
            "-m", "statetuner.cli", "quantize",
            "--model", quantizeSourcePath,
            "--out", quantizeOutputPath,
        ]
        if overwrite {
            argv.append("--overwrite")
        }
        consume(runner: runner!, stream: runner!.start(argv: argv, currentDirectory: PythonResolver.repoRoot))
    }

    func previewDataset(modelPath: String) {
        guard canPreviewDataset, !modelPath.isEmpty else {
            errorMessage = L10n.string("请先在窗口顶部选择模型，数据预览需要对应 tokenizer")
            return
        }
        removeDatasetPreviewCache()
        datasetAnalysis = nil
        resetDatasetPreviewPage()
        let cacheKey = DatasetPreflightCache.makeKey(
            modelPath: modelPath,
            dataPath: datasetSourcePath,
            ctxLen: datasetContextLength,
            template: "auto",
            turnPolicy: datasetTurnPolicy,
            promptKey: manualPromptKey,
            responseKey: manualResponseKey,
            trainingDataRoute: false
        )
        if let cached = DatasetPreflightCache.load(cacheKey) {
            errorMessage = nil
            warnings = []
            datasetNeedsRefresh = false
            presentationTool = "dataset"
            applyDatasetAnalysis(cached)
            datasetState = .completed
            statusMessage = L10n.string("已复用相同配置的数据检查缓存")
            return
        }
        let cachePath = cacheKey.previewURL.path
        pendingDatasetCacheKey = cacheKey
        datasetPreviewCachePath = cachePath
        begin(tool: "dataset")
        datasetState = .running
        datasetNeedsRefresh = false
        statusMessage = L10n.string("正在启动检查进程…")
        var argv = [
            "-m", "statetuner.cli", "dataset-preview",
            "--model", modelPath,
            "--data", datasetSourcePath,
            "--ctx-len", String(datasetContextLength),
            "--turn-policy", datasetTurnPolicy,
            "--template", "auto",
            "--cache-out", cachePath,
            "--page-size", String(datasetPreviewPageSize),
        ]
        appendManualMapping(to: &argv)
        consume(runner: runner!, stream: runner!.start(argv: argv, currentDirectory: PythonResolver.repoRoot))
    }

    func loadDatasetPreviewPage(_ page: Int) {
        guard
            !isRunning,
            page != datasetPreviewPage,
            page >= 1,
            page <= datasetPreviewPageCount,
            let cachePath = datasetPreviewCachePath
        else { return }

        // 翻页通常只需约 0.2 秒，不占用完整检查的进度展示区域，
        // 避免进度条瞬间出现/消失导致内容上下跳动。
        begin(tool: "dataset-page")
        datasetState = .running
        statusMessage = L10n.format("加载第 %lld 页", page)
        let argv = [
            "-m", "statetuner.cli", "dataset-preview-page",
            "--cache", cachePath,
            "--page", String(page),
            "--page-size", String(datasetPreviewPageSize),
        ]
        consume(runner: runner!, stream: runner!.start(argv: argv, currentDirectory: PythonResolver.repoRoot))
    }

    func importDataset() {
        guard canImportDataset else { return }
        begin(tool: "dataset")
        datasetState = .running
        var argv = [
            "-m", "statetuner.cli", "import",
            "--data", datasetSourcePath,
            "--out", datasetOutputPath,
            "--turn-policy", datasetTurnPolicy,
            "--events",
        ]
        appendManualMapping(to: &argv)
        consume(runner: runner!, stream: runner!.start(argv: argv, currentDirectory: PythonResolver.repoRoot))
    }

    func cancel() {
        runner?.cancel()
        statusMessage = L10n.string("正在取消…")
    }

    func clearError() { errorMessage = nil }

    /// 统一校验文件选择器与拖拽入口，避免把无效路径留到 Python 进程才报错。
    @discardableResult
    func selectModelSource(path: String) -> Bool {
        guard !isRunning else { return false }
        if let validationError = Self.modelSourceValidationError(for: path) {
            errorMessage = validationError
            return false
        }
        modelSourcePath = path
        modelState = .idle
        modelResult = nil
        errorMessage = nil
        return true
    }

    /// 量化源目录选择校验(目录 + config.json)。与 selectModelSource 同构。
    @discardableResult
    func selectQuantizeSource(path: String) -> Bool {
        guard !isRunning else { return false }
        if let validationError = Self.quantizeSourceValidationError(for: path) {
            errorMessage = validationError
            return false
        }
        quantizeSourcePath = path
        quantizeState = .idle
        quantizeResult = nil
        errorMessage = nil
        return true
    }

    /// 工具详情页切换时清理瞬时展示；输入和已完成产物保留。
    func clearPresentationForNavigation() {
        progress = nil
        progressCurrent = nil
        progressTotal = nil
        statusMessage = ""
        warnings = []
        errorMessage = nil
        presentationTool = nil
    }

    func selectDatasetSource(path: String) {
        guard !isRunning else { return }
        removeDatasetPreviewCache()
        datasetSourcePath = path
        datasetAnalysis = nil
        importedDatasetPath = nil
        resetDatasetPreviewPage()
        manualPromptKey = ""
        manualResponseKey = ""
        datasetState = .idle
        datasetNeedsRefresh = false
        clearPresentationForNavigation()
    }

    func invalidateDatasetAnalysis() {
        guard !isRunning else { return }
        datasetNeedsRefresh = datasetAnalysis != nil || datasetState == .completed
        removeDatasetPreviewCache()
        datasetAnalysis = nil
        importedDatasetPath = nil
        resetDatasetPreviewPage()
        clearPresentationForNavigation()
    }

    private func begin(tool: String) {
        runner = ToolJobRunner()
        progress = nil
        progressCurrent = nil
        progressTotal = nil
        statusMessage = L10n.string("准备中")
        warnings = []
        errorMessage = nil
        presentationTool = tool
        if tool == "model" { modelResult = nil }
        if tool == "quantize" { quantizeResult = nil }
        if tool == "dataset" { importedDatasetPath = nil }
    }

    private static func modelSourceValidationError(for path: String) -> String? {
        guard !path.isEmpty else { return L10n.string("请选择原生 RWKV-7 .pth 模型") }
        let url = URL(fileURLWithPath: path)
        guard url.pathExtension.lowercased() == "pth" else {
            return L10n.string("源模型必须是 .pth 文件")
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
              !isDirectory.boolValue else {
            return L10n.string("找不到所选的 .pth 模型文件")
        }
        return nil
    }

    /// 量化源校验:必须是含 config.json 的模型目录(转换产物),而非 .pth 文件。
    private static func quantizeSourceValidationError(for path: String) -> String? {
        guard !path.isEmpty else { return L10n.string("请选择源 BF16 模型目录") }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return L10n.string("源路径必须是模型目录（转换后的 HF 目录）")
        }
        guard FileManager.default.fileExists(atPath: "\(path)/config.json") else {
            return L10n.string("源目录缺少 config.json，不是有效的模型目录")
        }
        return nil
    }

    private func appendManualMapping(to argv: inout [String]) {
        guard !manualPromptKey.isEmpty, !manualResponseKey.isEmpty else { return }
        argv.append(contentsOf: [
            "--prompt-key", manualPromptKey,
            "--response-key", manualResponseKey,
        ])
    }

    private func consume(runner: ToolJobRunner, stream: AsyncStream<ToolEvent>) {
        Task { [weak self] in
            for await event in stream {
                self?.consume(event)
            }
            if self?.runner === runner { self?.runner = nil }
        }
    }

    private func consume(_ event: ToolEvent) {
        switch event.type {
        case .started:
            progress = nil
            progressCurrent = nil
            progressTotal = nil
        case .progress:
            progress = event.progress
            progressCurrent = event.current
            progressTotal = event.total
        case .warning, .completed, .failed, .cancelled:
            if let eventProgress = event.progress { progress = eventProgress }
            if let current = event.current { progressCurrent = current }
            if let total = event.total { progressTotal = total }
        }
        statusMessage = localizedStatus(for: event)
        switch event.type {
        case .started, .progress:
            break
        case .warning:
            warnings.append(L10n.backendMessage(event.message ?? "", fallback: "数据检查发现需要注意的问题"))
        case .completed:
            do {
                if event.tool == "model_conversion", let result = event.result {
                    modelResult = try result.decode(ModelConversionResult.self)
                    modelState = .completed
                } else if event.tool == "dataset_preview", let result = event.result {
                    let analysis = try result.decode(DatasetPreviewResult.self)
                    applyDatasetAnalysis(analysis)
                    if let key = pendingDatasetCacheKey {
                        DatasetPreflightCache.save(analysis, for: key)
                    }
                    datasetState = .completed
                } else if event.tool == "dataset_preview_page", let result = event.result {
                    let page = try result.decode(DatasetPreviewPageResult.self)
                    datasetPreviewSamples = page.preview
                    datasetPreviewPage = page.page
                    datasetPreviewPageSize = page.pageSize
                    datasetPreviewPageCount = page.pageCount
                    datasetPreviewTotal = page.total
                    datasetState = .completed
                } else if event.tool == "dataset_import" {
                    importedDatasetPath = event.path
                    datasetState = .completed
                } else if event.tool == "quantization", let result = event.result {
                    quantizeResult = try result.decode(QuantizationResult.self)
                    quantizeState = .completed
                }
            } catch {
                fail(L10n.format("无法解析工具结果：%@", error.localizedDescription))
            }
        case .failed:
            fail(L10n.backendMessage(event.message ?? "", fallback: "工具任务失败，请查看诊断日志"))
        case .cancelled:
            modelState = modelState == .running ? .cancelled : modelState
            quantizeState = quantizeState == .running ? .cancelled : quantizeState
            datasetState = datasetState == .running ? .cancelled : datasetState
        }
    }

    private func fail(_ message: String) {
        errorMessage = message
        if modelState == .running { modelState = .failed }
        if quantizeState == .running { quantizeState = .failed }
        if datasetState == .running { datasetState = .failed }
    }

    private func localizedStatus(for event: ToolEvent) -> String {
        switch event.type {
        case .started:
            switch event.tool {
            case "model_conversion": return L10n.string("正在转换模型…")
            case "quantization": return L10n.string("正在量化模型…")
            case "dataset_preview": return L10n.string("正在检查数据集…")
            case "dataset_preview_page": return L10n.string("正在加载预览…")
            case "dataset_import": return L10n.string("正在转换数据集…")
            default: return L10n.string("正在启动工具任务…")
            }
        case .progress:
            switch event.phase {
            case "read": return L10n.string("正在读取模型权重…")
            case "convert": return L10n.string("正在转换模型权重…")
            case "write": return L10n.string("正在写入模型…")
            case "load": return L10n.string("正在加载模型…")
            case "quantize": return L10n.string("正在量化模型权重…")
            case "save": return L10n.string("正在保存量化模型…")
            case "tokenizer": return L10n.string("正在加载 tokenizer…")
            case "inspect": return L10n.string("正在准备数据检查…")
            case "render": return L10n.string("正在检查并渲染样本…")
            default: return L10n.string("工具任务进行中…")
            }
        case .warning:
            return L10n.string("数据检查发现需要注意的问题")
        case .completed:
            return L10n.string("工具任务已完成")
        case .failed:
            return L10n.string("工具任务失败")
        case .cancelled:
            return L10n.string("工具任务已取消")
        }
    }

    private func resetDatasetPreviewPage() {
        datasetPreviewSamples = []
        datasetPreviewPage = 1
        datasetPreviewPageSize = 20
        datasetPreviewPageCount = 0
        datasetPreviewTotal = 0
    }

    private func applyDatasetAnalysis(_ analysis: DatasetPreviewResult) {
        datasetAnalysis = analysis
        datasetPreviewSamples = analysis.preview
        datasetPreviewPage = 1
        datasetPreviewPageSize = analysis.pagination?.pageSize ?? max(1, analysis.preview.count)
        datasetPreviewPageCount = analysis.pagination?.pageCount ?? (analysis.preview.isEmpty ? 0 : 1)
        datasetPreviewTotal = analysis.pagination?.total ?? analysis.preview.count
        datasetPreviewCachePath = analysis.pagination?.cachePath
    }

    private func removeDatasetPreviewCache() {
        guard let path = datasetPreviewCachePath else { return }
        // UUID-era temporary caches remain disposable. Stable shared preflight
        // caches live under ~/Library/Caches/Preen and are reused by training.
        if !path.hasPrefix(DatasetPreflightCache.rootURL.path) {
            try? FileManager.default.removeItem(atPath: path)
            try? FileManager.default.removeItem(atPath: path + ".meta.json")
        }
        datasetPreviewCachePath = nil
        pendingDatasetCacheKey = nil
    }
}
