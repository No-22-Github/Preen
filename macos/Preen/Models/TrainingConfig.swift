//
//  TrainingConfig.swift
//  Preen
//
//  训练超参,对齐 `statetuner train` CLI 的 17 个 flag。
//  默认值与 cli.py 一致(已核对);commandLineArguments() 生成 argv 给 TrainJobRunner。
//
//  关键 gotcha(子 Agent 契约):
//   - template 只 qa/instruction(train 不接受 raw;eval/chat/preview 才三选一)。
//   - 布尔开关用裸 flag(--early-stop / --no-early-stop、--export-pth),不带 true/false。
//   - --cache-limit-gb 默认 "auto"(物理内存 ×25%)。
//   - 没有 --reasoning / --think(train 命令不存在这两个参数)。
//   - --events-file 是额外 sink(与 stdout 内容相同);我们设它给诊断用,主读 stdout。
//

import Foundation

/// 训练任务模板(train 命令只接受这两个)。
enum TrainingTemplate: String, CaseIterable, Identifiable {
    case qa
    case instruction
    var id: String { rawValue }
    /// 中文标签(UI 用)。
    var label: String {
        switch self {
        case .qa: return L10n.string("QA（问答）")
        case .instruction: return L10n.string("Instruction（指令）")
        }
    }
}

/// 训练配置。@Observable Store 持有它(UI 编辑);commandLineArguments() 生成 argv。
struct TrainingConfig: Equatable {

    // === 路径(必填)===
    var modelPath: String = ""
    var dataPath: String = ""
    var outPath: String = ""  // 默认 state.npz,UI 会填
    var outputPathMode: TrainingOutputPathMode = .automatic
    private(set) var automaticOutputSourceKey: String? = nil
    var eventsFilePath: String = ""  // 可选,诊断用
    var datasetSource: String? = nil
    var datasetVersion: String? = nil
    var datasetSHA256: String? = nil

    // === 超参(默认值与 cli.py 一致)===
    var lr: Double = 1e-4
    var lrFloor: Double = 1e-5
    var warmup: Int = 50
    var ctxLen: Int = 512
    var epochs: Int = 5
    var gradClip: Double = 1.0
    var logEvery: Int = 1
    var earlyStop: Bool = true
    var earlyStopPatience: Int = 3
    var testRatio: Double = 0.1
    var checkpointEvery: Int = 2
    var checkpointDir: String = ""  // 空 = 不存 checkpoint
    var resumePath: String = ""  // 空 = 不 resume
    var seed: Int = 42
    var template: TrainingTemplate = .qa
    var cacheLimitGb: String = "auto"  // "auto" 或 GB 数字字符串
    var dropTruncated: Bool = false  // true = 丢弃超长样本;false(默认)= 截头保尾继续训练

    // === 额外(训完顺手导出 pth)===
    var exportPth: Bool = false
    var pthOutPath: String = ""  // 空 = 默认(out 同名 .pth)

    /// 默认配置。
    static let defaultConfig = TrainingConfig()

    // MARK: - 校验(UI 用)

    /// lr > 0.1 时给 inline warning(design.md §4:lr=1.0 会爆炸,实测)。
    var lrWarnsExplosion: Bool { lr > 0.1 }

    /// 按 service.run_training 的口径预估训练/验证条数与步数。
    /// 早停开启时从有效样本划 test_ratio 做验证，对齐 data.train_test_split 的
    /// `max(1, int(n * ratio))` 公式；早停关闭 = 全量训练，不划分。
    /// 与 Python 侧 total_steps 一致（步数 = 训练条数 × epochs）。
    /// 抽成静态纯函数便于单测锁定跨语言公式对齐。
    static func projectedCounts(
        effectiveValid: Int,
        truncated: Int,
        dropTruncated: Bool,
        earlyStop: Bool,
        testRatio: Double,
        epochs: Int
    ) -> (train: Int, heldOut: Int, steps: Int) {
        let valid = max(0, dropTruncated ? effectiveValid - truncated : effectiveValid)
        let heldOut = earlyStop ? max(1, Int(Double(valid) * testRatio)) : 0
        let train = max(0, valid - heldOut)
        return (train, heldOut, train * epochs)
    }

    /// 必填字段是否齐全(model + data + out)。
    var canStart: Bool {
        !modelPath.isEmpty && !dataPath.isEmpty && !outPath.isEmpty
    }

    // MARK: - argv 生成

    /// 生成 spawn argv:`-m statetuner.cli train --model X --data Y ...`
    func commandLineArguments() -> [String] {
        var argv: [String] = ["-m", "statetuner.cli", "train",
                              "--model", modelPath,
                              "--data", dataPath,
                              "--template", template.rawValue,
                              "--out", outPath,
                              "--lr", String(lr),
                              "--lr-floor", String(lrFloor),
                              "--warmup", String(warmup),
                              "--ctx-len", String(ctxLen),
                              "--epochs", String(epochs),
                              "--grad-clip", String(gradClip),
                              "--log-every", String(logEvery),
                              "--patience", String(earlyStopPatience),
                              "--test-ratio", String(testRatio),
                              "--checkpoint-every", String(checkpointEvery),
                              "--seed", String(seed),
                              "--cache-limit-gb", cacheLimitGb]

        // 布尔开关(裸 flag,不带值)。
        argv.append(earlyStop ? "--early-stop" : "--no-early-stop")
        argv.append(dropTruncated ? "--drop-truncated" : "--keep-truncated")

        if !checkpointDir.isEmpty {
            argv.append(contentsOf: ["--checkpoint-dir", checkpointDir])
        }
        if !resumePath.isEmpty {
            argv.append(contentsOf: ["--resume", resumePath])
        }
        if exportPth {
            argv.append("--export-pth")
            if !pthOutPath.isEmpty {
                argv.append(contentsOf: ["--pth-out", pthOutPath])
            }
        }
        if !eventsFilePath.isEmpty {
            argv.append(contentsOf: ["--events-file", eventsFilePath])
        }
        return argv
    }

    /// 折叠摘要的一行(design.md §4 措辞):
    /// `lr 0.0001 · ctx_len 512 · 3 轮 · 早停 patience 3 · seed 42`
    var summaryLine: String {
        var parts: [String] = [
            "lr \(lr)",
            "ctx_len \(ctxLen)",
            L10n.format("%lld 轮", epochs),
        ]
        if earlyStop {
            parts.append(L10n.format("早停 patience %lld", earlyStopPatience))
        }
        parts.append("seed \(seed)")
        return parts.joined(separator: " · ")
    }

    var persisted: PersistedTrainingConfig {
        PersistedTrainingConfig(
            modelPath: modelPath,
            dataPath: dataPath,
            outputPath: outPath,
            template: template.rawValue,
            learningRate: lr,
            learningRateFloor: lrFloor,
            warmup: warmup,
            contextLength: ctxLen,
            epochs: epochs,
            gradientClip: gradClip,
            earlyStop: earlyStop,
            earlyStopPatience: earlyStopPatience,
            testRatio: testRatio,
            seed: seed,
            cacheLimitGB: cacheLimitGb,
            checkpointDirectory: checkpointDir.isEmpty ? nil : checkpointDir,
            exportPth: exportPth,
            pthOutputPath: pthOutPath.isEmpty ? nil : pthOutPath,
            datasetSource: datasetSource,
            datasetVersion: datasetVersion,
            datasetSHA256: datasetSHA256
        )
    }

    mutating func markDataAsUserSelected(path: String) {
        dataPath = path
        datasetSource = nil
        datasetVersion = nil
        datasetSHA256 = nil
    }

    mutating func refreshAutomaticOutputPath(date: Date = Date(), rootURL: URL = PythonResolver.statesDirectory) {
        guard outputPathMode == .automatic, !modelPath.isEmpty, !dataPath.isEmpty else { return }
        let sourceKey = "\(dataPath)\u{0}\(modelPath)"
        guard outPath.isEmpty || sourceKey != automaticOutputSourceKey else { return }
        outPath = TrainingOutputPath.automaticURL(
            dataPath: dataPath,
            modelPath: modelPath,
            rootURL: rootURL,
            date: date
        ).path
        automaticOutputSourceKey = sourceKey
    }

    mutating func markOutputPathManual(_ path: String) {
        outPath = path
        outputPathMode = .manual
        automaticOutputSourceKey = nil
    }

    mutating func regenerateAutomaticOutputPath(date: Date = Date(), rootURL: URL = PythonResolver.statesDirectory) {
        guard outputPathMode == .automatic else { return }
        outPath = ""
        automaticOutputSourceKey = nil
        refreshAutomaticOutputPath(date: date, rootURL: rootURL)
    }
}
