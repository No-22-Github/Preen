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
        case .qa: return "QA(问答)"
        case .instruction: return "Instruction(指令)"
        }
    }
}

/// 训练配置。@Observable Store 持有它(UI 编辑);commandLineArguments() 生成 argv。
struct TrainingConfig: Equatable {

    // === 路径(必填)===
    var modelPath: String = ""
    var dataPath: String = ""
    var outPath: String = ""  // 默认 state.npz,UI 会填
    var eventsFilePath: String = ""  // 可选,诊断用

    // === 超参(默认值与 cli.py 一致)===
    var lr: Double = 0.01
    var lrFloor: Double = 1e-4
    var warmup: Int = 10
    var ctxLen: Int = 512
    var epochs: Int = 20
    var gradClip: Double = 1.0
    var logEvery: Int = 10
    var earlyStop: Bool = true
    var earlyStopPatience: Int = 3
    var testRatio: Double = 0.1
    var checkpointEvery: Int = 2
    var checkpointDir: String = ""  // 空 = 不存 checkpoint
    var resumePath: String = ""  // 空 = 不 resume
    var seed: Int = 42
    var template: TrainingTemplate = .qa
    var cacheLimitGb: String = "auto"  // "auto" 或 GB 数字字符串

    // === 额外(训完顺手导出 pth)===
    var exportPth: Bool = false
    var pthOutPath: String = ""  // 空 = 默认(out 同名 .pth)

    /// 默认配置。
    static let defaultConfig = TrainingConfig()

    // MARK: - 校验(UI 用)

    /// lr > 0.1 时给 inline warning(design.md §4:lr=1.0 会爆炸,实测)。
    var lrWarnsExplosion: Bool { lr > 0.1 }

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
    /// `lr 0.01 · ctx_len 512 · 3 轮 · 早停 patience 3 · seed 42`
    var summaryLine: String {
        var parts: [String] = [
            "lr \(lr)",
            "ctx_len \(ctxLen)",
            "\(epochs) 轮",
        ]
        if earlyStop {
            parts.append("早停 patience \(earlyStopPatience)")
        }
        parts.append("seed \(seed)")
        return parts.joined(separator: " · ")
    }
}
