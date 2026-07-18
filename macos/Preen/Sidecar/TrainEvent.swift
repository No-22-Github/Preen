//
//  TrainEvent.swift
//  Preen
//
//  对应 src/statetuner/events.py 的训练事件流(JSON lines)。
//
//  契约要点(已核对 events.py + train.py + service.py):
//   - Python 侧用单个 dataclass `Event` + 通用可选字段联合体,`type` 决定哪些字段已填。
//   - `to_dict()` = `asdict(self)` 后剔掉所有 == None 的字段(events.py:56-59)。
//     → Swift 侧每个 case 的关联值字段**都要按 Optional 解码**(缺键 = nil)。
//   - `timestamp` = Unix epoch 秒(Double,time.time()),每个事件都有。
//   - `log_every` 抽样:step 事件每 N 步发一条(N 默认 10),不是每步都发。
//
//  演进保护(design.md §4.2):
//   - 解码层有 `unknown` 兜底 —— Python 加新事件类型,旧 app 不崩。
//   - UI 层 switch 必须**穷举所有已知 case**,**不许 default**;
//     unknown case 至少要打 log,不能静默吞。
//

import Foundation

/// 训练事件。case 与 events.py 的 type 字符串一一对应。
///
/// 解码失败(未知 type 或字段类型不符)落入 `.unknown`,保留原始 type 字符串和 JSON 数据,
/// 便于诊断且不让旧 app 崩溃。
enum TrainEvent: Decodable {
    /// `{"type":"start","config":{...}}` —— 训练开始,带超参快照。
    case start(config: TrainConfigSnapshot, timestamp: Double)
    /// 最终 loader 口径的数据事实，在丢弃与训练/验证拆分后、训练循环前发出。
    case dataSummary(
        totalRecords: Int, validSamples: Int, trainSamples: Int, heldOutSamples: Int,
        truncatedSamples: Int, droppedSamples: Int, targetFullyTruncated: Int,
        timestamp: Double
    )
    /// `{"type":"resume","epoch":N,"message":"..."}` —— 从 checkpoint 恢复(train.py:186-192)。
    case resume(epoch: Int, message: String, timestamp: Double)
    /// `{"type":"epoch_start","epoch":N}` —— 每轮开始。
    case epochStart(epoch: Int, timestamp: Double)
    /// `{"type":"step","step":N,"total_steps":K,"loss":X,"lr":Y,"epoch":E?}` —— 抽样的训练步。
    case step(step: Int, totalSteps: Int, loss: Double, lr: Double, epoch: Int?, timestamp: Double)
    /// `{"type":"epoch_end","epoch":N,"loss":X,"state_std":S,"lr":Y,
    ///    "held_out_loss":H?,"best":B?,"patience_left":P?}` —— 每轮结束。
    case epochEnd(epoch: Int, loss: Double, stateStd: Double, lr: Double,
                  heldOutLoss: Double?, best: Double?, patienceLeft: Int?, timestamp: Double)
    /// `{"type":"std_warning","epoch":N,"state_std":S,"message":"..."}` —— state std 越阈(产品默认禁用)。
    case stdWarning(epoch: Int, stateStd: Double, message: String, timestamp: Double)
    /// `{"type":"checkpoint","epoch":N,"path":"..."}` —— 存了 checkpoint。
    case checkpoint(epoch: Int, path: String, timestamp: Double)
    /// `{"type":"early_stop","epoch":N,"best":B,"held_out_loss":H,"message":"..."}` —— 早停触发。
    case earlyStop(epoch: Int, best: Double, heldOutLoss: Double, message: String, timestamp: Double)
    /// `{"type":"final","path":"...","elapsed":T,"best":B?}` —— Trainer 跑完,**产物可能尚未落盘**。
    /// ⚠️ 这是「收尾中」信号,**不是完成**;UI 必须等 `completed` 才切完成态(design.md §4.2 验收 c)。
    case final(path: String, elapsed: Double, best: Double?, timestamp: Double)
    /// `{"type":"completed","path":"...","elapsed":T,"message":M?}` —— **唯一可信完成信号**。
    case completed(path: String, elapsed: Double, message: String?, timestamp: Double)
    /// `{"type":"failed","message":"...","path":P?}` —— 失败。
    case failed(message: String, path: String?, timestamp: Double)
    /// `{"type":"cancelled","message":"..."}` —— 用户 SIGINT 取消。
    case cancelled(message: String, timestamp: Double)
    /// 未知事件类型(演进兜底)。保留原始 type 字符串与未解析 JSON。
    case unknown(type: String, timestamp: Double, payload: [String: Any])

    // MARK: - 自定义解码(events.py 单 dataclass + 剔 None,自动合成不行)

    private struct AnyDecodable: Decodable {
        var value: Any?
        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let v = try? container.decode(Bool.self) { self.value = v }
            else if let v = try? container.decode(Int.self) { self.value = v }
            else if let v = try? container.decode(Double.self) { self.value = v }
            else if let v = try? container.decode(String.self) { self.value = v }
            else if let v = try? container.decode([AnyDecodable].self) {
                self.value = v.map { $0.value }
            } else {
                self.value = nil
            }
        }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Keys.self)
        let type = try c.decode(String.self, forKey: .type)
        let ts = try c.decodeIfPresent(Double.self, forKey: .timestamp) ?? 0

        switch type {
        case "start":
            let cfg = try c.decode(TrainConfigSnapshot.self, forKey: .config)
            self = .start(config: cfg, timestamp: ts)
        case "data_summary":
            self = .dataSummary(
                totalRecords: try c.decode(Int.self, forKey: .totalRecords),
                validSamples: try c.decode(Int.self, forKey: .validSamples),
                trainSamples: try c.decode(Int.self, forKey: .trainSamples),
                heldOutSamples: try c.decode(Int.self, forKey: .heldOutSamples),
                truncatedSamples: try c.decode(Int.self, forKey: .truncatedSamples),
                droppedSamples: try c.decode(Int.self, forKey: .droppedSamples),
                targetFullyTruncated: try c.decode(Int.self, forKey: .targetFullyTruncated),
                timestamp: ts
            )
        case "resume":
            self = .resume(
                epoch: try c.decode(Int.self, forKey: .epoch),
                message: try c.decode(String.self, forKey: .message),
                timestamp: ts
            )
        case "epoch_start":
            self = .epochStart(epoch: try c.decode(Int.self, forKey: .epoch), timestamp: ts)
        case "step":
            self = .step(
                step: try c.decode(Int.self, forKey: .step),
                totalSteps: try c.decode(Int.self, forKey: .totalSteps),
                loss: try c.decode(Double.self, forKey: .loss),
                lr: try c.decode(Double.self, forKey: .lr),
                epoch: try c.decodeIfPresent(Int.self, forKey: .epoch),
                timestamp: ts
            )
        case "epoch_end":
            self = .epochEnd(
                epoch: try c.decode(Int.self, forKey: .epoch),
                loss: try c.decode(Double.self, forKey: .loss),
                stateStd: try c.decode(Double.self, forKey: .stateStd),
                lr: try c.decode(Double.self, forKey: .lr),
                heldOutLoss: try c.decodeIfPresent(Double.self, forKey: .heldOutLoss),
                best: try c.decodeIfPresent(Double.self, forKey: .best),
                patienceLeft: try c.decodeIfPresent(Int.self, forKey: .patienceLeft),
                timestamp: ts
            )
        case "std_warning":
            self = .stdWarning(
                epoch: try c.decode(Int.self, forKey: .epoch),
                stateStd: try c.decode(Double.self, forKey: .stateStd),
                message: try c.decode(String.self, forKey: .message),
                timestamp: ts
            )
        case "checkpoint":
            self = .checkpoint(
                epoch: try c.decode(Int.self, forKey: .epoch),
                path: try c.decode(String.self, forKey: .path),
                timestamp: ts
            )
        case "early_stop":
            self = .earlyStop(
                epoch: try c.decode(Int.self, forKey: .epoch),
                best: try c.decode(Double.self, forKey: .best),
                heldOutLoss: try c.decode(Double.self, forKey: .heldOutLoss),
                message: try c.decode(String.self, forKey: .message),
                timestamp: ts
            )
        case "final":
            self = .final(
                path: try c.decode(String.self, forKey: .path),
                elapsed: try c.decode(Double.self, forKey: .elapsed),
                best: try c.decodeIfPresent(Double.self, forKey: .best),
                timestamp: ts
            )
        case "completed":
            self = .completed(
                path: try c.decode(String.self, forKey: .path),
                elapsed: try c.decode(Double.self, forKey: .elapsed),
                message: try c.decodeIfPresent(String.self, forKey: .message),
                timestamp: ts
            )
        case "failed":
            self = .failed(
                message: try c.decode(String.self, forKey: .message),
                path: try c.decodeIfPresent(String.self, forKey: .path),
                timestamp: ts
            )
        case "cancelled":
            self = .cancelled(
                message: try c.decode(String.self, forKey: .message),
                timestamp: ts
            )
        default:
            // 演进兜底:保留原始 payload 供诊断。
            var payload: [String: Any] = [:]
            if let raw = try? decoder.container(keyedBy: DynamicKey.self) {
                let keys = raw.allKeys
                for key in keys {
                    if let v = try? raw.decode(AnyDecodable.self, forKey: key), let val = v.value {
                        payload[key.stringValue] = val
                    }
                }
            }
            self = .unknown(type: type, timestamp: ts, payload: payload)
        }
    }

    private enum Keys: String, CodingKey {
        case type, timestamp
        case config
        case epoch, step, totalSteps = "total_steps", loss, lr
        case stateStd = "state_std"
        case heldOutLoss = "held_out_loss", best
        case patienceLeft = "patience_left"
        case message, path, elapsed
        case totalRecords = "total_records"
        case validSamples = "valid_samples"
        case trainSamples = "train_samples"
        case heldOutSamples = "held_out_samples"
        case truncatedSamples = "truncated_samples"
        case droppedSamples = "dropped_samples"
        case targetFullyTruncated = "target_fully_truncated"
    }

    private struct DynamicKey: CodingKey {
        var stringValue: String
        init?(stringValue: String) { self.stringValue = stringValue }
        var intValue: Int? { nil }
        init?(intValue: Int) { return nil }
    }

    // MARK: - 便利:事件时间戳统一访问

    /// 事件时间戳(Unix epoch 秒)。unknown 也保底返回。
    var timestamp: Double {
        switch self {
        case .start(_, let t): return t
        case .dataSummary(_, _, _, _, _, _, _, let t): return t
        case .resume(_, _, let t): return t
        case .epochStart(_, let t): return t
        case .step(_, _, _, _, _, let t): return t
        case .epochEnd(_, _, _, _, _, _, _, let t): return t
        case .stdWarning(_, _, _, let t): return t
        case .checkpoint(_, _, let t): return t
        case .earlyStop(_, _, _, _, let t): return t
        case .final(_, _, _, let t): return t
        case .completed(_, _, _, let t): return t
        case .failed(_, _, let t): return t
        case .cancelled(_, let t): return t
        case .unknown(_, let t, _): return t
        }
    }
}

/// `start` 事件的 config 字段 —— `TrainConfig.to_dict()` + `n_samples`。
///
/// `max_state_std`/`checkpoint_dir`/`resume` 在 Python 侧可能是 null,
/// Swift 侧按 Optional 解码。
struct TrainConfigSnapshot: Decodable {
    let lr: Double
    let lrFloor: Double
    let warmup: Int
    let ctxLen: Int
    let epochs: Int
    let gradClip: Double
    let logEvery: Int
    let maxStateStd: Double?
    let earlyStop: Bool
    let earlyStopPatience: Int
    let checkpointDir: String?
    let checkpointEvery: Int
    let resume: String?
    let seed: Int
    /// 额外字段(non-TrainConfig),service 层塞进来的样本数。
    let nSamples: Int

    enum CodingKeys: String, CodingKey {
        case lr
        case lrFloor = "lr_floor"
        case warmup
        case ctxLen = "ctx_len"
        case epochs
        case gradClip = "grad_clip"
        case logEvery = "log_every"
        case maxStateStd = "max_state_std"
        case earlyStop = "early_stop"
        case earlyStopPatience = "early_stop_patience"
        case checkpointDir = "checkpoint_dir"
        case checkpointEvery = "checkpoint_every"
        case resume
        case seed
        case nSamples = "n_samples"
    }
}
