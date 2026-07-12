//
//  ServeEvent.swift
//  Preen
//
//  对应 src/statetuner/serve.py 从 stdout 发出的 serve 事件流(JSON lines)。
//
//  契约要点(已核对 serve.py + inference.py):
//   - `ready` 是进程级事件,**无 id**,首条,UI 必须先收到它再发指令。
//   - 其余事件都带 `id`(原样透传请求 id)和可选 `session_id`。
//   - `text_chunk` 流式增量,`phase` 区分 think/answer(仅 reasoning && think=on && template!=raw)。
//   - `turn_end` 结算,`result` = GenerationResult.to_dict()(cache 字段已 pop)。
//     ab 模式带 `side: "with_state"|"baseline"`。
//   - `ok`/`error` 是终结事件,每个请求恰好一个。
//   - `error.code` 用结构化枚举,**不用字符串匹配**(serve.py 附录 E.4 教训)。
//

import Foundation

/// serve stdout 事件。所有 case 都带 `id`(`.ready` 除外 —— 它是进程级无 id 事件)。
enum ServeEvent: Decodable {
    /// 进程就绪。**无 id**,首条,UI 收到它才能发指令。
    /// payload = hello 内容(protocol_version / version / model / capabilities)。
    case ready(ReadyPayload)

    /// 流式增量文本。
    /// - `id`:对应请求 id;`session_id`:对应会话(可选)。
    /// - `delta`:增量文本片段。
    /// - `phase`:`.think` / `.answer`(决定 UI dim 渲染;仅 reasoning && think=on && template!=raw 时区分)。
    case textChunk(id: String, sessionId: String?, delta: String, phase: ServePhase)

    /// 一轮生成结算。result = GenerationResult(cache 已 pop)。
    /// ab 模式带 side;think=on 时顶层有 thinking/answer 拆分字段。
    case turnEnd(id: String, sessionId: String?, side: ServeSide?, result: GenerationResult,
                 thinking: String?, answer: String?)

    /// 终结事件:请求成功。具体 payload 随指令而异(如 new_session 带 session_id)。
    case ok(id: String, payload: OkPayload)

    /// 终结事件:请求失败。code 用结构化枚举。
    case error(id: String?, code: ServeErrorCode, message: String)

    // MARK: - 自定义解码(按 type 字段分支)

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Keys.self)
        let type = try c.decode(String.self, forKey: .type)

        switch type {
        case "ready":
            // ready 无 id。
            self = .ready(try c.decode(ReadyPayload.self, forKey: .payload))
        case "text_chunk":
            let phaseStr = try c.decodeIfPresent(String.self, forKey: .phase) ?? "answer"
            self = .textChunk(
                id: try c.decode(String.self, forKey: .id),
                sessionId: try c.decodeIfPresent(String.self, forKey: .sessionId),
                delta: try c.decodeIfPresent(String.self, forKey: .delta) ?? "",
                phase: ServePhase(raw: phaseStr)
            )
        case "turn_end":
            self = .turnEnd(
                id: try c.decode(String.self, forKey: .id),
                sessionId: try c.decodeIfPresent(String.self, forKey: .sessionId),
                side: try c.decodeIfPresent(ServeSide.self, forKey: .side),
                result: try c.decode(GenerationResult.self, forKey: .result),
                thinking: try c.decodeIfPresent(String.self, forKey: .thinking),
                answer: try c.decodeIfPresent(String.self, forKey: .answer)
            )
        case "ok":
            self = .ok(
                id: try c.decode(String.self, forKey: .id),
                payload: try c.decode(OkPayload.self, forKey: .payload)
            )
        case "error":
            let codeRaw = try c.decodeIfPresent(String.self, forKey: .code) ?? "internal"
            self = .error(
                id: try c.decodeIfPresent(String.self, forKey: .id),
                code: ServeErrorCode(raw: codeRaw),
                message: try c.decodeIfPresent(String.self, forKey: .message) ?? ""
            )
        default:
            throw ServeDecodeError.unknownEventType(type)
        }
    }

    private enum Keys: String, CodingKey {
        case type, id
        case sessionId = "session_id"
        case payload, phase, delta, result
        case side, thinking, answer
        case code, message
    }

    /// 此事件对应的请求 id(`.ready` 为 nil)。
    var requestId: String? {
        switch self {
        case .ready: return nil
        case .textChunk(let id, _, _, _): return id
        case .turnEnd(let id, _, _, _, _, _): return id
        case .ok(let id, _): return id
        case .error(let id, _, _): return id
        }
    }
}

enum ServeDecodeError: Error {
    case unknownEventType(String)
}

// MARK: - ready payload

/// `ready` 事件的 payload,也是 `hello` 指令的返回。
struct ReadyPayload: Codable {
    let protocolVersion: Int
    let version: String
    let model: String
    let capabilities: Capabilities

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case version, model, capabilities
    }
}

/// serve 能力声明。UI 据此决定 think 档位是否显示。
struct Capabilities: Codable {
    let templates: [String]
    let think: [String]
    let reasoning: Bool
}

// MARK: - text_chunk.phase

/// `text_chunk.phase` 字段。决定 UI dim 渲染。
enum ServePhase: String, Codable {
    case think
    case answer

    init(raw: String) {
        switch raw {
        case "think": self = .think
        default: self = .answer
        }
    }
}

// MARK: - turn_end.side

/// `turn_end.side` 字段(ab 模式)。
enum ServeSide: String, Codable {
    case withState = "with_state"
    case baseline
}

// MARK: - error.code(结构化,不用字符串匹配)

/// serve 错误码。对应 serve.py 的 ProtocolError.code 字段。
///
/// UI 处理建议(Phase3-总体Spec.md §3.5):
/// - badRequest:表单标红
/// - notFound:刷新列表
/// - busy:禁用发送兜底
/// - aborted:静默
/// - internal:弹错误框 + 建议重启 serve
enum ServeErrorCode: String, Codable {
    case badRequest = "bad_request"
    case notFound = "not_found"
    case busy
    case aborted
    case `internal`

    init(raw: String) {
        switch raw {
        case "bad_request": self = .badRequest
        case "not_found": self = .notFound
        case "busy": self = .busy
        case "aborted": self = .aborted
        default: self = .internal
        }
    }
}

// MARK: - turn_end.result(GenerationResult.to_dict())

/// 推理结果。对应 inference.py `GenerationResult.to_dict()`(cache 字段已 pop)。
struct GenerationResult: Decodable {
    let text: String
    /// 干净展示文本对应的 token 序列。
    let displayTokenIds: [Int]
    let tokenCount: Int
    /// eos / stop_sequence / max_tokens
    let stopReason: String
    let elapsed: Double
    let promptTokens: Int
    let promptTime: Double
    let generationTime: Double
    let promptTps: Double?
    let generationTps: Double?
    let usedState: Bool?
    /// cache 洁净性(stop_sequence 停止时为 false → 下轮自动重放)。
    let cacheClean: Bool?
    let config: GenConfigSnapshot?

    enum CodingKeys: String, CodingKey {
        case text
        case displayTokenIds = "display_token_ids"
        case tokenCount = "token_count"
        case stopReason = "stop_reason"
        case elapsed
        case promptTokens = "prompt_tokens"
        case promptTime = "prompt_time"
        case generationTime = "generation_time"
        case promptTps = "prompt_tps"
        case generationTps = "generation_tps"
        case usedState = "used_state"
        case cacheClean = "cache_clean"
        case config
    }
}

/// GenerationResult 里嵌入的 config 快照(只读展示用,字段全 Optional)。
struct GenConfigSnapshot: Decodable {
    let maxTokens: Int?
    let temperature: Double?
    let topP: Double?
    let seed: Int?
    let presencePenalty: Double?
    let frequencyPenalty: Double?
    let penaltyDecay: Double?

    enum CodingKeys: String, CodingKey {
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case seed
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case penaltyDecay = "penalty_decay"
    }
}

// MARK: - ok payload(随指令而异)

/// `ok` 事件的 payload。不同指令带不同字段,全 Optional 容错。
struct OkPayload: Decodable {
    let sessionId: String?
    let stateLabel: String?
    let historyCleared: Bool?
    let message: String?
    let roundsRemoved: Int?
    let historyLen: Int?
    let protocolVersion: Int?
    let version: String?
    let model: String?
    let capabilities: Capabilities?

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case stateLabel = "state_label"
        case historyCleared = "history_cleared"
        case message
        case roundsRemoved = "rounds_removed"
        case historyLen = "history_len"
        case protocolVersion = "protocol_version"
        case version, model, capabilities
    }
}
