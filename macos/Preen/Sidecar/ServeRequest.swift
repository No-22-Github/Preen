//
//  ServeRequest.swift
//  Preen
//
//  对应 src/statetuner/serve.py 的请求帧(stdin JSON lines)。
//
//  契约(serve.py §3.2-3.3):
//   - 请求 = 一行一个 JSON 对象 `{"id","cmd",...params}`。
//   - id 客户端生成,原样透传到所有由该请求触发的事件。
//   - 每个请求恰好一个终结事件(ok/error),中间可有流式事件。
//
//  本文件覆盖指令集全集;UI 实际调用的子集见 ServeClient。
//

import Foundation

/// serve 请求。`id` + `cmd` 必填,其余字段按指令而异。
///
/// 编码时 `id` / `cmd` 顶层,其余参数也顶层(不嵌套在 params 里)。
struct ServeRequest: Encodable {
    let id: String
    let cmd: String
    // 用一个内部 payload enum 把其余字段编码到顶层。
    private let params: Params

    enum Params {
        case hello
        case newSession(NewSessionParams)
        case send(SendParams)
        case abort
        case setState(SetStateParams)
        case setConfig(SetConfigParams)
        case rewind(RewindParams)
        case reset(SessionParams)
        case closeSession(SessionParams)
        case preview(PreviewParams)
        case shutdown
    }

    // MARK: 编码:cmd 之外的 params 字段都铺到顶层

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: TopKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(cmd, forKey: .cmd)
        // params 的字段用 dynamic 编码塞进同一层。
        try params.encode(to: encoder)
    }

    private enum TopKeys: String, CodingKey {
        case id, cmd
    }
}

// MARK: - 各指令的参数

struct NewSessionParams: Encodable {
    let template: String?
    let reasoning: Bool?
    let think: String?  // off / fast / on
    let statePath: String?
    let genConfig: GenConfigDTO?

    enum CodingKeys: String, CodingKey {
        case template, reasoning, think
        case statePath = "state_path"
        case genConfig = "gen_config"
    }
}

struct SendParams: Encodable {
    let sessionId: String
    let text: String

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case text
    }
}

struct SetStateParams: Encodable {
    let sessionId: String
    /// null = 关闭 state(重置会话);字符串 = 加载 state。
    let statePath: String?

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case statePath = "state_path"
    }
}

struct SetConfigParams: Encodable {
    let sessionId: String
    /// gen_config dict(7 字段子集)。
    let genConfig: GenConfigDTO

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case genConfig = "gen_config"
    }
}

struct RewindParams: Encodable {
    let sessionId: String
    /// 默认 1,>=1。一"轮" = user + assistant 对。
    let n: Int?

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case n
    }
}

struct SessionParams: Encodable {
    let sessionId: String

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
    }
}

struct PreviewParams: Encodable {
    let prompt: String
    let template: String?
    let reasoning: Bool?
    let think: String?
    let statePath: String?
    let ab: Bool?
    let genConfig: GenConfigDTO?

    enum CodingKeys: String, CodingKey {
        case prompt, template, reasoning, think
        case statePath = "state_path"
        case ab
        case genConfig = "gen_config"
    }
}

// MARK: - gen_config DTO

/// gen_config dict。7 个采样/惩罚字段,全 Optional(只传要改的)。
/// JSON key 用下划线(serve 协议),与 CLI 的连字符 flag 区分。
struct GenConfigDTO: Encodable {
    var maxTokens: Int?
    var temperature: Double?
    var topP: Double?
    var seed: Int?
    var presencePenalty: Double?
    var frequencyPenalty: Double?
    var penaltyDecay: Double?

    enum CodingKeys: String, CodingKey {
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case seed
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case penaltyDecay = "penalty_decay"
    }

    /// 全 nil 时编码为空 dict(避免发 `null`)。
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(maxTokens, forKey: .maxTokens)
        try c.encodeIfPresent(temperature, forKey: .temperature)
        try c.encodeIfPresent(topP, forKey: .topP)
        try c.encodeIfPresent(seed, forKey: .seed)
        try c.encodeIfPresent(presencePenalty, forKey: .presencePenalty)
        try c.encodeIfPresent(frequencyPenalty, forKey: .frequencyPenalty)
        try c.encodeIfPresent(penaltyDecay, forKey: .penaltyDecay)
    }
}

// MARK: - Params 的顶层编码(把字段铺到 encoder 顶层)

extension ServeRequest.Params {
    func encode(to encoder: Encoder) throws {
        switch self {
        case .hello, .abort, .shutdown:
            break  // 无额外参数
        case .newSession(let p):
            try p.encode(to: encoder)
        case .send(let p):
            try p.encode(to: encoder)
        case .setState(let p):
            try p.encode(to: encoder)
        case .setConfig(let p):
            try p.encode(to: encoder)
        case .rewind(let p):
            try p.encode(to: encoder)
        case .reset(let p), .closeSession(let p):
            try p.encode(to: encoder)
        case .preview(let p):
            try p.encode(to: encoder)
        }
    }
}

// MARK: - 便捷构造(给 ServeClient 用)

extension ServeRequest {
    static func hello(id: String) -> ServeRequest {
        ServeRequest(id: id, cmd: "hello", params: .hello)
    }
    static func newSession(id: String,
                           template: String? = "qa",
                           reasoning: Bool? = nil,
                           think: String? = nil,
                           statePath: String? = nil,
                           genConfig: GenConfigDTO? = nil) -> ServeRequest {
        ServeRequest(id: id, cmd: "new_session",
                     params: .newSession(NewSessionParams(template: template, reasoning: reasoning,
                                                          think: think, statePath: statePath,
                                                          genConfig: genConfig)))
    }
    static func send(id: String, sessionId: String, text: String) -> ServeRequest {
        ServeRequest(id: id, cmd: "send", params: .send(SendParams(sessionId: sessionId, text: text)))
    }
    static func abort(id: String) -> ServeRequest {
        ServeRequest(id: id, cmd: "abort", params: .abort)
    }
    static func setState(id: String, sessionId: String, statePath: String?) -> ServeRequest {
        ServeRequest(id: id, cmd: "set_state",
                     params: .setState(SetStateParams(sessionId: sessionId, statePath: statePath)))
    }
    static func setConfig(id: String, sessionId: String, genConfig: GenConfigDTO) -> ServeRequest {
        ServeRequest(id: id, cmd: "set_config",
                     params: .setConfig(SetConfigParams(sessionId: sessionId, genConfig: genConfig)))
    }
    static func rewind(id: String, sessionId: String, n: Int? = nil) -> ServeRequest {
        ServeRequest(id: id, cmd: "rewind", params: .rewind(RewindParams(sessionId: sessionId, n: n)))
    }
    static func reset(id: String, sessionId: String) -> ServeRequest {
        ServeRequest(id: id, cmd: "reset", params: .reset(SessionParams(sessionId: sessionId)))
    }
    static func closeSession(id: String, sessionId: String) -> ServeRequest {
        ServeRequest(id: id, cmd: "close_session", params: .closeSession(SessionParams(sessionId: sessionId)))
    }
    static func preview(id: String,
                        prompt: String,
                        template: String? = "raw",
                        reasoning: Bool? = nil,
                        think: String? = nil,
                        statePath: String? = nil,
                        ab: Bool? = nil,
                        genConfig: GenConfigDTO? = nil) -> ServeRequest {
        ServeRequest(id: id, cmd: "preview",
                     params: .preview(PreviewParams(prompt: prompt, template: template,
                                                    reasoning: reasoning, think: think,
                                                    statePath: statePath, ab: ab,
                                                    genConfig: genConfig)))
    }
    static func shutdown(id: String) -> ServeRequest {
        ServeRequest(id: id, cmd: "shutdown", params: .shutdown)
    }

    /// 把请求序列化成一行 JSON(带换行,准备写 stdin)。
    func encodeToLine() throws -> String {
        let data = try JSONEncoder().encode(self)
        // 不转义非 ASCII(中文 delta 原样写出,与 Python ensure_ascii=False 对齐)。
        // JSONEncoder 默认就是 UTF-8 不转义。
        return String(data: data, encoding: .utf8)! + "\n"
    }
}
