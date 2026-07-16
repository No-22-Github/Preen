//
//  ChatStore.swift
//  Preen
//
//  对话状态。@Observable @MainActor(UI 线程)。
//
//  本期最小版(单栏):基础多轮 + abort。
//  不做(留 #8):A/B 双栏、崩溃恢复重放、Inspector 会话区。
//
//  事件驱动:
//   - text_chunk(id, delta, phase) → 把增量追加到最后一条 assistant 消息(按 phase 分段)。
//   - turn_end(id, result, thinking?, answer?) → 用拆分后的 thinking/answer 覆盖最后一条;
//     非 think=on 用 result.text。
//   - ok/error 终结事件:ok 标完成,error{aborted} 标中断(保留已生成文本),其他 error 标错误。
//

import Foundation
import Observation

/// 消息角色。
enum ChatRole: Equatable {
    case user
    case assistant
}

/// 消息的一段文本(按 think/answer 分段)。
struct ChatSegment: Identifiable, Equatable {
    let id = UUID()
    let phase: ServePhase
    var text: String
}

/// 一条消息。
struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    let role: ChatRole
    /// assistant 消息按 phase 分段(think/answer);user 消息单段 answer。
    var segments: [ChatSegment]
    /// turn_end 的技术摘要(stop_reason / token 数 / t/s),dim 显示。
    var summary: String?
    /// 中断标记(abort 后保留已生成部分,UI 加"(已中断)")。
    var isAborted: Bool = false
    /// 错误消息(若该轮失败)。
    var errorText: String?

    /// 拼接所有段的文本(用于显示/重放)。
    var fullText: String {
        segments.map(\.text).joined()
    }
}

@Observable
@MainActor
final class ChatStore {
    private let backendStore: BackendStore

    init(backendStore: BackendStore) {
        self.backendStore = backendStore
    }

    convenience init() {
        self.init(backendStore: BackendStore())
    }

    // === 状态 ===
    private(set) var messages: [ChatMessage] = []
    private(set) var sessionId: String?
    private(set) var isGenerating: Bool = false  // UI 据此切换发送/abort 按钮
    private(set) var isConnected: Bool = false  // serve 进程 ready
    private(set) var lastError: String?

    // === 启动日志(连接时弹窗实时展示后端 stderr)===
    /// 后端 stderr 实时累积(连接时清空)。
    private(set) var startupLog: String = ""
    /// 启动失败时的错误信息(nil = 仍在启动 / 已成功)。驱动启动弹窗的成败态展示。
    private(set) var startupError: String?

    /// 当前请求 id(text_chunk/turn_end/终结事件用它配对)。
    private var inFlightId: String?

    /// ServeClient 持有。
    private var client: ServeClient?
    /// 事件流消费 Task。
    private var consumeTask: Task<Void, Never>?

    // === 配置 ===
    var genConfig: GenConfig = .defaultConfig
    var statePath: String?  // 当前 state 文件路径(nil = 无 state 基线)
    /// 当前已连接的模型路径(connect 时记录,disconnect 清)。供状态栏与 toolbar 展示。
    private(set) var connectedModelPath: String?
    /// 模型 + State 已真正落到可用会话上的递增信号。
    /// set_state 的 ok 或 new_session 成功后递增,供顶部模型 chip 做克制的完成反馈。
    private(set) var activationRevision: Int = 0

    // MARK: - 生命周期

    /// 启动 serve + 等待 ready + 自动建 session。
    /// 启动期间 stderr 实时推到 `startupLog`(供启动日志弹窗展示)。
    func connect(model: URL) {
        disconnect()
        connectedModelPath = model.path
        // 重置启动日志(新一轮连接)。
        startupLog = ""
        startupError = nil
        backendStore.updateInference(phase: .starting, message: "正在加载模型")
        let client = ServeClient()
        self.client = client
        // 后端 stderr → 主线程追加到 startupLog(@Observable 自动刷新 UI)。
        // readabilityHandler 在后台队列触发,这里 dispatch 到 MainActor。
        client.onStderr = { [weak self] chunk in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.startupLog += chunk
                self.backendStore.appendInferenceLog(chunk)
                // 上限保护(尾部保留,避免无限增长)。
                if self.startupLog.count > 64 * 1024 {
                    self.startupLog = String(self.startupLog.suffix(64 * 1024))
                }
            }
        }
        // stdout 镜像(仅启动期):看 ready 是否到达 Swift 端。connected 后忽略。
        client.onStdout = { [weak self] line in
            Task { @MainActor [weak self] in
                guard let self, !self.isConnected else { return }
                self.startupLog += "[stdout] \(line)\n"
            }
        }
        let stream = client.start(model: model)
        backendStore.updateInference(phase: .starting, pid: client.pid, message: "正在加载模型")
        consumeTask = Task { [weak self] in
            for await event in stream {
                self?.consume(event: event)
            }
            // stream 自然结束(非取消)= serve 进程 stdout 关闭 = 进程退出。
            if Task.isCancelled { return }
            self?.handleServeExit()
        }
    }

    /// 断开(杀进程)。
    func disconnect() {
        consumeTask?.cancel()
        consumeTask = nil
        client?.terminate()
        client = nil
        isConnected = false
        sessionId = nil
        isGenerating = false
        connectedModelPath = nil
        backendStore.updateInference(phase: .idle, message: "推理未启动")
        // 清理启动日志状态(下次连接重新累积)。
        startupError = nil
    }

    /// 终止 serve 并等待进程真正退出，供训练/推理互斥切换使用。
    func disconnectAndWait() async {
        let departingClient = client
        disconnect()
        await departingClient?.waitUntilExit()
    }

    var hasActiveProcess: Bool {
        client?.isRunning == true
    }

    var processID: Int32? {
        client?.pid
    }

    // MARK: - 用户动作

    /// 发送一条消息。
    func send(text: String) {
        guard let sid = sessionId, isConnected, !isGenerating else { return }
        guard !text.isEmpty else { return }

        // user 消息先入列。
        messages.append(ChatMessage(role: .user, segments: [ChatSegment(phase: .answer, text: text)]))
        // assistant 占位(空段,等 text_chunk 填)。
        messages.append(ChatMessage(role: .assistant, segments: [ChatSegment(phase: .answer, text: "")]))
        isGenerating = true
        lastError = nil

        // 异步发指令;流式事件经 consume(event:) 更新占位。
        Task { [weak self] in
            guard let self else { return }
            do {
                try await self.client?.send(sessionId: sid, text: text)
                // send 返回 = 终结事件到达;consume 已处理。
            } catch let err as ServeError {
                self.handleSendError(err)
            } catch {
                self.handleSendError(.ioError(error.localizedDescription))
            }
        }
    }

    /// abort 当前生成(独立通道:abort 自己的 id,不耦合被中断的 send)。
    func abort() {
        guard isGenerating else { return }
        Task { [weak self] in
            guard let self else { return }
            try? await self.client?.abort()
            // abort 的 ok 立即返回;被中断的 send 的 error{aborted} 会异步到达,
            // 由 consume(event:) 标记 isAborted。
        }
    }

    func clearLastError() {
        lastError = nil
    }

    /// 改采样配置(下一轮生效)。
    func applyConfig() {
        guard let sid = sessionId else { return }
        Task { [weak self] in
            guard let self else { return }
            do {
                let resp = try await self.client?.send(.setConfig(id: self.newClientId(),
                                                                   sessionId: sid,
                                                                   genConfig: self.genConfig.toDTO()))
                if case .error(_, let msg) = resp {
                    self.lastError = msg
                }
            } catch {
                self.lastError = error.localizedDescription
            }
        }
    }

    /// 切换 state 文件。
    func setState(path: String?) {
        guard let sid = sessionId else { return }
        statePath = path
        Task { [weak self] in
            guard let self else { return }
            do {
                let resp = try await self.client?.send(.setState(id: self.newClientId(),
                                                                 sessionId: sid,
                                                                 statePath: path))
                if case .ok(let payload) = resp {
                    // set_state 成功 = 重置会话(history/cache 清空)。
                    self.messages.removeAll()
                    _ = payload  // stateLabel 等可后续展示
                    // state 变了,刷新状态栏摘要(模型 + 新 state)。
                    self.backendStore.updateInference(phase: .ready, message: self.inferenceSummary)
                    self.activationRevision &+= 1
                }
            } catch {
                self.lastError = error.localizedDescription
            }
        }
    }

    /// 清除当前 state(切模型 / 用户卸下时调用)。
    /// 已连接时走后端 set_state(nil)(会重置会话);未连接只清本地字段,
    /// 这样下次连接 newSession() 不会把旧模型的 state 带给新模型。
    func clearState() {
        if sessionId != nil, isConnected {
            setState(path: nil)
        } else {
            statePath = nil
        }
    }

    /// 预注入 state 路径(不立即下发后端):供「去对话」一键流程在未连接时使用。
    /// 仅写本地 statePath,这样随后 connect → ready → newSession() 会自动带上该 state,
    /// 无需用户手动加载。
    func prepareInjectedState(path: String) {
        statePath = path
    }

    /// 推理状态摘要:「模型 X · state Y」或「模型 X · 基线」。供全局状态栏展示。
    var inferenceSummary: String {
        let modelPart = connectedModelPath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "模型"
        let statePart: String
        if let p = statePath {
            statePart = "state \(URL(fileURLWithPath: p).lastPathComponent)"
        } else {
            statePart = "基线"
        }
        return "\(modelPart) · \(statePart)"
    }

    /// 新建会话(连接后自动调,或换模板/切 state 后重建)。
    func newSession(template: String = "qa", reasoning: Bool? = nil, think: String? = nil) {
        Task { [weak self] in
            guard let self else { return }
            do {
                let sid = try await self.client?.newSession(template: template, reasoning: reasoning,
                                                            think: think, statePath: self.statePath,
                                                            genConfig: self.genConfig.toDTO())
                self.sessionId = sid
                self.messages.removeAll()
                self.activationRevision &+= 1
            } catch {
                self.lastError = "建会话失败：\(error.localizedDescription)"
            }
        }
    }

    // MARK: - 事件消费

    func consume(event: ServeEvent) {
        switch event {
        case .ready:
            isConnected = true
            backendStore.updateInference(phase: .ready, pid: client?.pid, message: inferenceSummary)
            // 自动建会话(qa 模板默认)。
            newSession()
        case .textChunk(let id, _, let delta, let phase):
            handleTextChunk(id: id, delta: delta, phase: phase)
        case .turnEnd(let id, _, _, let result, let thinking, let answer):
            handleTurnEnd(id: id, result: result, thinking: thinking, answer: answer)
        case .ok(let id, _):
            // 终结事件(非 send 的指令,如 set_config/set_state)。标记生成结束。
            if id == inFlightId {
                isGenerating = false
                inFlightId = nil
            }
        case .error(let id, let code, let message):
            // 启动期(未 connected)的 error = 后端启动失败,记录给启动日志弹窗。
            if !isConnected {
                startupError = message
            }
            handleError(id: id, code: code, message: message)
        }
    }

    // MARK: - 内部

    /// serve 进程退出(stream 自然结束)时调用。
    /// 若此时仍未 connected,说明没收到 ready = 启动失败,保留启动日志供排查。
    private func handleServeExit() {
        let wasConnected = isConnected
        isConnected = false
        backendStore.updateInference(
            phase: wasConnected ? .idle : .failed,
            message: wasConnected ? "推理进程已退出" : "推理启动失败"
        )
        guard !wasConnected else { return }
        if startupError == nil {
            startupError = "serve 进程已退出，未发出 ready 事件（请查看日志排查）"
        }
    }

    private func newClientId() -> String {
        // ServeClient 内部有自己的 id 生成;此处给那些直传 .send(...) 的便捷场景用。
        // 简单用时间戳+随机;ServeClient.newSession/setState 等自己生成,不走这里。
        "u\(UUID().uuidString.prefix(8))"
    }

    private func handleTextChunk(id: String, delta: String, phase: ServePhase) {
        inFlightId = id
        guard let last = messages.last, last.role == .assistant else { return }
        // 找最后一个同 phase 的段追加;没有就开新段。
        var msg = last
        if let idx = msg.segments.lastIndex(where: { $0.phase == phase }) {
            msg.segments[idx].text += delta
        } else {
            msg.segments.append(ChatSegment(phase: phase, text: delta))
        }
        messages[messages.count - 1] = msg
    }

    private func handleTurnEnd(id: String, result: GenerationResult, thinking: String?, answer: String?) {
        inFlightId = id
        guard let last = messages.last, last.role == .assistant else { return }
        var msg = last

        // think=on 时,用顶层 thinking/answer 覆盖(单一事实源,Swift 不重新拆分)。
        if let t = thinking, let a = answer {
            msg.segments = []
            if !t.isEmpty {
                msg.segments.append(ChatSegment(phase: .think, text: t))
            }
            msg.segments.append(ChatSegment(phase: .answer, text: a))
        } else {
            // 非 think=on:result.text 整体作为 answer 段。
            msg.segments = [ChatSegment(phase: .answer, text: result.text)]
        }

        // 技术摘要(design.md §6:dim 显示 stop_reason/token 数/t/s)。
        msg.summary = buildSummary(result: result)
        messages[messages.count - 1] = msg
        // turn_end 不是终结事件;等 ok/error 标 isGenerating=false。
    }

    private func handleError(id: String?, code: ServeErrorCode, message: String) {
        // error 是终结事件。
        if id == inFlightId {
            isGenerating = false
            inFlightId = nil
        }
        switch code {
        case .aborted:
            // 中断:保留已生成文本,标"(已中断)"。
            if let idx = messages.indices.last, messages[idx].role == .assistant {
                messages[idx].isAborted = true
            }
        case .busy:
            // busy = 已有 in-flight(理论上 UI 已禁用,这是兜底)。
            lastError = "服务器忙：\(message)"
            // 移除空占位(没生成的 assistant 消息)。
            if let idx = messages.indices.last, messages[idx].role == .assistant,
               messages[idx].segments.allSatisfy({ $0.text.isEmpty }) {
                messages.remove(at: idx)
            }
        default:
            lastError = message
            // 标记最后一条 assistant 消息的错误。
            if let idx = messages.indices.last, messages[idx].role == .assistant {
                if messages[idx].fullText.isEmpty {
                    messages[idx].errorText = message
                }
            }
        }
    }

    private func handleSendError(_ err: ServeError) {
        isGenerating = false
        inFlightId = nil
        switch err {
        case .aborted:
            if let idx = messages.indices.last, messages[idx].role == .assistant {
                messages[idx].isAborted = true
            }
        case .busy:
            lastError = "已有生成进行中"
        default:
            lastError = err.localizedDescription
        }
    }

    /// 拼技术摘要(stop_reason · token 数 · t/s)。
    private func buildSummary(result: GenerationResult) -> String {
        var parts: [String] = []
        parts.append("stop=\(result.stopReason)")
        parts.append("tokens=\(result.tokenCount)")
        if let tps = result.generationTps {
            parts.append(String(format: "%.1f t/s", tps))
        }
        return parts.joined(separator: " · ")
    }

    // MARK: - 派生

    /// 是否可以发送(已连接 + 有会话 + 不在生成)。
    var canSend: Bool {
        isConnected && sessionId != nil && !isGenerating
    }
}
