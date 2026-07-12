//
//  GenConfig.swift
//  Preen
//
//  推理采样配置 7 字段(set_config 指令支持的字段集)。
//
//  默认值用 **serve new_session 的高创造力档**(300/1.2/0.5/42/0.4/0.4/0.996),
//  对齐 chat CLI(chat.py:100-104 注释称"实测好档")。
//  **不是** 裸 GenerationConfig 默认(80/0.0/0.9)—— 那是 preview 用的贪心档。
//
//  config_groups 分组(照搬 chat.py::config_groups,design.md §6.2):
//   - 采样(4):temperature / top_p / max_tokens / seed
//   - 重复惩罚(3):presence_penalty / frequency_penalty / penalty_decay
//

import Foundation

/// 推理采样配置。Swift UI 编辑,提交时转 GenConfigDTO 发给 serve。
struct GenConfig: Equatable {

    // === 采样组(4)===
    var temperature: Double = 1.2
    var topP: Double = 0.5
    var maxTokens: Int = 300
    var seed: Int = 42

    // === 重复惩罚组(3,ChatRWKV 官方默认)===
    var presencePenalty: Double = 0.4
    var frequencyPenalty: Double = 0.4
    var penaltyDecay: Double = 0.996

    /// 默认配置(高创造力档)。
    static let defaultConfig = GenConfig()

    /// 转成 serve 协议 DTO(只发 7 个 set_config 允许的字段)。
    func toDTO() -> GenConfigDTO {
        GenConfigDTO(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            seed: seed,
            presencePenalty: presencePenalty,
            frequencyPenalty: frequencyPenalty,
            penaltyDecay: penaltyDecay
        )
    }
}
