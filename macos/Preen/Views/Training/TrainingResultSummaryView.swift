import SwiftUI

struct TrainingResultSummaryView: View {
    let facts: TrainingResultExplanation

    var body: some View {
        // HIG materials §macOS:内容层使用语义目的的 GroupBox,不自定义背景块。
        // 仅展示训练结果的叙事性结论(结束原因、loss 变化、最佳轮次、State std、耗时);
        // 数据来源、训练参数、模板、模型、SHA-256 等结构化字段已在右侧 inspector,
        // 不在中间重复展示(避免 inspector 与主区重复)。
        GroupBox {
            outcomeSection
                .padding(.vertical, 4)
        }
    }

    private var outcomeSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("训练结果").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 9) {
                factRow("结束原因", terminationText)
                factRow("实际轮数 / 配置上限", epochText)
                if facts.completedAtLeastOneEpoch {
                    if let first = facts.firstEpochLoss, let final = facts.finalEpochLoss {
                        factRow("训练 loss（第 1 轮→最终轮）", lossChangeText(first: first, final: final))
                    }
                    if let best = facts.bestHeldOutLoss {
                        factRow("验证 loss", heldOutText(best: best))
                    } else {
                        factRow("验证集", L10n.string("未启用验证集，无法判断泛化趋势"))
                    }
                    if let stateStd = facts.stateStd {
                        // State std 的健康区间未标定：只按事实显示，不着色、不报警。
                        factRow("State std", String(format: "%.4f", stateStd))
                    }
                } else {
                    factRow("已发生阶段", incompleteStageText)
                }
                if let elapsed = facts.elapsedSeconds {
                    factRow("总耗时", TrainStore.formatDuration(elapsed))
                }
            }
        }
    }

    private var epochText: String {
        let limit = facts.configuredEpochs.map(String.init) ?? "—"
        return "\(facts.actualEpochs) / \(limit)"
    }

    private var terminationText: String {
        switch facts.termination {
        case .completed:
            return L10n.string("正常完成")
        case .earlyStopped(let patience):
            if let patience {
                return L10n.format("因连续 %lld 轮未改善而早停", patience)
            }
            return L10n.string("因验证 loss 未继续改善而早停")
        case .cancelled:
            return L10n.string("用户取消")
        case .failed:
            return L10n.string("训练失败")
        case .interrupted:
            return L10n.string("App 退出导致中断")
        case .inProgress:
            return L10n.string("仍在进行")
        }
    }

    private func lossChangeText(first: Double, final: Double) -> String {
        let values = String(format: "%.4f → %.4f", first, final)
        guard let change = facts.relativeTrainLossChangePercent else { return values }
        if abs(change) < 0.05 { return values + " · " + L10n.string("无变化") }
        let direction = change < 0 ? L10n.string("下降") : L10n.string("上升")
        return values + String(format: " · %@ %.1f%%", direction, abs(change))
    }

    private func heldOutText(best: Double) -> String {
        var text: String
        if let first = facts.firstHeldOutLoss, let final = facts.finalHeldOutLoss {
            text = String(format: "%.4f → %.4f", first, final)
        } else {
            text = String(format: "%.4f", best)
        }
        if let epoch = facts.bestHeldOutEpoch {
            text += L10n.format(" · 最佳 %.4f（第 %lld 轮）", best, epoch)
        }
        return text
    }

    private var incompleteStageText: String {
        var parts = [L10n.string("尚未完成第 1 轮")]
        if let epoch = facts.lastStartedEpoch { parts.append(L10n.format("已进入第 %lld 轮", epoch)) }
        if let steps = facts.completedSteps, steps > 0 { parts.append(L10n.format("已记录 %lld 步", steps)) }
        return parts.joined(separator: " · ")
    }

    @ViewBuilder
    private func factRow(_ label: String, _ value: String, help: String? = nil) -> some View {
        GridRow {
            Text(L10n.string(label))
                .foregroundStyle(.secondary)
                .frame(width: 170, alignment: .leading)
            if let help {
                Text(value).font(.body.monospacedDigit()).textSelection(.enabled).help(help)
            } else {
                Text(value).font(.body.monospacedDigit()).textSelection(.enabled)
            }
        }
    }
}
