import SwiftUI

struct HomeView: View {
    @Bindable var appState: AppState

    private var recentRuns: [TrainingRun] {
        var runs = appState.runs
        if let current = appState.trainStore.currentRun,
           !runs.contains(where: { $0.id == current.id }) {
            runs.insert(current, at: 0)
        }
        return runs.sorted { $0.createdAt > $1.createdAt }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                Image("PreenTitle")
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 620, maxHeight: 160)
                    .frame(maxWidth: .infinity)
                    .accessibilityLabel("Preen")

                PreenGlassEffectGroup(spacing: 8) {
                    HStack(spacing: 12) {
                        QuickActionCard(
                            title: "开始训练", subtitle: "选择数据并配置 State Tuning", systemImage: "play.fill",
                            isEnabled: !appState.modelPath.isEmpty
                        ) { appState.selection = .training }
                        QuickActionCard(
                            title: "继续最近记录", subtitle: recentRuns.first.map { $0.status.label } ?? "暂无记录",
                            systemImage: "clock.arrow.circlepath", isEnabled: !recentRuns.isEmpty
                        ) {
                            appState.selectedRunID = recentRuns.first?.id
                            appState.selection = .history
                        }
                        let stateRun = recentRuns.first { $0.artifacts.statePath != nil }
                        QuickActionCard(
                            title: "测试最近 State", subtitle: stateRun?.artifacts.statePath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "暂无可用 State",
                            systemImage: "bubble.left.and.bubble.right", isEnabled: stateRun != nil && !appState.modelPath.isEmpty
                        ) {
                            if let path = stateRun?.artifacts.statePath {
                                appState.goToChat(stateURL: URL(fileURLWithPath: path))
                            }
                        }
                    }
                }

                if appState.trainStore.state == .running || appState.trainStore.state == .finishing {
                    Button { appState.selection = .training } label: {
                        HStack {
                            ProgressView(value: appState.trainStore.progress)
                            Text("当前训练 · \(Int(appState.trainStore.progress * 100))%")
                            Spacer()
                            Text("打开")
                        }
                    }
                    .buttonStyle(.bordered)
                }

                RecentRunsView(runs: recentRuns) { run in
                    appState.selectedRunID = run.id
                    appState.selection = .history
                }
            }
            .padding(28)
        }
        .task { await appState.refreshRuns() }
    }
}
