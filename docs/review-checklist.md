# macOS 产品化实施计划

> 目标：把当前 SwiftUI 链路验证版推进为可长期使用的 state tuning 工作台。
> 本文是实施 checklist；产品形态与术语仍以根目录 `design.md` 为准。

## 已裁决方向

- 首页是工作台入口，不是营销落地页：品牌头图 + 三个快捷动作 + 最近训练。
- 训练记录持久化到 `~/Library/Application Support/Preen/runs/<run-id>/`。
- App 启动即检查 Python 运行时；模型推理进程按需常驻，训练仍使用一次性独立进程。
- loss 不修改训练口径：保留 raw step loss，Swift 侧叠加可调 EMA 平滑线。
- GitHub 继续使用带白底的 `assets/preen_title.png`；App 使用 Asset Catalog 中的透明深浅色标题图。

## 交付顺序

每个 commit 都必须可独立编译，不把后续 UI placeholder 提前塞进前一片。建议提交信息：

| 顺序 | commit message | 交付物 | 停点 |
|---|---|---|---|
| 0 | `feat(macos): integrate Preen brand assets and productization plan` | AppIcon、深浅标题图、设计与计划 | 本轮完成 |
| 1 | `feat(macos): add backend runtime and run repository foundation` | BackendStore、运行时检查、RunRepository | 数据模型/API review |
| 2 | `feat(macos): persist training run lifecycle and logs` | run 全生命周期、events/stderr 持久化 | 重启恢复 review |
| 3 | `feat(macos): add home workspace and global backend status` | 首页、快捷入口、后端状态、单一模型源 | 深浅色截图 review |
| 4 | `feat(macos): improve training metrics and smoothed loss chart` | 正确进度/ETA、raw+EMA、hover 图表 | 曲线 fixture review |
| 5 | `feat(macos): add training history and state workspace` | 记录列表、详情、日志与产物操作 | 端到端工作流 review |
| 6 | `feat(macos): add process metrics and background feedback` | RSS/swap、状态栏、Dock、通知 | 0.4B 真机 smoke |

## Commit 1：运行时与持久化地基

主要文件：

- `macos/Preen/Models/BackendStatus.swift`
- `macos/Preen/Persistence/TrainingRun.swift`
- `macos/Preen/Persistence/RunRepository.swift`
- `macos/Preen/Sidecar/RuntimeCheckRunner.swift`
- `macos/Preen/Stores/BackendStore.swift`
- `macos/Preen/Stores/AppState.swift`
- `macos/Preen/Sidecar/PythonResolver.swift`
- `macos/Preen/PreenApp.swift`

- [x] 新增 `BackendStore`：统一暴露 runtime / inference / training 状态。
- [x] App 启动执行 `statetuner doctor --json`，不调用 `load_model`。
- [x] `TrainJobRunner` / `ServeClient` 暴露 PID、退出原因和日志增量。
- [x] 新增 `RunRepository` 与 `TrainingRun` Codable 模型，写入采用临时文件 + replace。
- [x] `PythonResolver` 统一提供 `runs/`、`states/`、`datasets/`、`logs/` 路径。
- [x] 新增 macOS 单测 target，先覆盖 Codable round-trip、原子写和目录扫描。

验收：启动后左下角 3 秒内显示运行时状态；未选模型时 Python/MLX 检查可完成，模型内存保持未加载。

验证：`xcodebuild ... build`；运行时检查 fixture 成功/缺 Python/缺 MLX 三种结果均有确定状态。

## Commit 2：训练记录全生命周期

主要文件：

- `macos/Preen/Sidecar/TrainJobRunner.swift`
- `macos/Preen/Stores/TrainStore.swift`
- `macos/Preen/Persistence/RunRepository.swift`
- `macos/Preen/Models/TrainingConfig.swift`

- [x] 点击开始训练即创建 UUID run，不等训练完成。
- [x] 自动指定 `events.jsonl`，并持续写入 `stderr.log`。
- [x] 状态穷举：preparing / running / finishing / completed / failed / cancelled / interrupted。
- [x] completed 后关联现有 `.meta.json`、state、pth、checkpoint。
- [x] App 重启扫描 runs；没有终结事件的旧 running 记录标记 interrupted。

验收：成功、失败、取消和 App 异常退出四条路径都留下可读记录；重启 App 后仍能打开曲线与日志。

验证：用合成 `TrainEvent` 流覆盖六种终结状态，不在这一片加载真实模型。

## Commit 3：首页与全局骨架

主要文件：

- `macos/Preen/Views/Home/HomeView.swift`
- `macos/Preen/Views/Home/QuickActionCard.swift`
- `macos/Preen/Views/Home/RecentRunsView.swift`
- `macos/Preen/Views/Backend/BackendStatusView.swift`
- `macos/Preen/Views/Backend/BackendLogSheet.swift`
- `macos/Preen/Views/Sidebar.swift`
- `macos/Preen/ContentView.swift`
- `macos/Preen/Stores/AppState.swift`

- [x] 侧边栏增加「首页」，设为默认入口。
- [x] 顶部使用 `Image("PreenTitle")`，高度控制在 140-180pt，适配深浅外观。
- [x] 三个快捷入口：开始训练、继续最近记录、测试最近 State。
- [x] 下方显示最近训练列表与当前任务，不做卡片墙。
- [x] 侧边栏左下角增加后端状态入口；点击打开日志/环境面板。
- [x] 模型路径改为 App 级单一事实源，训练与对话不再各选一次。

验收：默认窗口首屏能看到品牌、主操作、最近记录和后端状态；空数据时仍有明确下一步。

验证：1180x760、1000x680、深色和浅色四组截图；AppIcon 在 Dock/Finder 中不得出现内层硬方块或整片灰底。

> 视觉审查停点：Commit 3 完成后先做深色/浅色截图评审，再继续训练 Dashboard。

## Commit 4：训练正确性与 TensorBoard 风格曲线

主要文件：

- `macos/Preen/Stores/TrainStore.swift`
- `macos/Preen/Views/Training/TrainingRunningView.swift`
- `macos/Preen/Views/Training/TrainingChartView.swift`
- `macos/Preen/Models/TrainingMetric.swift`
- `macos/PreenTests/TrainingMetricTests.swift`

- [x] 修正 0-based step 的显示、进度和 ETA 计算。
- [x] ETA 改用 step 时间戳的滚动统计，不混入模型加载时间。
- [x] held-out loss 使用真实 epoch 结束 step；保留 epoch 平均 train loss。
- [x] raw loss 用低透明度细线；EMA 用 2pt 主线，默认 smoothing 0.6，可调 0-0.95。
- [x] 移除每点常驻 PointMark，加入 hover 十字线和 raw / EMA / lr / epoch 提示。
- [x] 失败、取消、完成态都能重新打开同一张曲线。

验收：平滑只影响显示，不改 events 与训练结果；调到 0 可还原 raw 走势；历史记录重开后曲线一致。

验证：固定事件 fixture 断言 EMA 数值、held-out 横坐标、最终进度 100% 和 ETA 不含加载时间。

## Commit 5：训练记录与产物工作台

主要文件：

- `macos/Preen/Views/History/TrainingHistoryView.swift`
- `macos/Preen/Views/History/TrainingRunDetailView.swift`
- `macos/Preen/Views/History/EventLogView.swift`
- `macos/Preen/Models/StateMetadata.swift`
- `macos/Preen/Persistence/RunRepository.swift`
- `macos/Preen/ContentView.swift`

- [x] 将当前 State 库 placeholder 扩成训练记录列表。
- [x] 详情展示配置、数据哈希、实际轮数、final/held-out loss、state std、耗时和全部产物。
- [x] 支持筛选状态、复制日志、导出事件、Finder 中显示、去对话、导出 `.pth`。
- [x] 外部 state 可作为 imported record 登记，不伪造训练来源。

验收：用户不用打开终端即可回答“这次用什么数据和参数训的、为什么失败、产物在哪里”。

验证：用仓库现有 `output/nekoqa_state.meta.json` 加成功/失败 run fixtures 做列表与详情测试。

## Commit 6：性能监视与后台反馈

主要文件：

- `macos/Preen/System/ProcessMetricsSampler.swift`
- `macos/Preen/Stores/BackendStore.swift`
- `macos/Preen/Views/Status/GlobalStatusBar.swift`
- `macos/Preen/Views/Training/TrainingChartView.swift`
- `macos/Preen/System/DockProgressController.swift`
- `macos/Preen/System/TrainingNotificationController.swift`

- [x] Swift 侧 1 Hz 采集训练子进程 phys_footprint、系统压力、swap 和 s/步。
- [x] loss/RSS 双轨共用 step 轴，单位统一 GB（除以 1e9）。
- [x] 全局状态栏显示健康状态；Dock 显示进度；完成/失败/取消发送系统通知。
- [x] 日志面板聚合 runtime、serve、train 三个来源并明确标记来源。

验收：训练窗口从后台切回后，3 秒内能读到机器压力、当前进度、loss 趋势和预计剩余时间。

验证：先用 mock PID/metrics fixture 测 UI，再按 AGENTS.md 只跑一次 0.4B × 200 条真机 smoke；不串联第二个重任务。

## 每片固定验证

```bash
xcodebuild -project macos/Preen.xcodeproj -scheme Preen \
  -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO build
```

增加 macOS tests target 后，每片再跑对应的纯 Swift 单测。只有修改 `src/statetuner/` 时才追加
`.venv/bin/python -m pytest -q`；UI-only commit 不为形式主义重复跑 Python 全测。

## 暂不扩大范围

- 不做常驻 Python supervisor daemon；Swift 继续直接管理子进程。
- 不做关闭 App 后继续训练或 LaunchAgent 接管。
- 不做模型下载器、主题系统和多模型同时加载。
- 不修改 `Trainer` 的 loss、默认超参或精度方案。
