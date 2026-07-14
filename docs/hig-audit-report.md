# Preen macOS HIG 规范性审查报告

> 审查日期:2026-07-13
> 依据:apple-hig skill(全量 tier-1 + `designing-for-macos` + 各页相关组件规范)
> 范围:按侧边栏顺序自上而下逐页对照;菜单栏功能未补齐,已跳过。

窗口结构:`NavigationSplitView`(两栏)+ 底部 `GlobalStatusBar`。侧边栏导航自上而下:**训练 → 对话 → 训练记录 → 工具箱**。共享外壳(容器 / 侧边栏 / 状态栏 / 模型选择器)单列在最后。

**说明:** macOS 不支持 Dynamic Type(typography.md),故固定字号不计缺陷,但按 macOS 默认 13pt / 最小 10pt 与 accessibility.md 的 4.5:1 对比度基线判定。所有条目可追溯到具体 distilled 文件与 `file:line`。

---

## 页面 1:训练 (Training)

`Views/Training/` 全部子视图。整体质量最高的一页(取消确认流是范本级)。

**[Major] 配置摘要行用 `.onTapGesture` 展开** — `TrainingConfigView.swift:31,99-110`。带 chevron 的 `HStack` 靠手势展开,VoiceOver 不识别为可展开控件、键盘无法聚焦。应改 `DisclosureGroup` 或包成 `Button`(accessibility.md:手势需可达替代 + 正确 trait)。

**[Major] 运行中有确定进度却无 determinate 进度条** — `TrainingRunningView.swift:70-76`。`totalSteps>0` 时进度已知,却只用文本显示「步 X/Y、百分比」。progress-indicators.md:「Prefer determinate — 帮助用户决定等待/暂停/放弃」。应加 `ProgressView(value: progress)`。

**[Major] loss 图多序列缺图例** — `TrainingChartView.swift:56-86`。Raw / EMA 同为 `accentColor`,仅靠不透明度+线宽区分,无 legend。charts.md:「Don't rely solely on color」+ legend 描述颜色/形状类别。内存图 RSS/Swap 同问题(:143-161)。

**Minor(12 项):** 终态标题层级不一致(`largeTitle` vs `title` vs `headline`,TrainingPanel.swift:110/127/155);cancelledView 两按钮尺寸不一致(:168-174,应靠 style 非 size 区分);终态恢复动作缺 `.defaultAction`;有界整数字段用裸 `TextField` 而非 `Stepper`(TrainingConfigView.swift:228,254);多字段仅靠 placeholder 兼作标签(:142-164);lr 警告与被折叠隐藏的 lr 字段分离(:39-44);「返回」用文字未用标准符号且未进菜单栏(TrainingPanel.swift:57);疑似无效符号 `chart.line.downtrend`(RunningView:82)与 `progress.indicator`(RecentRunsView:66);完成态按钮尺寸不一(TrainingDoneView.swift:36-60);图表单一 `.accessibilityLabel` 压制逐点无障碍元素(:138,187);「恢复默认」borderless 图标命中区可能 <28pt(:243,269)。

**Nit(部分):** failed/cancelled 的 hero 符号未 `.accessibilityHidden`;裸 CLI 字段中文括注不齐;dropZone 无障碍不可达;取消按钮在窗口底部(有 Esc 兜底);平滑滑块缺冒号引导标签与 min/max 刻度;RecentRuns 空态分支为死代码。

**做得好:** 取消训练的破坏性确认(destructive role + `.cancelAction` + 精确后果说明)是范本;状态从不单靠颜色(图标+文字+色);空态给明确下一步+拖拽/点击双通道;主动线 prominent + 禁用逻辑 + `⌘Return`;语义色贯穿;决定性数值 `.monospacedDigit()`、文件名 `.textSelection`;实时图表关闭动画符合 motion.md。

小计:Blocker 0 / Major 3 / Minor 12 / Nit ~10。

---

## 页面 2:对话 (Chat)

`Views/Chat/` 全部子视图 + sampler sheet。

**[Major] 自动滚动覆盖读者** — `ChatPanel.swift:210-223`。每次 `messages.count` 变化和每个流式增量都无条件 `scrollTo(.bottom)`。用户上滑读历史时会被强拉回底部。scroll-views.md:「Automatic scrolling: only as much as necessary」。应仅在已贴近底部时自动滚动。

**[Major] 会话中途错误不可见** — `ChatPanel.swift:243` vs `ChatStore.swift:356,384`。`lastError` 只在 `emptyState` 里渲染,而 busy/发送失败发生时消息已存在、emptyState 已消失,错误永远看不到。feedback.md:「Explain when a command fails」。应在输入栏上方内联横幅常驻显示。

**[Major] 启动日志背景硬编码** — `StartupLogSheet.swift:97` `Color.black.opacity(0.06)`。dark-mode.md/color.md:「Never hard-code color values」。深色模式下近乎不可见,也做不出注释里想要的「黑底终端」。改用语义背景或标准材质。

**[Major] 无生成式 AI 披露 / 准确性提示** — 整个面板把模型输出当普通对话呈现。generative-ai.md:「Disclose AI use」+「Communicate that outputs may contain errors」。应加一次性说明或常驻弱提示。

**Minor(16 项,择要):** 输入框无 placeholder 提示(ChatInputBar.swift:27-33);多行 `TextEditor` 里裸 Return 提交违反 macOS 文本视图惯例(应 Return 换行 + `⌘Return` 发送,:40-47);图标按钮 30×30 低于 44pt 命中区(:53,62,71);生成中禁用输入影响连续输入(:35);技术摘要/「思考」标签处 10pt + secondary/tertiary 对比临界(ChatMessageView.swift:67,85);startup sheet 不可缩放(:33)+ 就绪即自动消失使成功态一闪而过(:34-39);空态无 Connect 动作按钮;「选 state」打开 NSOpenPanel 却缺省略号(:110);无 Retry/Regenerate 与反馈(赞/踩)控件。

**做得好:** 气泡靠位置+颜色双通道区分(非单色);清除会话有 destructive 确认且文案说明不可撤销;按钮 role/style 正确(primary=prominent、abort=destructive、apply=`.defaultAction`);每个图标按钮都有 `.help` tooltip;启动用不定式 spinner + 具体状态文案 + 失败保留日志与重试;多行输入用 `TextEditor` 且有高度区间与 2000 字上限。

小计:Blocker 0 / Major 4 / Minor 16 / Nit 2。

---

## 页面 3:训练记录 (History)

`Views/History/` 三视图 + `HSplitView` 主从布局。

**[Major] 零记录时列表空白且右侧提示自相矛盾** — `TrainingHistoryView.swift:31,55-59`。无记录时左栏空白,右栏却显示「选择一条训练记录」——但根本无可选项。loading.md:「A blank screen signals a broken app」;writing.md:空态需明确下一步+动作。应在无记录时显示「还没有训练记录」并给行动入口;筛选无结果时显示「没有符合该状态的记录」。

**Minor(13 项,择要):** 导出进行中始终显示绿色 `checkmark.circle`,「正在导出…」期间暗示已成功(TrainingRunDetailView.swift:62-66,190),应改 spinner;「登记 State」/「导出…」/「导出 .pth」打开面板缺省略号(:26,169,189);`+` 图标命中区可能 <28pt;行内元信息 `.caption2`(10pt)处字号下限;loss/内存图缺图例;交互图缺逐点无障碍;`.animation(nil)` 关闭图表切换过渡;数值 `String(format:)` 未走本地化数字管线;事件时间戳只有时间无日期(EventLogView.swift:39);类型筛选无空结果态;无删除/管理记录途径;头部命令未进菜单栏。

**Nit:** 列表状态图标缺 a11y 标签;详情内嵌套多层滚动区域;筛选用「全部」魔法字符串当哨兵。

**做得好:** 状态三通道编码(颜色+独立 SF Symbol 形状+文字);全程动态系统色自动适配深色;标识/路径/日志用等宽字体且 `.textSelection`;日期走本地化 `FormatStyle`;`HSplitView` 持久选中 + 合理 min/max 栏宽;`ContentUnavailableView` 用法规范;`DisclosureGroup` 默认折叠 stderr 且标签描述清晰;错误 alert 标题具体非「错误」。

小计:Blocker 0 / Major 1 / Minor 13 / Nit 3。

---

## 页面 4:工具箱 (Toolbox)

`Views/Toolbox/ToolboxView.swift`,首页 System-Settings 风格分组列表 + 三工具(模型转换 / 数据集预览 / 数据集转换)。

**[Major] 数据集预览正文仅靠蓝/绿区分「输入前缀」与「训练目标」** — `ToolboxView.swift:453-454` `.foregroundColor(.blue)+.foregroundColor(.green)`。三重问题:① 色盲无法分辨边界(顶部图例有文字但正文无非颜色标记);② 系统 green 13pt 正文在 `.quaternary.opacity(0.35)` 浅底上对比度远低于 4.5:1;③ `Text(prefix)+Text(target)` 拼接使 VoiceOver 连读无边界。accessibility.md/color.md:「Never rely on color alone」。应加背景高亮块/分段标签,并加 `.accessibilityLabel`。

**Minor(8 项):** 主操作(开始转换/检查数据集/转换并保存)缺 Return/⌘ 快捷键(:244-255);拖拽区无落点高亮反馈(:662);手动字段映射两 `TextField` 仅 placeholder 作标签且并排(:407-408);同一 turn-policy 在预览页与转换页文案不一致(:297 vs :544);`Stepper "ctx …"` 用 jargon 缩写(:305);警告指标仅靠橙色无图标(:388-393);内容层操作未进菜单栏;`toolPathRow` detail 10pt + tertiary 对比风险(:648-650);`.foregroundColor` 为过时非语义 API。

**Nit:** `toolRow` 的 `tint` 参数是死代码(:154-167,传入却从未使用)。

**做得好:** 覆盖操作 `confirmationDialog` + destructive/cancel 角色 + 明确后果,教科书级;首页 `Form.grouped` + `chevron.right` 下钻贴合系统设置;进度 determinate 优先 + 可取消 + 文案简洁;打开面板按钮带省略号;全量语义色无硬编码;icon-only 按钮全配 `.help`;高级选项 `DisclosureGroup`;Pop-up 有合理默认「BF16(推荐)」+ 可见标签;路径 `.truncationMode(.middle)`。

小计:Blocker 0 / Major 1 / Minor 8 / Nit 1。

---

## 共享外壳 (容器 / 侧边栏 / 状态栏 / 模型选择器)

`ContentView.swift`、`Sidebar.swift`、`GlobalStatusBar.swift`、`BackendStatusView.swift`。

**[Major] 「需要模型」空态指向已失效的旧位置** — `ContentView.swift:98-140`。文案「请先在**侧边栏底部**选择模型」,但模型选择器已移到 toolbar primaryAction(见文件头注释与 :24-28)。writing.md:「Be accurate — 指令必须与实际 UI 一致」。用户按提示去侧边栏底部找不到控件。应改为「请在窗口右上角选择模型」,或直接在该空态内放一个「选择模型…」按钮(writing.md:空态需带动作)。

**Minor:** 状态栏 `info.circle`「关于」为 borderless caption 图标,命中区可能 <28×28pt(GlobalStatusBar.swift:45-52,accessibility.md macOS 默认 28×28);状态栏与「关于/后端状态」入口全部集中在窗口底部,layout.md macOS:「Avoid placing controls at the bottom of windows」(状态栏本身放底部是常规,但唯一的关于/后端入口只在此处、可发现性弱);后端状态经状态簇点击打开 sheet、其内「查看日志」再开二级 sheet(BackendStatusView.swift:46),嵌套 sheet 偏深;`BackendStatusView` 的 Python 版本/平台信息未 `.textSelection`(:29-30,排障场景 labels.md 建议可选中);全 app 命令(关于、后端检查、模型选择)未在菜单栏提供,削弱键盘工作流(designing-for-macos.md)。

**做得好:** `GlobalStatusBar` 用 `.accessibilityElement(children:.combine)` + 合并标签;健康状态点始终配文字标签(非单色,healthLabel);内存压力同时用文字+色;`NavigationSplitView` 栏宽 `min200/ideal220/max280`、窗口 `minWidth1000/minHeight680` 合理;模型选择器菜单选中项打勾、「选择其他模型…」带省略号、`.help` 齐全;侧边栏四项扁平导航 + 语义化 SF Symbols;`BackendStatusView`「完成」按钮 `.defaultAction`、状态行点+文字双通道、状态文本可选中。

小计:Blocker 0 / Major 1 / Minor ~5 / Nit 0。

---

## 全局汇总

| 页面 | Blocker | Major | Minor | Nit |
|---|---|---|---|---|
| 1 训练 | 0 | 3 | 12 | ~10 |
| 2 对话 | 0 | 4 | 16 | 2 |
| 3 训练记录 | 0 | 1 | 13 | 3 |
| 4 工具箱 | 0 | 1 | 8 | 1 |
| 共享外壳 | 0 | 1 | ~5 | 0 |
| **合计** | **0** | **10** | **~54** | **~16** |

### 跨页复现的系统性问题(建议统一整改,一次修复多页收益)

1. **仅靠颜色传达信息** — 数据集预览前缀/目标(页4 Major)、图表多序列(页1/3 Major/Minor)、警告指标(页4)。加形状/图标/标签第二通道 + 校验对比度。
2. **打开面板/窗口的按钮缺尾部省略号** — 页1「选择」、页3「登记 State/导出」、页2「选 state」。统一补 `…`。
3. **内容层命令未进菜单栏、缺键盘快捷键** — 四页普遍。designing-for-macos.md 要求所有命令进菜单栏并支持纯键盘工作流。
4. **空态文案与实际 UI/数据不符** — 共享外壳指向失效的「侧边栏底部」(Major)、页3 零记录矛盾提示(Major)、页2 空态无动作按钮。
5. **10pt(caption2)+ secondary/tertiary 低对比** — 页2/3/4 多处贴 accessibility.md 4.5:1 边界。
6. **可疑/过时 API** — `chart.line.downtrend`、`progress.indicator` 疑似无效符号;`.foregroundColor` 应换 `.foregroundStyle`。

### 优先级建议

先清 10 个 Major(尤其页4 预览可访问性、页2 自动滚动/错误可见性/AI 披露、两处空态文案),再按上述 6 类系统性问题批量整改 Minor。整体没有 Blocker,基础架构(语义色、破坏性确认流、状态双通道、Liquid Glass 版本门控)扎实,问题集中在可访问性细节、空态/错误反馈完整性与 macOS 键盘/菜单栏惯例。

