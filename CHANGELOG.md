# 更新日志

> 记录首个预览版 [v0.1.0-beta.1](https://github.com/No-22-Github/Preen/releases/tag/v0.1.0-beta.1) 之后的变更。
> 格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/),版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
>
> **写入规矩**(固化在 AGENTS.md):有 commit 就记;一连串同主题改动(如纯 UI 体验)合并写一行即可;涉及 API / 后端 / 协议 / 数据格式的改动必须展开说明。

---

## [未发布]

### 新增

- **会话格式成为显式配置**：对话参数面板新增 QA / Instruction / Raw、Reasoning 与 Off / Fast / On 设置，toolbar 持续显示当前口径；新会话会把格式与七个采样字段完整下发给 `serve`，默认保持 `qa + reasoning=false + think=off`，非法组合在界面禁用且后端继续校验。有历史时更改格式会先确认，取消不会污染当前设置；think=on 继续按后端 `phase / thinking / answer` 分段展示。
- **训练配置与 State 元数据同源进入验证会话**：训练完成页与记录入口现在传递不可变训练配置，外部 State 则读取同目录 `.meta.json`，来源优先级为训练记录 → metadata → App 默认；用户明确调整后的会话格式始终优先。缺少模板时必须先选择，模型名称不同时显示当前/训练模型并允许切换或继续，最终仍由后端校验 State 层数、层号和 shape。
- **State 效果 A/B 比较**：训练完成页、训练记录和对话 toolbar 提供可见入口，共用一个输入框按“无 State 基线 → 有 State”顺序逐侧流式生成；两侧固定使用同一模板、reasoning、think、seed 与七个采样/惩罚字段，每侧独立展示进度、错误、停止原因、token 数和速度。停止会保留当前侧已生成文本、只中断当前侧并继续另一侧；用户可主动把 prompt、两侧文本、实际配置与技术摘要保存到对应 run，重开 App 后仍可在记录详情查看，默认不会保存自然语言内容。`serve` 协议版本同步升级至 v3，`preview` 的 `text_chunk` / `turn_end` 带 `side`，非终结 `side_error{code=aborted}` 表示当前侧被中断；服务端清除该侧 abort 标记后继续下一侧，整次请求仍严格只有一个终结 `ok` / `error`。单路 `send` / `preview` 的 abort 语义不变。
- **会话替换统一为 App 级安全事务**：加载/卸下 State、更改 template/reasoning/think、切换模型、断开后端，以及从训练完成页或记录打开比较，全部先形成包含目标模型、State、配置和落地页的意图；有消息、A/B 内容或生成中才显示同一确认，取消不会改变消息、模型、State、页面或生成。确认后先停止生成再只重建一次会话，失败进入明确断开态并保留诊断错误，不伪装回滚旧 cache；State 在确认前通过 `state-info` 只读检查路径、格式、层号连续性与 RWKV-7 shape。
- **内置 NekoQA 200 首次训练**：App 随包提供经人工与自动审查的固定 200 条角色风格 QA 子集、版本化 manifest、源索引、SHA-256、Apache-2.0 LICENSE 与归属 NOTICE；欢迎页和训练页可在已有 BF16 模型时直接进入配置，无需选择数据文件，INT8/未选模型会先给出 BF16 引导。内置数据固定使用 `qa` 与 `ctx_len=512` 的语义默认值并复用现有训练参数和同源预检，运行记录新增 `dataset_source=builtin:nekoqa_200`、子集版本与 SHA；界面明确说明示例用于学习角色与表达风格、不用于学习新知识，用户仍可随时切回自己的数据。
- **训练输出路径自动生成**：选择模型与数据后，App 默认在 `~/Library/Application Support/Preen/states/<数据>-<模型>-<时间>/state.npz` 生成带“自动”标记且不冲突的路径，模型或数据变化时自动更新，用户经“更改…”选择后则保持手动路径；同目录关联 `.meta.json` 与可选 `.pth`。启动前按模型层数、head 数和维度估算实际 fp32 State/PTH 及原子写入空间，并检查目录可写性、剩余容量和全部关联目标冲突；Python `validate_training_request` 同步改为在模型加载前拒绝覆盖已有 State/metadata/PTH，PTH 先写临时文件并完成 round-trip 校验后再原子提交。
- **训练前数据预检与实际 loader 同源**：训练配置页现在通过 `dataset-preview --training-data-route` 按最终模型 tokenizer、模板、`ctx_len` 和 importer sidecar 全量渲染，默认展示 schema/置信度、有效数、token 分布、互斥的部分/完全截断、训练/验证与步数，并可展开 3 条结构化 prefix/target。数据、tokenizer、模板、上下文和映射共同形成工具箱共享缓存键；10K 以内自动重算，更大数据必须手动完成最终配置的完整检查。未知格式会带原路径进入数据导入器，转换完成后自动回到训练页选择标准产物；检查失败、配置过期或丢弃后样本归零均不会启动训练。
- **可追溯的训练结果摘要**：完成页、取消/失败页与训练记录统一展示实际轮数/配置上限、第 1 轮平均训练 loss 到最终轮的相对变化、最佳 held-out loss 及轮次、客观结束原因、未着色的 State std、总耗时，以及训练/验证样本、模板、ctx、截断/丢弃、模型精度和 `data_sha256` 短摘要；无验证集或首轮未完成时明确隐藏不存在的结论。主操作直达使用本次模型/模板/State 的效果比较，次操作打开 Finder，菜单提供 PTH 导出、完整曲线和诊断复制。训练事件协议新增 `data_summary`，metadata v2 的 `data_stats`/`result` 向后兼容地补充最终拆分数量、丢弃数与最佳验证轮次，供历史记录复算而不引入评分或未标定阈值。

### 变更

- **macOS 27 界面与交互适配**：训练首页降低内置示例的视觉权重，训练预检压缩为三行摘要，模板预览与记录日志改为整行可展开；统一整数 Stepper 与数值框对齐、工具箱四个工具的固定底部操作区，并将“开始训练”恢复到配置页右下角；训练结果详情改为左对齐且移除卡片背景，完成页垂直居中。运行状态使用进度指示，诊断/训练日志增加原生容器并将误导性的“错误日志”更名为“运行日志”；训练图悬停越过实际进度时固定到最新点，训练期间禁止切换模型；模型加载完成反馈改为整块淡绿气泡，避免系统裁切外扩描边。
- **State metadata 升级为向后兼容的 v2 契约**：新文件新增 `model_name`、`model_path`、`state_format=npz` 与 `state_dtype=float32`，保留 v1 `model` 别名供旧 App 读取；Swift 可解码字段不完整的旧 v1 与最小 v2 文件，不会因缺少非必需训练摘要而拒绝登记。

- **模型转换改为 mmap 流式读写,并以完整目录为单位安全提交**:`model_converter.convert` 此前通过 `read_pth` 全量加载源 storage,再构建完整目标权重 dict,转换 1.5B 模型峰值约 6GB。现改为:
  - 新增 `pth_io.peek_pth_tensors` 与 `iter_pth`:前者只读 `data.pkl` 和 ZIP 元数据,预先校验 storage 大小、tensor offset/stride、配置键、映射与 shape;后者把未压缩 storage 只读 `mmap` 为 numpy 视图,按目标键顺序逐 tensor 访问。1.5B 完整转换实测物理内存峰值降至约 0.65GB;`read_pth` / `write_pth` 既有 API 不变。
  - safetensors writer 由预检 manifest 先生成 header,再用零拷贝 byte view 直接写入单个 `model.safetensors`,不再生成第二份权重中转文件。单一精度产物与官方 `save_file` 逐字节一致;转换临时磁盘需求从旧流式草案的约两份输出权重降为约一份,启动前会检查可用空间。
  - 权重、配置与 tokenizer 先写入输出目录同级 staging 并通过 safetensors 校验,全部成功后才整体提交。失败或取消会清理 staging 且不改动旧模型;`--overwrite` 会完整替换旧目录,不再残留历史分片或 index。
  - mmap 路径仅支持 RWKV 官方采用的 `ZIP_STORED` 未压缩 `.pth`;压缩 storage 会在创建 staging 前明确拒绝,避免静默回退到高内存全量加载。

### 修复

- **训练前数据预检界面卡死(P0-02/05/07/08 验收阻塞)**:`TrainingDataPreflightRunner` 把 `ToolJobRunner` 作为局部变量创建,函数返回 stream 后 runner 立即被 ARC 释放,触发 `ToolJobRunner.deinit` 调用 `process.terminate()` 杀掉 Python 子进程,且 `readTask` 的 `[weak self]` 提前退出后 `continuation.finish()` 永远不会被调用,调用方 `for await event in stream` 死等。表现是:在训练配置页选择内置 NekoQA 200 或外部数据时,界面一直显示"正在按最终模板检查全部样本…"且永远不进入摘要/预览,导致训练无法启动,P0-05/07/08 无法验收。改为把 runner 提升为 `TrainingDataPreflightRunner` 的实例属性,函数体内 `defer { self.runner = nil }` 保证生命周期覆盖整个 `for await` 循环,完成后立即释放。CLI 命令本身实测 0.7s 完成,问题完全在 Swift 端。
- **A/B 对比视图从横排变成竖排(P0-03)**:`comparisonContent` 用 `ViewThatFits` 在横排 HStack 与竖排 ScrollView 之间二选一,但 ViewThatFits 的 "fits" 判据是各分支的 ideal size,而 pane 内 `ChatMessageView` 在 State 侧流式文本变长时会撑大 ideal 宽度,触发内容驱动的降级 —— 用户看到的现象(基线生成完后左右分栏消失、变成两条上下堆叠的聊天记录、第二条左侧多出蓝色竖线)正是竖排 fallback 渲染。改为用 `GeometryReader` 按容器宽度(阈值 1100pt)显式选择横排或竖排,且每栏强制 `frame(maxWidth: .infinity)` 防止内容撑宽再次触发切换;窄窗时仍允许上下堆叠,右栏的轻量 accent 边界保留(PRD §四)。

### 变更

- **RAW 模板改为整页纯续写界面(P0-01)**:RAW 模板对应"模型从给定前缀往后直接续写"的纯续写语义,没有 User/Assistant 包装,是 RWKV 这类因果语言模型最基础的用法。新会话模板选 `raw` 时,对话页主体从聊天气泡列表切换为上半部分大 `TextEditor`(前缀文本,占满主体)+ 下半部分只读"模型续写"区(实时流式追加),底部按钮为「续写 / 停止 / 清空续写 / 采纳续写」。续写触发 `store.send(text:)`,store 与后端不变(已支持 RAW template);"采纳续写"会把模型生成结果拼到前缀文本末尾,可继续往后续写。切回 QA/Instruction 自动恢复聊天气泡列表。
- **会话替换确认弹窗加「本次运行内不再提醒」(P0-04)**:原 `.confirmationDialog` 不支持复选框(macOS 限制),改为自定义 `.sheet` 弹窗,内含标题、后果文案、「本次运行内不再确认会话替换」复选框与取消/确认按钮。勾选后本次 App 运行期间(进程生命周期内)切换模型、加载/卸下 State、更改模板等所有会话替换动作直接执行,不再弹此 sheet;重启 App 自动重置为需要确认。PRD §七「不增加永久不再提醒」的边界由"仅本次运行"维持,不写入持久化偏好。
- **对话页 toolbar 显示会话配置来源(P0-03 配套)**:当会话配置来源(`sessionConfigSource`)非 App 默认时,在 toolbar 的格式 chip 旁显示「建议来自训练记录」/「用户已调整」/「建议来自 State 元数据」标签。这让用户能直接看到为什么从 RAW 训练记录进入对话后没有自动切回 QA —— 这是 PRD P0-02 §六的设计行为("用户手动修改配置并确认后,训练记录或 metadata 不会再自动改写该选择"),不是 bug。
- **多处界面回归 macOS 原生风格**:训练前数据检查、训练结果摘要等卡片改用原生 `GroupBox` 与 `LabeledContent` 行,去除自定义 `RoundedRectangle` + `.quaternary.opacity` 背景块、`Color.blue/green/orange/yellow.opacity(0.08)` 等装饰性色块与数值着色;指标改为原生右对齐标签 + secondary 文案 + SF Symbol(不再靠颜色区分截断严重度),预览样本用 `DisclosureGroup` + `Divider` 分段。训练记录详情中间卡片精简为只展示训练结果叙事(结束原因 / loss 变化 / 最佳轮次 / State std / 耗时),数据来源、模板、模型、SHA-256、训练参数等结构化字段统一收敛到右侧 inspector,消除中间与侧栏的内容重复。输出 State 路径旁的「自动」徽标移除(自动模式本就是默认预期,无需视觉标注)。A/B 对比横排/竖排切换阈值从 1100pt 降为 900pt,适配 13" MacBook Air 默认缩放下的窗口宽度。
- **高步数训练图表性能优化**:`TrainingChartView` 原本每次 `body` 都全量重算 loss EMA、内存 EMA 与两个压力梯度(每次刷新 O(N)),训练每个 step 追加一个点会触发一次刷新,万步训练累计成本约 O(N²)=1 亿次操作,且 hover 期间每帧重算导致明显卡顿。改为按指纹(`lossPoints.count + smoothing + 末尾 step`、`processMetrics.count + totalSteps + capacity`)缓存派生量,`.onChange` 指纹变化时才重算,缓存命中时 `body` 内 O(1) 返回;hover 期间的频繁重绘不再触发任何 O(N) 计算。`chartHoverTracking` 同步去掉每次 `body` 都重建的 `selectableSteps: [Int]` 数组,选中步直接 clamp 到 X 轴 domain(每步都有数据,clamp 与二分等价),避免随数据增长分配数组。
- **训练图表 hover 卡顿根治**:前一轮 EMA 缓存解决了训练事件时的重算,但 hover 拖动仍掉帧 —— 真热点是 `Chart { }` builder 闭包依赖了 `selectedPoint`,hover 时 `IndependentChartSelection.selection` 变化触发子树重建 → `Chart { ForEach(900+) { LineMark… } }` 重新构造所有 mark(每个 mark 构造有固定开销,900 点 × 3 条线 × 每帧 = 数千次构造/帧)。重构为 Swift Charts 官方推荐的"静态 Chart + chartOverlay 选中可视化"模式:三张 Chart(loss / learning rate / memory)的 builder 闭包完全不引用 `selection`,选中竖线、圆点与 annotation label 改由新的 `ChartHoverOverlay` 用 `ChartProxy.position(forX:forY:)` 转屏幕坐标后在 `.chartOverlay` 里画原生 SwiftUI 视图。hover 时只有轻量 overlay 重建,Chart 本体保持稳定;删除已无人使用的 `chartHoverTracking` ViewModifier。

### 修复(2026-07-19,HIG 一致性整改)

本轮为对照 Apple Human Interface Guidelines 的全方位审查后的整改,聚焦 Mac 原生交互可达性与可访问性。

- **新增「前往 / 训练 / 对话 / 模型 / 工具」五个 app 菜单**:之前 `.commands` 只注册了「关于」与「欢迎」两项,引用了仓内不存在的 `InspectorCommands()`,工具栏动作也没有菜单等价物。现在面板切换支持 ⌘1–4,开始/停止训练支持 ⇧⌘N / ⌘.,连接/断开、A/B、加载/卸下 State、停止生成等都有菜单入口;工具菜单的「诊断日志…」与后端状态页一致。同时新增标准 ⌘, `Settings` 场景,当前承载「启动时显示欢迎窗口」一项。
- **诊断日志从 sheet 改为独立窗口**:HIG 指出"repeated input-and-observe workflows 应使用 panel/window 而非 sheet"。日志是训练/推理过程中需要持续 tail 的视图,原 `.sheet` 阻挡父窗口且无法并排观察。新增 `Window("诊断日志", id: "backend-logs")`,后端状态页的「诊断日志…」按钮改用 `openWindow`。
- **会话中后端进程退出不再静默退场**:`ChatStore.handleServeExit` 原本在已连接会话期间进程崩溃时 `guard !wasConnected else { return }` 早退,既不设 `lastError` 也不留消息,用户看到对话凭空消失回空状态。现在 mid-session 死亡会写入「推理进程意外退出,请重新连接」并保留 `messages` 供用户参考上下文(HIG Feedback: "Explain when a command fails")。
- **对话面板补全 VoiceOver 标注**:`ChatInputBar` 的清除/发送/停止按钮、`ChatMessageView` 的每个气泡(区分「你的消息」/「助手消息」/「(已中断)」/「(出错)」并携带完整文本作为 `accessibilityValue`)、「回到最新消息」按钮、错误 banner 全部加上 `accessibilityLabel`,生成中的助手气泡追加 `accessibilityHint("正在生成")`。此前 `Views/Chat/` 目录零 accessibility 标注,其它面板共有 27 处。
- **单 Pane 对话显示「思考中…」占位**:助手占位气泡在收到第一个 token 前是空的(只靠发送按钮变形为停止按钮暗示),A/B 与 RAW 模式却都有 `ProgressView`。现在最后一条 assistant 消息生成中且文本为空时,在气泡内渲染 `ProgressView().controlSize(.small) + "思考中…"`,与 A/B、RAW 一致。
- **会话输入栏回到 macOS 多行编辑惯例**:原 `onKeyPress(.return)` 在纯 Return(无 modifiers)时直接发送,Shift+Return 才换行 —— 这是 iOS 惯例,macOS 多行 TextEditor 的惯例是 Return 换行、⌘⏎ 发送(后者本就存在)。删除纯 Return 拦截,同时为空文本加 `输入消息…` 占位符 overlay。
- **训练参数行加 Stepper**:`TrainingIntParameterRow` 原本对 epochs / ctx_len / warmup / log_every / patience / checkpoint_every 这些有界整数全部用自由 `TextField`。现新增可选 `range` 参数并附 `Stepper`(seed 范围不固定保持纯 TextField),HIG macOS: "Stepper for bounded numerics"。
- **Toolbox 成功态的「设为当前模型」降级为 `.bordered`**:`modelConversionView` 与 `modelQuantizationView` 在结果区出现该按钮时与底部 footer 主操作同时为 `.borderedProminent`,违反"1-2 prominent/视图"。改为 `.bordered`,让 footer 主操作保持唯一 prominent。
- **两处主操作不再叠 `.tint(.orange)`**:`SessionReplacementConfirmationSheet` 的确认按钮已是 `role: .destructive`,再叠橙色覆盖了系统红;`ChatRawContinuationView` 的「停止」按钮用 prominent + orange 而非语义 destructive。前者删 tint、后者改 `Button(role: .destructive)` + `.borderedProminent`,均回归系统语义色。
- **错误 banner × 关闭按钮扩展到 44pt 命中区**:可视区保持 28pt,通过 `.padding(8)` + `.contentShape(Rectangle())` 把命中区扩到 44pt(macOS 最小命中尺寸)。

## [1.0.0] - 2026-07-16

首个正式版。以下为 [v0.1.0-beta.1](https://github.com/No-22-Github/Preen/releases/tag/v0.1.0-beta.1) 之后的全部变更。

### 新增

- **中英文界面支持**:macOS App 新增完整英文与简体中文本地化资源,语言选择和未支持语言的 fallback 完全交由 Apple 系统规则处理;动态参数、工具进度、错误提示、辅助功能标签与历史记录同样走本地化,中文环境不会直接暴露生硬的英文后端文案。
- 训练统计图新增实时 Learning Rate 曲线与 warmup 完成标记,复用逐步训练事件中的实际 LR;实时训练时 LR 与进程内存并排展示,三张图随窗口高度自适应且 Loss 获得更大的展示区域,Loss 纵轴按 Raw、EMA 与 Held-out 的实时范围自动缩放并保留上下留白,矮窗口下图表可滚动且取消训练底栏保持可见;步数轴采用一基整齐刻度并强制显示任务总步数,三张图的鼠标悬停互不联动,仿 macOS"股市"以竖线和沿 EMA 主线移动的圆点呈现选中状态,顶部按 Raw、EMA 顺序同时显示读数,不再显示悬停横线;内存图移除压力区间图例和黄色阈值线,仅保留红色严重阈值虚线;训练记录详情可重放旧任务的历史 LR 曲线。
- **后端环境诊断与 Issue 信息复制**:环境窗口采用 macOS 原生 Form、Section 与 LabeledContent 呈现整体健康状态、芯片与系统版本、设备内存和 MLX 工作集上限,检查时间以自动更新的相对时间显示,说明文字按系统设置习惯置于左侧标签下方,组件版本标题整行均可点击展开且正常组件不重复显示状态图标,底部操作采用克制的纯文字按钮;诊断日志按来源提供居中的原生空状态与对应说明,有日志时支持等宽显示和文本选择;新增一键复制脱敏 Markdown 诊断摘要,包含 Preen/macOS/Python/MLX 环境及硬件标识与安全状态枚举,不自动包含序列号、硬件 UUID、日志、PID 或本地路径。
- 新增 `tools/bench_bandwidth.py` MLX decode roofline 探针，以约 2.1GB 随机 bf16 权重分别测量大 GEMV 实效带宽与 sum reduction 上限，并据此估算 1.5B bf16 的理论 ms/token 地板。

### 变更

- **Python 用户文案统一为英文**:`statetuner` 的 CLI 帮助、交互式对话、校验错误、训练/工具事件与 serve 协议错误统一输出英文,便于跨语言环境诊断与集成;App 对结构化后端事件按当前系统语言呈现本地化摘要,英文原始诊断仍保留在日志中。
- **训练路径展示与默认训练长度调整**:调参页的训练数据和输出 State 改为显示完整路径,空间不足时从中间截断并可悬停查看;Swift 新建任务、参数复位、Python `TrainConfig` 与 CLI 缺省值统一从 `20 epochs / 10 warmup` 调整为 `5 epochs / 50 warmup`,减少默认训练轮数并让学习率升温更平缓。已有记录及显式传参不受影响。
- **WKV7 训练图与数据通路继续提速**:默认 Metal fast path 将 masked loss + backward 通过 `mx.compile` 按输入形状复用,训练步数、学习率、梯度裁剪、Adam 更新和事件进度口径不变;checkpoint kernel 改为直接读取 bf16 模型激活,仅在 Metal kernel 内提升为 float 累加,state/checkpoint 继续保持 fp32。1.5B × 320 token × 400 步实测编译阶段 277.4s → 239.6s(-13.6%),末态 loss/std 偏差分别 -0.21%/+0.54%;bf16 直读交替 A/B 再快 2.4%,完整训练末态 loss/std 相对编译基线偏差 +0.39%/+0.45%。`--no-fast-wkv` 的 ops 排查路径仍保持 eager。
- **训练后台反馈**:训练期间 Dock 改用 macOS 原生红色百分比角标,并在训练开始时注册系统通知的“标记”能力,确保该开关和百分比角标可用;收到 State 已落盘的 `completed` 事件后清除角标并发送系统通知,App 位于前台时也会显示通知横幅。失败或取消仍会清除角标并发送对应通知。
- **训练默认学习率调整**:Swift 配置与 Python CLI/`TrainConfig` 的峰值学习率从 `0.01` 调整为 `0.0001`,最低学习率从 `0.0001` 调整为 `0.00001`,并同步参数重置按钮、CLI 帮助和 smoke 脚本。仅影响新建任务及恢复默认值;显式传参、已有训练记录和产物元数据保持原值。
- **训练进程内存压力图**:进程内存改为按训练步展示的 EMA 0.90 面积图,纵轴固定为本机物理内存容量,并以 70%/85% 容量阈值结合 macOS 内存压力信号显示绿、黄、红状态;采样从“最近 3,600 个秒级点”改为“全程每步峰值”,长训练不再丢失前段视图且不会按秒无限累积。
- **内存展示与后端口径分离**:Python 后端、`doctor_report()`、CLI、事件、日志、缓存和训练/削顶判据统一只使用十进制 GB;macOS App 收到容量 GB 后自行还原 bytes,其进程内存、swap、设备容量、MLX 工作集、图表和压力比例统一改用 GiB 数值计算,但产品界面标为 `GB`(例如 16GiB 显示 `16 GB`),不再出现 `GiB` 或 `G` 标签。Swift 仅保留旧 `_gib` 字段的解码兼容。环境诊断仍只读取安全的硬件与容量白名单,不接触序列号或硬件 UUID。
- **RWKV7 推理 prefill 词表投影优化**:保留完整 bf16 RWKV 主体、state/cache 更新与采样逻辑，但只对 prompt 最后一个 hidden 做 `lm_head`，不再构造 `[prompt_len, vocab_size]` 整段 logits，以降低长 prompt 的临时内存并提高 prefill 速度；未知模型保留原前向兼容路径。
- **RWKV7 bf16 decode 整步编译与异步流水优化**:将单 token RWKV 前向建模为 `(input_token, cache_state) -> (logits, next_cache_state)` 纯函数并通过 `mx.compile` 跨请求复用；采样 token 保持在 GPU 上，先用 `mx.async_eval` 提交下一步前向，再由 CPU 读取当前 token 并处理文本/停止条件。cache 不绑定具体会话，EOS/stop 时丢弃投机 state，生成结束后写回原对象，多轮续传、State 注入、重复惩罚及采样语义不变。int8 实测无稳定编译收益，量化模型继续走 eager 路径。
  - `bench_inference.py` 新增 `--decode-backend eager|compile|pipeline`，可在完全相同的 token 间隔计时口径下分别测原始 eager、同步整步编译和异步流水，默认 `pipeline`；量化模型即使请求编译也会报告实际回退的 `eager`。
- **欢迎窗口改为模态 sheet**:此前「欢迎使用 Preen」是一个独立普通窗口,会被切到主窗口后面、且定位随意。
  现改为挂在主窗口上的模态 sheet:从主窗口顶部滑出、盖在主窗口上方居中、带背景遮罩,点击主窗口区域不响应(真正的模态锁定)。
  - 去掉了独立的 welcome `WindowGroup` scene;首启与「窗口 → 欢迎使用 Preen」菜单改为翻转 `appState.isWelcomePresented` 标志,由 `ContentView` 的 `.sheet(isPresented:)` 驱动。
  - 同一标志仍驱动侧栏收起(背景呈空状态),语义不变。点入口项 / 「开始使用」/ 点背景遮罩均可关闭。
- **WKV7 训练路径改用 Metal checkpoint kernel(默认开启)**:训练中 RWKV-7 的 WKV 递归从 Python `_wkv7_step_ops` 循环(每 token 一次 GPU dispatch)换为整段 Metal kernel(forward + backward 各一次 dispatch),通过 `mx.custom_function` 注册 VJP,可训练 S₀ 的梯度仍能穿透整段递归。1.5B 模型实测长序列(320 token)6.67× 加速(28 分钟 → 4 分钟),内存反降约 3GB,loss 末态与 ops 基线差 0.19%(数值等价)。完整实验记录见 `docs/decision-fast-wkv7.md`。
  - kernel 移植自 [rwkv-metal](https://github.com/RafaelUI/rwkv-metal)(Apache-2.0),核心改动是让 `make_wkv7_checkpoint` 工厂暴露 `h_in` 参数以透传可训练 S₀。
  - 训练样本长度不整除 chunk(16)时,闭包内就地 pad 序列末尾(算完 slice 回真实长度),因果递归保证 pad 段对结果零影响,不改动数据管线。
  - `TrainConfig` 新增 `wkv_mode`(`"metal"` 默认 / `"ops"`)和 `wkv_chunk`(默认 16)字段;CLI 新增 `--fast-wkv/--no-fast-wkv` 与 `--fast-wkv-chunk`,启动日志与 `events.start` 事件均记录当前 kernel 模式与 chunk,便于排查训练异常是否源于加速路径。`--no-fast-wkv` 回退旧的 Python ops 循环。
  - 推理路径不受影响(`load_model(patch=False)` 仍走 mlx-lm 自带 kernel,实测比上游推理 kernel 快 10%)。
- **致谢清单补登加速来源**:关于窗口「引用项目与致谢」功勋墙、README 致谢表均新增 [rwkv-metal](https://github.com/RafaelUI/rwkv-metal)(Apache-2.0,作者 RafaelUI)条目,记录 WKV7 Metal checkpoint kernel 的移植来源;README「一些取舍」里原先「训练走 ops、推理走 kernel」的论述已过时,同步改写为训练默认走可微 Metal checkpoint kernel、推理走 mlx-lm 自带 kernel。
- 关于窗口致谢墙按对 Preen 的实际贡献重排顺序(引擎地基 → 关键加速 → 模型权重 → 方法数据 → 导出校验链路 → 格式与基础设施),rwkv-metal 因提供训练 Metal kernel 加速提升至第三位。
- **同步 README 滞后的训练默认学习率**:峰值学习率从 `0.01` 调整为 `0.0001`、最低学习率调整为 `0.00001` 的变更此前已在 CLI/`TrainConfig` 与 `docs/快速上手.md` 落地,但 README 的「三步最小流程」示例仍写 `--lr 0.01`、「一些取舍」节仍按 0.01 论述。本次将示例改为 `--lr 0.0001`、取舍节改为从 1e-4 起步 + cosine 衰减口径,并补「历史实验显式传入更大 lr 仍按原配方解读」的说明,与快速上手 FAQ 对齐。
- **训练总结页「去对话」改为一键启动**:此前点「去对话」只是切到对话页,若后端未连接还需用户手动选模型、连接、加载 State。现改为:点「去对话」自动切换到训练所用的模型(若与当前全局选中模型不同)、启动推理后端并加载训练产物 State;会话真正就绪后顶部模型 chip 短暂变绿再复原,悬停可核对模型与 State 完整路径,不再用绿色横幅打断内容区。全局状态栏的推理指示同步显示当前模型与 State 摘要,历史记录详情页复用同一一键流程。
- **训练总结页「再训一个」改为「返回首页」**:原按钮仅重置训练状态停留在训练面板;现改为返回训练面板空态落地页(选数据 / 最近记录),语义更清晰。
- 优化训练完成页视觉层级:移除灰色产物卡和分散的 Finder 按钮,改为轻量双列摘要,集中展示 held-out loss 变化、数据、模型精度、State、实际轮数与耗时,并突出「去对话」主操作;摘要行字号与内边距对齐其他终态页(失败/取消页),主操作按钮改用 macOS 原生圆角矩形样式并收紧按钮间距。

### 修复

- 补齐工具箱数据预览动态标题及 VoiceOver 标签的英文显示，并让运行时检查、数据预检、State 导出与推理启动失败在中文界面显示本地化摘要、英文界面保留原始诊断。
- 修复训练进程内存图的悬停圆点与读数错误跟随原始 RSS、没有落在 EMA 平滑曲线上的问题。
- **Metal 训练反向梯度不稳定**:修复 WKV7 checkpoint backward 在相邻时间步复用 threadgroup shared arrays 时缺少同步屏障的问题；此前较快线程可能提前覆盖 `w_sh/a_sh` 等当前步仍在读取的数据，使相同输入的 S₀ 梯度随 GPU 调度漂移，`mx.compile` 与 bf16 直读优化会改变调度并放大为训练后 State 重复或胡言乱语。现补齐时间步边界 barrier，保留训练图编译与 bf16 激活直读加速，并增加 eager/compiled 重复 backward 的逐元素确定性回归测试。受影响版本生成的 State 需要重新训练。
- **调参面板训练步数预估偏高**:开启早停时,面板预估用「有效条数 × 轮数」(200 条 × 2 轮 = 400 步),但实际训练会先按验证集比例(默认 10%)划出 held-out 验证集,训练子集只有 180 条,实际 `total_steps` 是 360 步 —— 面板多算了那 20 条验证样本 × 2 轮。
  - 预估改为对齐 Python `data.train_test_split` 的 `max(1, int(n * ratio))` 公式:先扣验证集再算步数,早停关闭时才用全量。文案从「N 条有效 · 预计 ~M 步」改为「N 条训练 · K 条验证 · 预计 ~M 步」,让训练 / 验证划分对用户可见。
  - 抽出可测纯函数 `TrainingConfig.projectedCounts(...)` 锁定跨语言公式对齐,并删掉 `DataInspectionRunner` 里无人调用却写错口径(同样没扣 held-out)的 `estimatedSteps` 旧 helper。
- **推理速度 benchmark 口径失真**:
  - `GenerationResult` 新增 `decode_steps`, `generation_tps` 改按 `generation_time` 实际覆盖的 `step>0` 前向次数计算,不再把归入 prefill 的首 token 重复计入 decode 分子; 分段计时改用单调高精度时钟。
  - 异步流水启用后，`generation_time` 统一按“首 token 就绪到最后一个 decode token 就绪”的墙钟区间统计，使 BF16 流水线与 int8 eager 都反映 GPU/CPU 重叠后的实际连续出字速度，避免累加局部阻塞时间造成虚高。
  - `bench_inference.py` 改测常驻 serve 的稳态性能:正式测量前先用首档做 4 次进程级全局 warmup,每档再做 1 次 shape/allocator warmup;warmup 与正式 runs 之间不再清 allocator cache。`--slow` 在首档前也会冷却,使各档起点一致。汇总改用 p50 中位数并显示 min–max 波动范围;单档或跨档 decode 差异超过预先锁定的 3% 阈值时自动标记为不可用于性能裁决。
  - benchmark 从模型 `config.json` 读取实际精度,避免离线量化模型被误标为 bf16;运行时量化改为覆盖外层 `lm_head`,使该对照项的测量范围与标签一致。
- **训练记录面板撑宽窗口**:切到训练记录界面时,若窗口整体宽度不足会撑出窗口、显示不全。
  - inspector(参数栏)默认改为收起(换 SceneStorage key 让旧用户也生效),toolbar 按钮从纯图标改为「参数」图文。
  - 左侧记录列表从硬固定宽度 260 改为可压缩范围(200~260),窗口偏窄时自动收窄,不再撑窗。
- **对话 State 无法卸下 / 切换模型时 State 被继承**:
  - `ChatStore` 新增 `clearState()`:已连接时走后端 `set_state(nil)` 重置会话,未连接时只清本地字段。
  - 切换模型、校验移除失效模型时,自动清除当前 State 及跨面板意图(`injectedStatePath`),避免旧模型的 State 继承给新模型。
  - 对话 toolbar「加载State…」改为双态胶囊:未选 State 时显示 `folder` 加载入口;已选时显示 `doc.fill` + 文件名 + `×`(淡色圆形背景)可点卸下。
  - 加载 / 卸下 State 时若当前已有聊天记录,会弹出确认对话框(与清空会话的垃圾桶按钮同一逻辑,因切换 State 会重置会话历史)。
- **切到训练记录误改 Dock 角标 / 误弹通知**:打开应用无异常,但一切换到历史训练记录,Dock 图标角标就被改成那条记录的训练进度,已完成记录还会触发系统通知。
  - 根因:`TrainStore.consume(event:)` 一身二职——既是实时训练事件入口,又被历史详情视图拿来重放事件以绘制曲线。`consume` 内含 Dock 进度更新、完成/失败/取消通知、全局状态栏进度等实时副作用,回放时全部泄漏到外界。
  - 修复:把 `consume` 拆成纯数据装配 `applyDataOnly(event:)` 与实时副作用 `applySideEffects(for:)`;新增 `replay(events:)` 仅调前者。`TrainingRunDetailView.loadDetails` 从 `consume` 改用 `replay`,切历史记录只装配曲线数据,不碰 Dock / 通知 / 全局状态。实时训练路径行为不变。
- **训练完成后 Dock 角标卡在 100% 不消失**:停在训练总结页或切到其他页面,Dock 图标一直显示 100%,只有手动新建训练才会清掉。
  - 根因:`consume` 先调 `applyDataOnly` 把 `state` 改成 `.completed`,再调 `applySideEffects` 判「是否通知」时读 `state == .running` 已不成立,`shouldNotify` 永远为假,清角标分支从不执行。
  - 修复:`consume` 在改状态前快照 `previousState` 透传给 `applySideEffects`,完成 / 失败 / 取消三个终态分支据 `previousState` 判通知,角标在终态瞬间被正确清除。`applyDataOnly` 保持纯净,回放路径不受影响。
- **有聊天记录时断开连接缺二次确认**:对话页 toolbar 的「断开」按钮一点即断,会终止后端进程并清除当前会话历史,没有破坏性操作的确认提示(不符合 HIG)。
  - 现在断开复用 State 加载/卸下同一套确认拦截:有聊天记录时弹出确认对话框(「断开会清除当前会话?」),说明后果并需用户确认;空会话直接断开,无破坏性后果,不打扰用户。
