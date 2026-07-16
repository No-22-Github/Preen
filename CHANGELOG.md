# 更新日志

> 记录首个预览版 [v0.1.0-beta.1](https://github.com/No-22-Github/Preen/releases/latest) 之后的变更。
> 格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/),版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
>
> **写入规矩**(固化在 AGENTS.md):有 commit 就记;一连串同主题改动(如纯 UI 体验)合并写一行即可;涉及 API / 后端 / 协议 / 数据格式的改动必须展开说明。

---

## [未发布]

### 新增

- 训练统计图新增实时 Learning Rate 曲线与 warmup 完成标记,复用逐步训练事件中的实际 LR;实时训练时 LR 与进程内存并排展示,三张图随窗口高度自适应且 Loss 获得更大的展示区域,Loss 纵轴按 Raw、EMA 与 Held-out 的实时范围自动缩放并保留上下留白,矮窗口下图表可滚动且取消训练底栏保持可见;步数轴采用一基整齐刻度并强制显示任务总步数,三张图的鼠标悬停互不联动,仿 macOS"股市"以竖线和沿 EMA 主线移动的圆点呈现选中状态,顶部按 Raw、EMA 顺序同时显示读数,不再显示悬停横线;内存图移除压力区间图例和黄色阈值线,仅保留红色严重阈值虚线;训练记录详情可重放旧任务的历史 LR 曲线。
- **后端环境诊断与 Issue 信息复制**:环境窗口采用 macOS 原生 Form、Section 与 LabeledContent 呈现整体健康状态、芯片与系统版本、设备内存和 MLX 工作集上限,检查时间以自动更新的相对时间显示,说明文字按系统设置习惯置于左侧标签下方,组件版本标题整行均可点击展开且正常组件不重复显示状态图标,底部操作采用克制的纯文字按钮;诊断日志按来源提供居中的原生空状态与对应说明,有日志时支持等宽显示和文本选择;新增一键复制脱敏 Markdown 诊断摘要,包含 Preen/macOS/Python/MLX 环境及硬件标识与安全状态枚举,不自动包含序列号、硬件 UUID、日志、PID 或本地路径。
- 新增 `tools/bench_bandwidth.py` MLX decode roofline 探针，以约 2.1GB 随机 bf16 权重分别测量大 GEMV 实效带宽与 sum reduction 上限，并据此估算 1.5B bf16 的理论 ms/token 地板。

### 变更

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
