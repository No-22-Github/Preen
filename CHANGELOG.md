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

- **训练后台反馈**:训练期间 Dock 改用 macOS 原生红色百分比角标,并在训练开始时注册系统通知的“标记”能力,确保该开关和百分比角标可用;收到 State 已落盘的 `completed` 事件后清除角标并发送系统通知,App 位于前台时也会显示通知横幅。失败或取消仍会清除角标并发送对应通知。
- **训练默认学习率调整**:Swift 配置与 Python CLI/`TrainConfig` 的峰值学习率从 `0.01` 调整为 `0.0001`,最低学习率从 `0.0001` 调整为 `0.00001`,并同步参数重置按钮、CLI 帮助和 smoke 脚本。仅影响新建任务及恢复默认值;显式传参、已有训练记录和产物元数据保持原值。
- **训练进程内存压力图**:进程内存改为按训练步展示的 EMA 0.90 面积图,纵轴固定为本机物理内存容量,并以 70%/85% 容量阈值结合 macOS 内存压力信号显示绿、黄、红状态;采样从“最近 3,600 个秒级点”改为“全程每步峰值”,长训练不再丢失前段视图且不会按秒无限累积。
- **`doctor` 设备报告与内存展示口径**:`doctor_report()` 新增 `chip_name`、`hardware_model`、`os_version`、`os_build`、`memory_size_gib` 和 `working_set_gib`,仅从安全的 `sysctl` 白名单读取设备信息,不接触序列号或硬件 UUID。环境窗口、命令行摘要与 Issue 诊断中的设备总内存和 MLX 建议工作集统一以 GiB 展示(16G 机器分别显示 `16 GiB`、`11.84 GiB`);原 `memory_size_gb`、`working_set_gb` 继续保留以兼容既有调用方,所有训练、缓存和削顶判据仍使用十进制 GB。
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

### 修复
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
