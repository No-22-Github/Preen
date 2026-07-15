# 更新日志

> 记录首个预览版 [v0.1.0-beta.1](https://github.com/No-22-Github/Preen/releases/latest) 之后的变更。
> 格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/),版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
>
> **写入规矩**(固化在 AGENTS.md):有 commit 就记;一连串同主题改动(如纯 UI 体验)合并写一行即可;涉及 API / 后端 / 协议 / 数据格式的改动必须展开说明。

---

## [未发布]

### 新增

- 新增 `tools/bench_bandwidth.py` MLX decode roofline 探针，以约 2.1GB 随机 bf16 权重分别测量大 GEMV 实效带宽与 sum reduction 上限，并据此估算 1.5B bf16 的理论 ms/token 地板。

### 变更

- **RWKV7 推理 prefill 词表投影优化**:保留完整 bf16 RWKV 主体、state/cache 更新与采样逻辑，但只对 prompt 最后一个 hidden 做 `lm_head`，不再构造 `[prompt_len, vocab_size]` 整段 logits，以降低长 prompt 的临时内存并提高 prefill 速度；未知模型保留原前向兼容路径。
- **RWKV7 bf16 decode 整步编译与异步流水优化**:将单 token RWKV 前向建模为 `(input_token, cache_state) -> (logits, next_cache_state)` 纯函数并通过 `mx.compile` 跨请求复用；采样 token 保持在 GPU 上，先用 `mx.async_eval` 提交下一步前向，再由 CPU 读取当前 token 并处理文本/停止条件。cache 不绑定具体会话，EOS/stop 时丢弃投机 state，生成结束后写回原对象，多轮续传、State 注入、重复惩罚及采样语义不变。int8 实测无稳定编译收益，量化模型继续走 eager 路径。
  - `bench_inference.py` 新增 `--decode-backend eager|compile|pipeline`，可在完全相同的 token 间隔计时口径下分别测原始 eager、同步整步编译和异步流水，默认 `pipeline`；量化模型即使请求编译也会报告实际回退的 `eager`。
- **欢迎窗口改为模态 sheet**:此前「欢迎使用 Preen」是一个独立普通窗口,会被切到主窗口后面、且定位随意。
  现改为挂在主窗口上的模态 sheet:从主窗口顶部滑出、盖在主窗口上方居中、带背景遮罩,点击主窗口区域不响应(真正的模态锁定)。
  - 去掉了独立的 welcome `WindowGroup` scene;首启与「窗口 → 欢迎使用 Preen」菜单改为翻转 `appState.isWelcomePresented` 标志,由 `ContentView` 的 `.sheet(isPresented:)` 驱动。
  - 同一标志仍驱动侧栏收起(背景呈空状态),语义不变。点入口项 / 「开始使用」/ 点背景遮罩均可关闭。

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
