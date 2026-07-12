# Preen macOS 开发指南

> Swift 壳 + Python sidecar 的本地开发流程。Phase 3 §5 spike 阶段(SidecarClient 链路验证)。

---

## 一、工程结构

```
macos/
├── Preen.xcodeproj                 # Xcode 工程(PBXFileSystemSynchronizedRootGroup)
└── Preen/
    ├── PreenApp.swift              # @main 入口
    ├── ContentView.swift           # NavigationSplitView 两栏
    ├── Sidecar/                    # 地基层(生产代码,spike 验证载体)
    │   ├── PythonResolver.swift    # 解释器路径 + 子进程环境
    │   ├── TrainEvent.swift        # Codable enum(穷举 events.py 12 类型 + unknown)
    │   ├── ServeEvent.swift        # Codable(ready/text_chunk/turn_end/ok/error)
    │   ├── ServeRequest.swift      # 请求 DTO(snake_case)
    │   ├── TrainJobRunner.swift    # Process + AsyncStream<TrainEvent>
    │   └── ServeClient.swift       # Process + id→Continuation + abort 双通道
    ├── Models/
    │   ├── TrainingConfig.swift    # train CLI 17 flag 默认值 + argv 生成
    │   └── GenConfig.swift         # 采样 7 字段(serve new_session 高创造力档)
    ├── Stores/
    │   ├── AppState.swift          # app 全局(当前面板/模型/跨面板 state 传递)
    │   ├── TrainStore.swift        # 训练状态机(idle/running/finishing/completed/...)
    │   └── ChatStore.swift         # 对话历史 + abort
    └── Views/
        ├── Sidebar.swift           # 侧边栏(模型选择器钉底部)
        ├── Training/               # 四状态:空/配置/运行/完成
        ├── Chat/                   # 单栏最小版(面板 + 消息 + 输入)
        └── Common/
```

**`PBXFileSystemSynchronizedRootGroup`**:Xcode 16+ 自动同步。往 `Preen/` 加 .swift,Xcode 自动纳入编译,**不用手改 pbxproj**。

---

## 二、本地开发:配 sidecar 环境变量

App 启动 sidecar 时通过 `PythonResolver` 解析解释器路径,顺序:

1. `PREEN_SIDECAR_PYTHON` 环境变量(开发态 → 指向本地 uv venv 的 `.venv/bin/python3`)
2. `Bundle.main/python/bin/python3`(发布形态)

仓库根(给 `PYTHONPATH=src`)同理:

1. `PREEN_REPO_ROOT` 环境变量
2. 从 `PREEN_SIDECAR_PYTHON` 反推(`.venv/bin/python3` → 父父父父 = repo root)

### 配置 Xcode Scheme 的环境变量

**Xcode → Product → Scheme → Edit Scheme → Run → Arguments → Environment Variables**,加:

| Key | Value |
|---|---|
| `PREEN_SIDECAR_PYTHON` | `/Users/no22/Projects/Preen/.venv/bin/python3` |
| `PREEN_REPO_ROOT` | `/Users/no22/Projects/Preen` |

这两条让 spike 阶段直接用本地 uv venv,不用等 bundle runtime 打包(#10)。

---

## 三、验证训练链路(Spec §5 验收 a 的一半)

1. 在 Xcode 选好模型目录(侧边栏底部「模型 → 选…」),例如 `models/converted/rwkv7-g1d-0.4b`(已转换的 HF 格式)。
2. 进训练面板 → 选训练数据(`train_data/NekoQA_10k/nekoqa_smoke_200.json`)。
3. 配超参(默认就行,或调小 epochs=3)。
4. 点「开始训练」。
5. **期望**:
   - loss 折线实时刷新(Swift Charts)。
   - 顶部摘要显示「第 N 轮 · 步 M / K · loss X · lr Y · 剩余 ...」。
   - 点「取消训练」→ 收到 `cancelled` 事件 → 切「已停止」态,曲线保留。

**finishing 中间态验证**(design.md §4.2 验收 c 铁律):
- 训练正常跑完会先看到「收尾中…」(final 事件,产物未落盘),然后才切「训练完成」(completed 事件)。
- 这两步之间有几百毫秒到几秒(取决于 state 大小),UI 必须停在「收尾中」不能跳完成。

---

## 四、验证 serve 链路(Spec §5 验收 a 的另一半)

1. 侧边栏底部选模型目录(同上)。
2. 切对话面板 → 点「连接」。
3. **期望**:连接状态变绿,serve 进程启动并发出 `ready` 事件(后台 console 会打 `[ServeClient] ready: protocol=1 ...`)。
4. 在输入框打一条消息 → 回车发送。
5. **期望**:assistant 消息流式追加(text_chunk 事件),完成后底部显示技术摘要(`stop=... · tokens=... · X t/s`)。
6. 发第二条 → 中途点「停止」按钮(或按 Esc)。
7. **期望**:abort 立即返回(按钮变回发送),已生成的文本保留并标「(已中断)」;被中断的请求的 error{aborted} 异步到达(ChatStore 标 isAborted)。
8. 发第三条 → 应能正常进行(abort 不影响后续)。

**busy 兜底验证**:理论上 UI 已禁用发送按钮(isGenerating 时),但如果绕过 UI(如并发 Task),serve 会返回 `error{busy}`,ChatStore 在 `lastError` 显示。

---

## 五、abort 双通道正确性(关键)

design.md / Spec §5 的 abort 是**双向独立通道**:

- 发 `abort` 指令 → 立即收 `ok`(读线程内联处理,不进队列)。
- 被中断的原 `send` 请求(不同 id)→ 稍后收 `error{aborted}`。

**`ServeClient` 里两者的 continuation 完全独立,绝不耦合**。验证:
- abort 的 await 立即返回(ServeClient.abort() 抛 ServeError 不算成功)。
- 原 send 的 await 收到 ServeError.aborted(ChatStore.handleSendError 标 isAborted)。
- 两者 id 不同 —— abort 用 `newId()` 生成自己的,不碰原 send 的 id。

---

## 六、Swift 6 迁移(留作 Sidecar 层稳定后的独立工作)

**当前 SWIFT_VERSION = 5.0(刻意)**:今晚写的是 Process + FileHandle + AsyncStream + continuation 表 + @Observable 的组合,严格并发下跟 Sendable 边界搏斗会挤掉功能,或逼出 `@unchecked Sendable` / `nonisolated(unsafe)` 这种更糟的债。

`SWIFT_APPROACHABLE_CONCURRENCY = NO`,`SWIFT_DEFAULT_ACTOR_ISOLATION` / `MEMBER_IMPORT_VISIBILITY` 也一并关掉(与语言模式 5 语义统一)。

**迁移时机**:SidecarClient 链路跑通、稳定之后,把 SWIFT_VERSION 改成 6.0,让编译器逐个指着并发问题修。届时:
- `Process` / `FileHandle` 后台读循环可能要标 `nonisolated` 或包 actor。
- continuation 表的 `NSLock` 可能换 `actor` + `dictionary`。
- `@MainActor` Store 跨 Task 调用要确认 Sendable 边界。

---

## 七、已知边界(本期不做,留后续)

| 项 | 去向 |
|---|---|
| 双轨共轴图(RSS 采样 + 削顶线 + 换页条) | #7 |
| 导入流程 token 着色预览 | #7 |
| A/B 双栏(对话面板本期单栏) | #8 |
| Inspector 完整(⌥⌘I 会话区) | #8 |
| 崩溃恢复重放(Spec §5 验收 c) | #8 |
| State 库面板(侧边栏 placeholder) | #9 |
| build_app.sh 打包 | #10 |

---

## 八、编译命令(给 agent 用)

```bash
xcodebuild -project macos/Preen.xcodeproj -scheme Preen \
  -destination 'platform=macOS' build 2>&1 | tail -30
```

成功标志:最后一行 `** BUILD SUCCEEDED **`。

失败排查:看 `error:` 行,通常类型不符 / API 拼写 / 平台可用性。
