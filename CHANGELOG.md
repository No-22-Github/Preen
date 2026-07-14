# 更新日志

> 记录首个预览版 [v0.1.0-beta.1](https://github.com/No-22-Github/Preen/releases/latest) 之后的变更。
> 格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/),版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
>
> **写入规矩**(固化在 AGENTS.md):有 commit 就记;一连串同主题改动(如纯 UI 体验)合并写一行即可;涉及 API / 后端 / 协议 / 数据格式的改动必须展开说明。

---

## [未发布]

### 修复

- **训练记录面板撑宽窗口**:切到训练记录界面时,若窗口整体宽度不足会撑出窗口、显示不全。
  - inspector(参数栏)默认改为收起(换 SceneStorage key 让旧用户也生效),toolbar 按钮从纯图标改为「参数」图文。
  - 左侧记录列表从硬固定宽度 260 改为可压缩范围(200~260),窗口偏窄时自动收窄,不再撑窗。
- **对话 State 无法卸下 / 切换模型时 State 被继承**:
  - `ChatStore` 新增 `clearState()`:已连接时走后端 `set_state(nil)` 重置会话,未连接时只清本地字段。
  - 切换模型、校验移除失效模型时,自动清除当前 State 及跨面板意图(`injectedStatePath`),避免旧模型的 State 继承给新模型。
  - 对话 toolbar「加载State…」改为双态胶囊:未选 State 时显示 `folder` 加载入口;已选时显示 `doc.fill` + 文件名 + `×`(淡色圆形背景)可点卸下。
  - 加载 / 卸下 State 时若当前已有聊天记录,会弹出确认对话框(与清空会话的垃圾桶按钮同一逻辑,因切换 State 会重置会话历史)。
