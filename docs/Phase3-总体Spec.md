# Preen Phase 3 总体 Spec — SwiftUI 壳 + 模板重构 + 多轮会话 + 分发

> 状态：定稿待实施。本文整合了模板分类学、serve 会话协议、多轮 cache 生命周期、
> 数据导入、进程模型、UI 结构、打包分发、Demo 产出八个决策块。
> 各节末尾附验收标准，沿用仓库惯例：先注册验收，后写代码，不做事后叙事。
>
> 本文档是 **Phase3-总体Spec.md 与 Phase3-总体Spec-修订.md 的合并版**(2026-07-12 合并)。
> 裁决修订记录见文末「附录 C」(§1.X 落地裁决)与「附录 D」(§2 落地裁决)。

---

## 0. 范围与非目标

**范围（v1）**

- 模板系统重构：与 RWKV 官方格式规范对齐（qa / instruction 训练模板 + think 档位推理包装）
- InferenceEngine 多轮改造：state 续传为主路径（仅纯 qa），历史重放为修复路径（reasoning 全量走重放）
- `statetuner serve`：session-stateful 的常驻推理进程，stdin/stdout JSON lines
- 数据导入：schema 自动探测（Alpaca / ShareGPT / Messages / 裸 QA）→ 内部标准 jsonl
- SwiftUI 三面板：训练 / 预览 / State 库
- 打包分发：ZIP + ad-hoc 签名 + 自包含 Python runtime，"删 .app 即卸载"
- Demo 产出：A/B 对比 GIF、预训练 NekoQA state、流程录屏

**非目标（明确不做，防 scope creep）**

- HF Hub 浏览器 / 在线下载数据集（用户自己下文件丢进来）
- 多轮 A/B 对比（双 cache 并行）→ v1.5
- parquet 数据导入（pyarrow 依赖 ~100MB，伤 bundle 体积）→ 后续按需
- 付费开发者证书 / 公证 / DMG
- 推理性能优化（接口已稳定，优化在 InferenceEngine 内部进行，不阻塞本期）
- 偏好对（DPO chosen/rejected）、纯文本预训练格式的导入支持
- gradient checkpointing 低内存模式（沿用 Roadmap 既有结论：非必须）

---

## 1. 模板系统重构

### 1.1 分类学

两个正交维度，彻底分开：

| 维度 | 取值 | 作用域 |
|---|---|---|
| **训练/推理模板** | `qa` / `instruction` / `raw` | 决定样本结构与 prompt 包装 |
| **think 档位** | `off` / `fast` / `on` | 仅推理侧，仅 reasoning 模型，只影响 prompt 尾部渲染 |

对齐官方文档的映射（写进 README，一一对应）：

| 官方概念 | 本产品 |
|---|---|
| QA 格式（默认训练格式）`User: {q}\n\nAssistant:` | `template=qa` |
| 指令问答格式 `Instruction:/Input:/Response:` | `template=instruction` |
| 不思考模式 `Assistant: <think>\n</think>`（空 think 标签） | `think=fast` |
| 快思考模式（同上，空 think 标签跳过思考） | `think=fast` |
| 思考模式 `Assistant: <think`（模型续写思考段） | `think=on` |
| 裸 `Assistant:`（无 think 标签，**本产品自定义档，偏离训练分布**） | `think=off` |
| G1c+ 的 `(think)` / `(think a bit)` / `(think a lot)` | v1 不做，记录在案 |
| 材料问答 / few-shot / function call | 提示词技巧，非数据格式，不建模 |

> **修订（2026-07-12，附录 D.1）**：原表把官方"不思考模式"映射到 `think=off`，
> 实测官方 `chat_template` 的"不思考"实为 **fast 档**（`enable_thinking=False` → 空 think 标签）。
> 官方 `chat_template` 只有 fast/on 两档，无 off 档——`think=off`（裸 `Assistant:`）是本产品
> 自定义档，模型训练时 `Assistant:` 后必有 `<think>` 标签，off 档偏离训练分布。

### 1.2 templates.py 变更

```python
QA = TaskTemplate(
    prefix_template="User: {q}\n\nAssistant:",
    target_template=" {a}",
    stop_token=0,
    inference_stop_sequences=("\nUser:",),
    # 新增：多轮续传时，上一轮回答文本之后、本轮 User: 之前喂入的胶水
    continuation_prefix_template="\n\nUser: {q}\n\nAssistant:",
)

INSTRUCTION = TaskTemplate(
    prefix_template="Instruction: {instruction}\n\nInput: {input}\n\nResponse:",
    target_template=" {a}",
    stop_token=0,
    inference_stop_sequences=("\nInstruction:",),
    # instruction 格式语义上是单任务，不定义多轮续传（continuation=None，
    # chat 面板选 instruction 模板时禁用多轮，只允许单轮）
)
```

规则：

- `NEKO_QA` 更名为 `QA`，**硬重命名，不留 alias，不做弃用路径**。
  项目 0.1.0 未发布、无外部用户，breaking rename 此刻免费，以后不会更便宜。
  CLI / serve / 文档 / 测试全量替换，`--template nekoqa` 直接报"不支持的模板"。
- `G1G` 模板同样移除，拆解为 `qa` 结构 + reasoning 方言（bos 前缀 + think 标签），无 alias。
  实现为渲染函数 `render_prompt(prompt, template, *, reasoning: bool, think: ThinkMode)`：
  - `reasoning=True` 时 prefix 前加 `<|rwkv_tokenizer_end_of_text|>`，
    `Assistant:` 后按 think 档位追加 `""` / `" <think>\n</think>"` / `" <think"`。
  - CLI 新增 `--reasoning / --think [off|fast|on]` 选项替代原 `--template g1g`
    （`--think` 仅在 `--reasoning` 时合法，否则报参数错误）。
- **命名边界**：死掉的是模板标识符 `nekoqa`，数据集名 NekoQA 照常存在——
  demo 数据集、docs 里的实验记录、train_data 文件名都不改。
  历史实验报告（experiments/、docs 下已归档的）是记录，不回溯修改。
- **训练侧 target 永远不含 think 内容**。训练命令只接受 `qa` / `instruction`，
  think 参数在 train 命令中不存在（不是忽略，是没有这个参数）。
- `instruction` 的 `input` 字段为空时，prefix 降级为
  `Instruction: {instruction}\n\nResponse:`（去掉 Input 段及其前后空行）。
  这条规则写在 TaskTemplate 层，导入器与推理共用。
- templates.py 仍是全仓唯一允许手写 `\n` 拼接格式字面量的位置（沿用验收 d）。

### 1.3 内部命名清债

data.py 的 `Sample.cn / Sample.en` 是翻译实验遗产，统一更名：

- `cn` → `prompt_text`，`en` → `target_text`
- 内部标准 jsonl 的字段名定为 `{"prompt": ..., "response": ...}`
  （instruction 数据额外允许 `{"instruction": ..., "input": ..., "response": ...}`）
- 迁移一次做完：grep 全仓 `\.cn\b|\.en\b|"cn"|"en"`，含 tests 与 experiments 下的活跃脚本

### 1.4 验收标准

> 落地状态（2026-07，§1.X 已实施）：全部通过。详见各条勾注与实测证据。
> 裁决修订记录见文末「附录 C：§1.X 落地裁决」。

- [x] a. `template=qa` 的渲染输出与旧 `NEKO_QA` 字面格式逐 token 相等（重命名不改行为）
      _实测：真实 World tokenizer 上 `render_prompt("你好","qa")` 编码 (9 tokens)
      与旧 `QA.format_prefix` 编码逐 token 相等；golden `nekoqa_generate.json`
      5/5 条逐字零回退。_
- [x] b. `qa + reasoning + think=fast` 的渲染输出与旧 `G1G` 模板逐 token 相等
      _实测：真实 tokenizer 上 17 tokens 逐 token 相等（bos 单 token 0 + qa prefix
      + 空 think 标签）。注：参数名从 `g1_dialect` 改为 `reasoning`（裁决 C.1）。_
- [x] c. `think=off/fast/on` 三档的渲染输出与官方文档三种模式的字面格式一致（含尾部截断的 `<think`）
      _实测：off→`Assistant:`、fast→`Assistant: <think>\n</think>`、
      on→`Assistant: <think`，fast/on 与官方 chat_template 两档对齐。
      off 档无官方对照（附录 D.1）。_
- [x] d. instruction 模板空 input 降级后无残留空行（`"\n\n\n"` 不出现）
      _实测：`render_prompt("X","instruction",instruction_input="")` ==
      `"Instruction: X\n\nResponse:"`，无三连空行。_
- [x] e. train 命令传入 think 参数时报"未知参数"而非静默忽略
      _实测：`statetuner train --think fast` → Typer 报 `No such option: --think`。_
- [x] f. src/ 与 tests/ 无 `nekoqa`、`g1g`、`NEKO_QA`、`G1G` 标识符残留（CI grep；
      experiments/ 与历史 docs 豁免）
      _实测：`grep -rnE 'NEKO_QA|G1G|nekoqa|g1g' src/` 仅剩 docs 文件路径引用。
      **口径修订（裁决 C.2）**：CI grep 只扫 src/，tests/ 标识符作为场景描述豁免。_
- [x] g. 全仓无 `cn/en` 字段残留（CI grep，同样豁免历史记录）
      _实测：Sample 字段 `cn`/`en` → `prompt_text`/`target_text`，
      `grep -rnE '\.cn\b|\.en\b|"cn"|"en"' src/` 为空。_

---

## 2. InferenceEngine 多轮改造

### 2.1 核心决策

**state 续传（B）为主路径，历史重放（A）为修复路径——但续传仅限纯 qa，reasoning 全量走重放。**

依据 g1g 多轮 token 对齐实测（§2.5，结论见 `docs/g1g-decode-alignment.md §8`）与上游生态调研
（RWKV Runner / Ai00 / llama.cpp / HF chat_template 均为"渲染文本是真相，cache 只是前缀优化"的
重放语义；reasoning 模型历史轮剥离 think 内容是全行业惯例，由此导致的 cache 失效是公认固有成本）：

| 组合 | 续传 | 说明 |
|---|---|---|
| **qa（无方言）** | ✅ | 已实测续传 == 重放（World tokenizer 上 token 级等价） |
| **reasoning + think=off** | ❌ 强制重放 | off 档无官方对照（chat_template 只有 fast/on），偏离训练分布（附录 D.1） |
| **reasoning + think=fast/on** | ❌ 强制重放 | jinja 剥离历史 think，续传结构性偏离训练分布；失配点在上一轮 Assistant: 处，每轮全量走重放 |

> **续传分级简化（附录 D.2 裁决）**：原修订版曾把 `g1+off` 列为续传✅（待实测），
> 实测发现 chat_template 无 off 档后，经裁决改为：**只有纯 qa 走续传，所有 reasoning
> 组合（off/fast/on）走重放**。`continuation_safe = (template == "qa" and not reasoning)`，单一判定。

实现：ChatSession 层判定 `continuation_safe`；不安全的组合下 ChatSession 永不复用 cache，
引擎 API 不变。fast/on 档下 state snapshot 式的"RNN prefix cache"留作 v1.5 优化项，v1 不做。

- 续传：轮间保留 running cache，下轮只 prefill `continuation_prefix`。RNN 原生语义，每轮成本 O(新输入)。
- 重放：从 S₀ + 完整历史文本重新 prefill。用于三种场景：
  1. cache 被 stop_sequence 污染（见 2.2）
  2. 用户编辑/删除历史消息、rewind
  3. serve 进程崩溃后 UI 侧凭 history 恢复会话
  4. reasoning 方言（结构性偏离训练分布，见 §2.5 实测结论）

### 2.2 cache 洁净性规则

现状事实（已核对生成循环）：

- **eos 路径干净**：eos 在 append 与喂入之前检测，永不进 cache。
- **stop_sequence 路径污染**：token 采样后先喂入前向再解码检测，触发 stop 时
  `\nUser:` 的若干 token 已进 cache。

规则：

```
stop_reason == "eos"           → cache_clean = True，可直接续传
stop_reason == "stop_sequence" → cache_clean = False，标脏
stop_reason == "max_tokens"    → cache_clean = True（未产生越界 token），可续传
```

cache 脏时**不修补、不回滚**，下一轮自动走重放重建。理由：tuned state 的正常路径
是 eos 停（训练显式教了 stop_token=0），重放只是低频兜底，为它维护 snapshot
机制不值得。v1 不做任何 state snapshot。

### 2.3 token 账本

stop_sequence 分支现有 `generated = tokenizer.encode(final_text)`（干净文本重编码），
返回值与实际喂入模型的 token 不一致。多轮下"cache 里实际有什么"必须可追溯：

- `GenerationResult` 拆分两个字段：
  - `display_token_ids`：干净展示文本对应的 token（现有语义，改名）
  - `fed_token_ids`：实际走过前向的完整 token 序列（含污染部分）
- 重放重建时以 history 的**文本**为准（display 文本 + continuation 模板重新渲染编码），
  不用 fed_token_ids 拼接——文本是唯一事实源，token 只是审计记录。

### 2.4 API 变更

```python
class InferenceEngine:
    def generate(
        self, prompt: str, *,
        state: StateInput = None,
        cache=None,                # 新增：传入则续传，None 则按 state 新建
        config=None, on_text=None,
    ) -> GenerationResult:
        ...
    # GenerationResult 新增: cache（传出）、cache_clean: bool
```

`ChatSession` 变更：

- 持有 `history: list[Turn]`（Turn = role + display 文本）、`cache`、`cache_clean`
- `handle()` 流程：首轮 cache=None 新建；续传安全且 cache 干净 → 编码 `continuation_prefix`
  喂入续传；reasoning 方言 / cache 脏 / 历史被改 → `_replay()` 从 S₀ 重建
- `/clear` 获得真实语义：清空 history、丢弃 cache、回到 S₀
- 新增 `/rewind [n]`：截断最后 n 轮（默认 1），触发重放
- `/state PATH` 在多轮中途切换 state：清空会话（换 S₀ = 换人设，续传旧对话无意义），
  回复中明确提示"已重置会话"

### 2.5 g1g 多轮格式实测（前置任务，已完成）

未知数：G1 方言多轮时**轮间是否重复 bos（token 0）**、think 标签是否每轮渲染。
官方文档未覆盖，禁止猜测。

方法：用官方 `tokenizer_config.json` 的 `chat_template`（g1g/g1d/g1h 三模型完全一致）
渲染多轮对话，与本产品 continuation 渲染结果做逐 token 对比。

**结论（已合入 `docs/g1g-decode-alignment.md §8`）**：
1. **bos 每对话一次**，不每轮重复（jinja 里 bos 在 `{% for %}` 之前）。
2. **think 标签只在当前生成轮**，历史 assistant 是裸内容。
3. **官方 chat_template 只有 fast/on 两档，无 off 档**——off 是本产品自定义档。
4. 纯 qa 多轮：续传 == 重放（token 级等价）成立。
5. reasoning 多轮：续传结构性偏离训练分布（历史 think 标签污染），走重放。

### 2.6 验收标准

> 落地状态（2026-07，§2 已实施）：

- [x] a. 续传安全的组合（纯 qa）：同一 3 轮对话，续传路径与重放路径的最终生成结果
      逐 token 相等（temperature=0）。_实测：World tokenizer 上
      `encode(prefix)+encode(' A1')+encode(continuation) == encode(整体)`，
      QA target 带前导空格保证边界编码稳定（docs §8.4）。reasoning 组合不适用此项
      （设计上即重放，§2.1）。_
- [x] b. 人为构造 stop_sequence 停止后，下一轮自动重放（cache=None）。
      _实测：`test_dirty_cache_triggers_replay_next_turn`_
- [x] c. `/rewind` 后再提问，结果与"从头只进行撤销后剩余轮次"一致。
      _实测：`test_rewind_truncates_history_and_replays`_
- [x] d. `fed_token_ids` 与 `display_token_ids` 在 eos 停止时相等、stop_sequence 停止时前者为后者超集。
      _实测：`test_fed_token_ids_superset_of_display_on_stop_sequence`_
- [ ] e. 10 轮对话过程中常驻内存增量 < 100MB（cache 单份 + history 文本，无泄漏）。
      _（需真实模型长对话压测，留待集成验证）_
- [x] f. g1g 多轮 token 对齐实测报告合入 docs。
      _已合入 `docs/g1g-decode-alignment.md §8`。_

> **删除的验收 a2**（附录 D.2 裁决）：原修订版 §2.6.a2「g1+off 的 fed 序列与官方
> chat_template 渲染逐 token 相等」——g1+off 无官方对照（chat_template 无 off 档），
> 该验收无法成立，删除。

---

## 3. serve 会话协议

### 3.1 定位与进程模型

| 任务 | 进程形态 | 通信 |
|---|---|---|
| 训练 | 一次性子进程 `statetuner train` | stdout JSON lines（现状零改动），取消 = SIGINT |
| 推理（preview / chat / A/B） | 常驻进程 `statetuner serve` | stdin 收指令行，stdout 发事件行 |

serve 单进程单模型：启动时加载一个模型，常驻。换模型 = UI 重启 serve 进程
（简单可靠，v1 不做进程内换模型）。同一时刻至多一个 in-flight 生成，
取消用协议指令 `abort` 而非信号（进程是常驻的，SIGINT 语义留给"杀掉整个 serve"）。

### 3.2 帧格式

- 请求：一行一个 JSON 对象，`{"id": "<客户端生成的请求id>", "cmd": "<指令>", ...params}`
- 响应：一行一个 JSON 事件。凡由请求触发的事件带回 `"id"` 原样透传。
- 每个请求**必然**以一个终结事件收尾：`{"id", "type": "ok", ...}` 或
  `{"id", "type": "error", "code", "message"}`。中间可有任意条流式事件。
- 字段全部原生类型，复用 events.py 的序列化基建。
- stdout 上**只有** JSON 行。人类可读日志一律走 stderr（沿用 preview 的 `err=True` 惯例）。

### 3.3 指令集

```
hello                                   → ok {version, model, capabilities:
                                          {templates:[qa,instruction,raw],
                                           think:[off,fast,on], reasoning: bool}}
new_session {template, reasoning?, think?, state_path?, gen_config?}
                                        → ok {session_id}
send {session_id, text}                 → 流式 text_chunk* → turn_end → ok
abort {}                                → 中断当前生成 → 被中断请求收 error{code:aborted} → ok
set_state {session_id, state_path|null} → ok（附带重置会话，见 2.4）
set_config {session_id, ...gen_config}  → ok（下一轮生效）
rewind {session_id, n=1}                → ok {history_len}
reset {session_id}                      → ok（= /clear）
close_session {session_id}              → ok
preview {prompt, template, reasoning?, think?, state_path?, gen_config?, ab: bool}
                                        → ab=false: text_chunk* → turn_end → ok
                                        → ab=true:  turn_end{side:with_state} →
                                                    turn_end{side:baseline} → ok
                                          （一次性，不建 session，内部即建即弃 cache）
shutdown                                → ok 后进程退出 0
```

### 3.4 事件扩展（events.py 新增类型）

```
ready       serve 启动完成、模型加载完毕（无 id，进程级）
text_chunk  {id, session_id?, delta}          流式文本增量
turn_end    {id, session_id?, side?, result}  result = GenerationResult.to_dict()
                                              含 cache_clean / display_token_ids /
                                              stop_reason / 计时分段
ok          {id, ...payload}                  请求成功终结
error       {id?, code, message}              请求失败终结；无 id 的 error 表示协议级
                                              错误（如无法解析的行）
```

现有训练事件类型不变，train 一次性进程继续用。

### 3.5 错误语义

| code | 含义 | UI 处理 |
|---|---|---|
| `bad_request` | 参数校验失败（复用 service 层校验风格） | 表单标红 |
| `not_found` | session_id / state 文件不存在 | 提示并刷新列表 |
| `busy` | 已有 in-flight 生成 | 禁用发送按钮兜底 |
| `aborted` | 被 abort 中断 | 静默，UI 主动触发的 |
| `internal` | 未预期异常（附 traceback 摘要进 message） | 弹错误框 + 建议重启 serve |

协议不变式：**任何输入行都不能让 serve 进程崩溃**。解析失败发无 id 的
`error{code:bad_request}` 并继续读下一行。

### 3.6 验收标准

- [ ] a. 每个请求恰好收到一个终结事件（ok 或 error），乱序/并发请求下 id 对应无误
- [ ] b. 生成中途 abort，300ms 内收到 aborted 终结，进程可继续服务下一请求
- [ ] c. 向 stdin 灌 fuzz 垃圾行（非 JSON / 超长行 / 未知 cmd），进程存活且逐行回 error
- [ ] d. preview ab=true 的两路 result 与现有 CLI `preview --ab --json` 数值一致（同 seed）
- [ ] e. stdout 全程无非 JSON 行（管道 `| jq .` 通读不报错）

---

## 4. 数据导入与格式转换

### 4.1 形态

不做独立"转换器"页面。导入是训练面板的第一步：
**拖入文件 → schema 探测 → 字段映射确认 → 模板渲染预览 → 落成内部 jsonl**。
CLI 对应命令 `statetuner import`（UI 与 CLI 共用同一 service 层实现）。

支持输入：`.jsonl` / `.json`（数组或对象包 list）/ `.csv`（stdlib，utf-8）。
parquet 明确不支持，报错信息里给一句转换提示。

### 4.2 探测规则（按优先级）

读前 N=50 行采样，按下列顺序命中即停：

1. **Messages/ChatML**：存在 `messages: [{role, content}]` 结构
2. **ShareGPT**：存在 `conversations: [{from, value}]` 结构
3. **Alpaca**：同时存在 `instruction` 与 `output` 键（`input` 可选）
4. **裸 QA**：命中别名表中的一对键
   - prompt 侧别名：`prompt / question / q / query / user / 问 / instruction`（仅当无 output 键）
   - response 侧别名：`response / answer / a / completion / output / assistant / 答`
5. 全部不中 → 探测失败，UI 展示前 3 行原始数据 + 两个下拉框手动指定字段

采样中若 >10% 行不符合命中的 schema → 降级为"探测不确定"，同样走手动确认。

### 4.3 转换语义

| 源格式 | 目标 | 规则 |
|---|---|---|
| Alpaca | instruction 模板 jsonl | `{instruction, input, response}` 直通；input 全空的数据集提示"可选：降级为 qa（instruction→prompt）" |
| ShareGPT / Messages | qa 模板 jsonl | 多轮拆分策略 `--turn-policy first`（默认，只取首对）或 `all`（每个相邻 user/assistant 对独立成样本）。UI 里是一个单选。system 消息：first 策略下丢弃并计数提示；all 策略下同样丢弃 |
| 裸 QA | qa 模板 jsonl | 键名映射 |

产物：`{"prompt": ..., "response": ...}` 或 instruction 三字段版，
附 sidecar 文件 `<name>.import.json` 记录来源文件 hash、探测结果、策略、丢弃计数
（沿用 metadata 惯例，可追溯）。

### 4.4 渲染预览（必做，不是锦上添花）

导入确认前，展示前 3 条样本套用所选模板后的**最终喂给模型的文本**
（含 `User:`/`Assistant:` 包装、loss mask 边界用颜色区分 prefix/target 段）。
这是对"RWKV 对格式敏感"的产品化回应：train/inference 同构在代码层已保证，
预览把保证可视化。复用 `inspect_data` 的既有能力。

### 4.5 验收标准

- [ ] a. 四类格式各取一个真实 HF 数据集样本文件，探测全部正确命中
- [ ] b. 探测失败路径：故意喂 DPO 格式（chosen/rejected），走到手动映射而非误判
- [ ] c. ShareGPT 多轮文件，`first` 与 `all` 两种策略的产物行数与人工计数一致
- [ ] d. 导入产物直接喂给 train 命令可跑通（端到端冒烟）
- [ ] e. import.json 的 hash 可复现（同文件两次导入 hash 相等）

---

## 5. Swift 侧架构

三层，自底向上：

**SidecarClient（先写先测，即 spike）**

- `TrainJobRunner`：`Process` spawn `statetuner train`，
  `FileHandle.bytes.lines` 异步逐行 → `Codable` 解码为 `TrainEvent` enum
  （case 与 events.py 类型一一对应），取消 = `process.interrupt()`（SIGINT）
- `ServeClient`：spawn `statetuner serve` 常驻，写 stdin 行 / 读 stdout 行，
  维护 `id → CheckedContinuation` 表实现请求-响应配对，流式事件走 `AsyncStream`
- 解释器路径解析：环境变量 `PREEN_SIDECAR_PYTHON` 优先（开发指向本地 uv venv），
  否则 `Bundle.main.resourcePath/python/bin/python3`（发布形态）。
  开发与发布共用同一套代码，无分支逻辑
- sidecar 环境注入：`HF_HOME`、`PREEN_DATA_DIR`（见 §6）

**Store 层（@Observable）**

- `TrainStore`：攒训练事件、loss 曲线点列、job 状态机（idle/running/completed/failed/cancelled）
- `ChatStore`：history、per-session 状态、cache_clean 展示（脏时下一轮打"重建中"标记）
- `LibraryStore`：扫描输出目录、解析 metadata.json

**View 层：三面板 + 侧边栏**

1. **训练**：模型/数据 picker → 导入流程（§4.4 的预览嵌在这里）→ 超参表单
   （默认值来自 decision-precision 结论，默认折叠）→ 运行视图
   （Swift Charts loss 曲线吃 step/epoch_end 事件、进度条、取消）→
   completed 后产物卡片 + "去预览"按钮
2. **预览**：state 选择器、prompt 输入、A/B 开关（左右分栏 with_state vs baseline）、
   think 档位下拉（G1 模型时显示）、流式输出。产品灵魂，demo 素材源
3. **State 库**：List + metadata 详情 + 导出 .pth 按钮（对接 RWKV Runner）+
   "在 Finder 中显示"

**验收标准**

- [ ] a. spike：丑陋单按钮壳完成两条链路端到端——tiny 数据训练出 loss 曲线并可取消；
      serve 会话完成 3 轮对话并 abort 一次。**此项通过前不写任何面板 UI**
- [ ] b. SIGINT 打断训练时机压测：在 mx.eval 执行中途发送，进程干净退出且发出 cancelled 事件
- [ ] c. serve 进程被 kill -9 后，ChatStore 凭 history 自动重启进程并重放恢复会话

---

## 6. 打包与分发

### 6.1 形态

ZIP 内含 `Preen.app` + `README`。用户侧两步：解压、`xattr -cr Preen.app`。
卸载 = 删 .app。目标用户会开终端，不做更多妥协。

### 6.2 Bundle 结构与数据边界

```
Preen.app/Contents/
  MacOS/Preen                      Swift 可执行
  Resources/python/                python-build-standalone
  Resources/python/lib/.../site-packages/   pip install --target 灌入
~/Library/Application Support/Preen/
  models/        模型权重（GB 级，绝不进 bundle）
  states/        训练产物 state + metadata
  datasets/      导入产物 jsonl + import.json
  hf-cache/      HF_HOME 指到这里，禁止污染 ~/.cache
```

"删 .app 留数据"是 macOS 标准语义。体面化两件套：设置页"在 Finder 中显示数据目录"
按钮；README 一行"完全卸载 = 删 app + 删该目录"。

### 6.3 build_app.sh 流水线（顺序固定）

1. 下载/缓存 python-build-standalone（版本钉死在脚本内）
2. `pip install --target` 灌依赖。**mlx-lm 的 git 依赖钉 commit hash**
   （pyproject 同步改为 `mlx-lm @ git+...@<hash>`，构建可复现是硬需求）
3. 裁剪：`__pycache__`、`tests/`、`*.dist-info` 杂物、transformers 中无关模型代码、
   `.so` strip。目标：压缩后 ZIP ≤ 300MB
4. 组装 .app、拷入 Swift 产物
5. **ad-hoc 签名**：`codesign --force --deep -s - Preen.app`
   （Apple Silicon 上无签名二进制直接被杀且报错误导人；此步必须是对 bundle 的
   最后一次写操作，签后任何文件改动都会失效）
6. zip、产出 sha256

### 6.4 验收标准

- [ ] a. 干净机器（或新建用户账户）：无 uv、无 conda、无 Xcode CLT，
      解压 + xattr 后双击可用，完整走通"导入 → 训练 → 预览"
- [ ] b. 全程无 `~/.cache/huggingface` 目录产生
- [ ] c. `codesign --verify --deep` 通过；ZIP ≤ 300MB
- [ ] d. 同一 commit 两次构建，site-packages 内容 diff 为空（依赖钉死验证）
- [ ] e. spike 阶段（§5 验收 a 之前）先用 bundle 内 runtime 跑一次冒烟，
      动态库加载/路径问题前置暴露

---

## 7. Demo 产出

Demo 不是独立开发项，预览面板本身就是 demo。三样产物：

1. **README 头图 GIF**：A/B 对比，同一 prompt，左基线正经回答、右 NekoQA 猫娘腔。
   ≤ 10 秒，录屏转 GIF。零解释成本呈现"state tuning 是什么"
2. **Release 附预训练 NekoQA state**：用户下载 app 后两分钟内体验到 A/B 预览，
   不必先训练。附 metadata（训练配置、数据版本），本身也是产品可追溯性的示范
3. **60–90 秒完整流程录屏**：拖入数据 → 探测/预览 → 训练 loss 下降 → A/B 预览，
   发 RWKV 社区（兼作实习作品集素材）

发布检查：GIF 中的输出必须真实（不摆拍），seed 固定可复现，
prompt 选一个基线漂移明显的（先跑 5–10 个候选挑效果）。

---

## 8. 实施顺序

依赖关系决定顺序，模板先行（serve 协议字段依赖分类学）：

| # | 工作块 | 前置 | 预估 |
|---|---|---|---|
| 1 | 模板重构（§1）：改名 + INSTRUCTION + think 渲染 + cn/en 清债 | — | 1–1.5d |
| 2 | g1g 多轮 token 对齐实测（§2.5） | 1 | 0.5d |
| 3 | 引擎多轮改造（§2）：cache 传入传出 + 洁净性 + ChatSession | 1,2 | 1.5–2d |
| 4 | serve 实现（§3） | 1,3 | 1–1.5d |
| 5 | bundle runtime 冒烟（§6 验收 e） | —（可并行） | 0.5d |
| 6 | Swift spike：SidecarClient 两条链路（§5 验收 a） | 4,5 | 2d |
| 7 | 训练面板 + 导入流程（§4, §5） | 6 | 3–4d |
| 8 | 预览面板 → **此时可录 GIF，先发不等全量** | 6 | 2d |
| 9 | State 库面板 | 6 | 1d |
| 10 | build_app.sh 全量 + 干净机验收（§6） | 7–9 | 2–3d |
| 11 | Demo 三件套（§7） | 8,10 | 1d |

关键纪律：**#6 spike 不通过，#7–9 不动工**；**#2 实测没结论，g1 方言多轮不合入**。

> **#2/#3 落地状态（2026-07-12）**：#2 g1g 多轮实测已完成（结论见 docs §8），
> #3 引擎多轮改造已实施（§2 验收 a/b/c/d/f 通过，e 待真实模型压测）。

---

## 附录 A：本期决策记录（含否决项）

| 决策 | 结论 | 理由摘要 |
|---|---|---|
| 进程模型 | 训练一次性 / 推理常驻，混合式 | 训练要崩溃隔离与零改动复用；推理要省 load model 时间 |
| 多轮路线 | state 续传主路径(仅纯 qa) + 重放修复(reasoning 全量) | RNN 原生语义、与产品概念自洽；eos 干净/stop 污染的事实支持混合；reasoning 续传偏离训练分布(§2.5 实测) |
| state snapshot | 不做 | 脏 cache 低频（训练教了 eos），重放兜底足够 |
| 多轮 A/B | 推迟 v1.5 | 双 cache 管理 + UI 复杂度，不阻塞 demo |
| nekoqa/g1g 命名 | 硬重命名，无 alias | 0.1.0 未发布，breaking 免费；通用格式不该用数据集命名；g1g 实为 qa+方言+快思考。数据集名 NekoQA 与历史记录不动 |
| HF Hub 浏览器 | 不做 | 独立产品级复杂度，v1 用户自带文件 |
| parquet | 不做 | pyarrow ~100MB 伤 bundle |
| 签名 | ad-hoc（免费） | Apple Silicon 硬要求有签名，不要求付费证书 |
| Python runtime | 打进 bundle | 对齐"删 .app 即卸载"审美；uv 引导方案否决 |
| 取消机制 | 训练 SIGINT / serve 协议 abort | 一次性进程信号语义清晰；常驻进程信号留给整体退出 |

---

## 附录 C：§1.X 落地裁决（2026-07-12，实施时）

§1.X（模板系统重构）落地时，对 §1.2/§1.4 的三处口径做了用户裁决修订。
正文保留原始契约，修订记录在此可追溯。

### C.1 参数名 `g1_dialect` → `reasoning`（§1.2 修订）

**正文（§1.2）原契约**：`render_prompt(prompt, template, *, g1_dialect: bool, think: ThinkMode)`，
CLI `--g1 / --think [off|fast|on]`。

**裁决修订**：参数名改为 `reasoning` / `--reasoning`。

**理由**：`g1` 是 RWKV 模型版本号（g1g/g1h/g1i 会迭代），写死版本号在参数名里
会在下次版本迭代时语义失效或需要 breaking rename。`reasoning` 描述的是模型属性
（"是 reasoning 类模型，需要 bos + think 外壳"），语义稳定。
config.json 里 `model_type` 全是 `rwkv7`，无法自动区分版本，只能靠用户知道
自己的模型是哪类——参数语义名比版本号更合适。

实施时 `render_prompt(prompt, template, *, reasoning: bool, think: ThinkMode)`，
CLI `--reasoning / --think [off|fast|on]`（`--think` 仅在 `--reasoning` 时合法）。

### C.2 验收 f 口径：CI grep 只扫 src/（§1.4 修订）

**正文（§1.4.f）原契约**：src/ 与 tests/ 均不得有 `nekoqa`/`g1g`/`NEKO_QA`/`G1G`
标识符残留。

**裁决修订**：CI grep 只扫 `src/`。`tests/` 的文件名（`test_nekoqa.py`）、
函数名（`test_nekoqa_*` / `test_eval_g1g_*`）、类名作为**场景描述**保留，
不强制重命名。数据集文件名字符串（`nekoqa_04b_s42.npz`、`nekoqa_smoke_200.json`、
`golden/nekoqa_*.json`）按 §1.2「数据集名 NekoQA 照常存在」豁免。

**理由**：tests 里的命名是测试场景的语义描述（"NekoQA 数据管线的测试"），
不是代码标识符债。强制重命名会增加改动面却不提升清晰度。
tests 内用 `from statetuner.templates import QA as NEKO_QA` 局部别名维持断言可读性，
不污染 src/。

### C.3 chat 命令默认值：裸 qa，不默认开 reasoning（§5/§1 隐含决策）

**裁决**：chat 命令 `--template qa`（默认）、`--reasoning=False`（默认）、
`--think=off`（默认）。启动 banner 提示「G1 系列加 `--reasoning --think fast`」。

**理由**：与 C.1 一致——不依赖模型版本检测（config.json 拿不到版本信号），
用户知道自己的模型是否 reasoning 类。默认裸 qa 可预测，不写死版本。
旧 chat 默认 `template=g1g` 的行为（产品主线 G1 原生对话）改为用户显式开启。

### 其他实施细节（非裁决，记录用）

- `TaskTemplate` 新增 `continuation_prefix_template`（多轮续传胶水，§2 消费）
  和 `drop_input_when_empty`（instruction 空 input 降级开关）字段。
- `G1G` 模板的 `inference_stop_sequences=("\nUser:", "\nSystem:")` 合并到
  `QA.inference_stop_sequences=("\nUser:",)`——reasoning 方言不改 stop 边界，
  `\nSystem:` 是历史冗余兜底。
- 内部标准 jsonl 字段名契约（§1.3）钉死在 `data.py` docstring：
  `{"prompt": ..., "response": ...}` 或 `{"instruction": ..., "input": ..., "response": ...}`。
- AGENTS.md「g1g 推理对齐」段保留为历史实测记录，加 API 迁移说明
  （旧 `G1G` == 新 `render_prompt(p, "qa", reasoning=True, think="fast")`），
  机制事实不变。

---

## 附录 D：§2 落地裁决（2026-07-12，InferenceEngine 多轮改造实施时）

§2（InferenceEngine 多轮改造）落地时，基于 g1g 多轮 token 对齐实测（§2.5，
结论见 `docs/g1g-decode-alignment.md §8`），对 §2.1 续传分级表、§1.1 官方映射表、
§2.6 验收做了裁决修订。

### D.1 官方"不思考模式"映射修正：`off` → `fast`（§1.1 修订）

**实测发现**：官方 `tokenizer_config.json` 的 `chat_template` 只有 fast
（`enable_thinking=False` → `Assistant: <think>\n</think>`）和 on（default →
`Assistant: <think`）两档，**没有 off 档**。本产品的 `think=off`（`Assistant:` 后
什么都不加）是自定义档，模型训练时 `Assistant:` 后必有 `<think>` 标签，off 档
偏离训练分布。

**裁决**：
- §1.1 官方映射表"不思考模式"映射到 `think=fast`（对齐 chat_template ground truth）。
- `think=off` 保留为本产品自定义档（裸 `Assistant:`），但标注其偏离训练分布。
- §1.4.c 验收说明更新：fast/on 与官方 chat_template 两档对齐，off 档无官方对照。

### D.2 续传分级简化：g1+off 归入重放（§2.1 修订）

**原修订版 §2.1 续传分级表**：`qa` 续传✅、`g1+off` 续传✅（待实测）、
`g1+fast/on` 强制重放。

**实测冲突**：g1+off 无官方对照（chat_template 无 off 档，D.1），单轮已偏离训练
分布，续传/重放等价性无意义。原 §2.6.a2「g1+off fed 序列对齐官方」无法成立。

**裁决**：续传分级简化为单一判定——
`continuation_safe = (template == "qa" and not reasoning)`。只有纯 qa 走续传，
所有 reasoning 组合（off/fast/on）走重放。§2.6.a2 删除。

**理由**：只有纯 qa 有 token 级等价保证（World tokenizer 实测续传 == 重放），
所有 reasoning 组合都走重放（反正都是 O(history) 重算，实现统一更简单）。
reasoning 续传偏离训练分布不是设计缺陷，是 reasoning 模型品类结构性属性
（DeepSeek-R1 / Qwen3 等惯例都是历史剥 think）。

### D.3 其他实施细节（非裁决，记录用）

- `GenerationResult` 字段：`token_ids` → `display_token_ids`（rename，现有语义），
  新增 `fed_token_ids`（实际喂入前向的完整 token，含污染）、`cache`（传出）、
  `cache_clean`（洁净性）。`token_count` property 基于 `display_token_ids`。
- `InferenceEngine.generate` 新增 `cache` 参数：传入则续传（复用 cache，只 prefill
  新 prompt），None 则按 state 新建。
- `ChatSession` 新增 `history: list[Turn]`、`cache`、`cache_clean` 字段。
  `_handle_single` 三路状态机：首轮（cache=None）/ 续传（`_can_continue()` True）/
  重放（其他）。
- `/rewind [n]`：每轮 = [user, assistant] 两个 turn，截断 n 轮 = 删最后 2n 个 turn，
  clamp 到 0，触发重放。
- `/state` 中途切换：清空 history + cache（换 S₀ = 换人设）。
- `to_dict()` 排除 `cache` 字段（不透明的模型对象，不可 JSON 序列化）。
