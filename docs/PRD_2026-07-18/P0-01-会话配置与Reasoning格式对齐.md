# P0-01：会话配置与 Reasoning 格式对齐

> 优先级：P0  
> 状态：已评审，待实施  
> 所属版本：v1.1.0

## 一、问题

当前 Swift 对话端建立会话时默认使用 `template=qa`，没有可见的 `reasoning` 与 `think` 设置。协议层已经支持 `template / reasoning / think`，但产品界面没有把能力暴露出来。

对于带官方 reasoning chat template 的 RWKV-7 模型，缺少 BOS 或 think 标签可能导致输出偏离训练分布。即使简单问候能正常回答，也无法保证长问题、多轮对话和 State 验证始终可靠。

## 二、目标

- 将模板与 reasoning 方言提升为一等会话配置。
- 不对缺乏统一规范的模型做自动能力猜测；用户明确选择会话格式。
- 用户随时能看见当前会话的真实 prompt 口径。
- 改变会话格式时明确提示会重置历史。

## 三、用户故事

- 作为 G1 reasoning 模型用户，我希望 App 明确告诉我如何选择 reasoning / think，而不是在背景中猜测模型能力。
- 作为高级用户，我希望可以切换 `off / fast / on`，并理解各档行为。
- 作为排障用户，我希望能确认当前会话使用了什么模板，而不是猜测。

## 四、方案

### 4.1 新增统一会话配置

Swift 新增 UI 中立的会话配置结构，至少包含：

```text
template: qa | instruction | raw
reasoning: Bool
think: off | fast | on
gen_config: 现有七个采样字段
```

配置由 `ChatStore` 持有，创建或重建 session 时完整传给 `new_session`。

### 4.2 模型信息的使用边界

- App 不根据模型目录名、`tokenizer_config.json` 的字符串特征或未标准化的 Jinja 分支自动开启 reasoning。
- 模型名只用于展示与记录，不作为能力判定。
- 帮助文案给出 RWKV G1 类模型通常使用 `qa + reasoning=true + think=fast` 的明确示例，但最终选择权属于用户。

### 4.3 默认策略

- 新建普通会话默认 `qa + reasoning=false + think=off`。
- 从训练记录进入时继承训练模板，reasoning / think 仍使用当前用户设置，不因模型名自动改写。
- `instruction/raw` 与 reasoning 的非法组合在 UI 层禁用，并由后端继续兜底校验。

### 4.4 界面

对话 toolbar 显示紧凑会话口径，例如：

```text
qa · fast
```

生成参数 sheet 顶部增加“会话格式”区：

- 模板：QA / Instruction / Raw。
- Reasoning 格式：开关。
- 思考：Off / Fast / On。
- 一行解释：Fast 使用空 think 标签直接回答；On 展示完整思考阶段。

用户改变任一格式字段时，如果已有消息，走 P0-04 的统一确认流程。

## 五、技术约束

- `templates.py` 继续是格式文本单一事实源，Swift 不重新拼 prompt。
- Swift 只选择参数，不复刻 Python 渲染逻辑。
- think 拆分继续消费 serve 的 `phase/thinking/answer` 字段，不在 Swift 重写解析。
- reasoning 会话继续遵循现有“全量重放”语义，不能误标为 cache 续传。

## 六、验收标准

- [ ] 首次连接默认创建 `reasoning=false, think=off` 的会话。
- [ ] App 不根据模型名或未标准化的 chat template 自动改写 reasoning / think。
- [ ] 用户可在三次点击内找到 reasoning / think 设置与 G1 类模型的格式说明。
- [ ] 三种模板和三种 think 档位的合法组合均可正确下发。
- [ ] 非法组合在 UI 不可选择，构造协议请求时后端仍返回 `bad_request`。
- [ ] 使用固定 prompt 验证 App 下发配置后的后端实际 prompt 与 Python `render_prompt` 的 token IDs 逐 token 相等。
- [ ] think=on 时思考段与正式回答分开展示。
- [ ] 有历史时修改格式必须先确认，取消后会话与设置均不变化。

## 七、不做

- 不自动根据输出质量切换 think 档位。
- 不为每个模型名称维护人工规则表。
- 不在 Swift 展示或编辑完整 Jinja chat template。
