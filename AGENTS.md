# AGENTS.md — Preen 工作指南

> 本文件给未来 ZCode agent 读。写的是**代码里看不出来、踩过坑才知道**的事。
> 整体规划见 **[docs/RWKV-StateTuner-Roadmap.md](docs/RWKV-StateTuner-Roadmap.md)**(落地路线图、技术决策、风险、内存口径说明)。
> 原理见 [docs/P0-理论指南.md](docs/P0-理论指南.md)。

---

## 这是什么项目

Mac 原生的 RWKV-7 **state tuning** 工具。冻结模型全部权重,只训练每层初始状态矩阵 S₀(64×64)。
当前阶段:**P2 已完成**(CLI + 独立推理引擎 + `.pth` 导出就绪),NekoQA 风格迁移已跑通。

**git 分支注意**:正式代码全部在 `main` 分支(`src/statetuner/` + 根 `tests/` + 根 `pyproject.toml`)。
唯一存活的实验分支是 `exp/precision`(混合精度实验归档),用不到也不要动它。

---

## 目录与架构边界

```
src/statetuner/          正式包(改这里)
├── core.py              patch ops 路径 + 可训练 state + generate
├── data.py              数据管线 (encode_template_sample 通用 / load_qa_dataset 遗留 / load_standard_jsonl 导入产物)
├── templates.py         ★ 格式模板单一事实源(QA / INSTRUCTION;Phase 3 §1 重构后)
├── train.py             训练循环(产品化)
├── events.py            结构化事件(JSON lines,为 IPC 铺路)
├── export.py            .pth 导出(RWKV Runner 挂载)
├── pth_io.py            纯 Python torch .pth 读写(无 torch;convert 与 export 共用)
├── inference.py         独立推理引擎(generate 支持 should_abort 钩子,§3 serve 用)
├── chat.py              ChatSession(多轮续传/重放,持有 abort_checker)
├── importer.py          ★ 数据导入(§4):Alpaca/ShareGPT/Messages/裸 QA 探测 → 标准 jsonl
├── model_converter.py   ★ 原生 RWKV-7 .pth → HF safetensors 正式转换模块
├── tool_events.py       ★ 离线工具任务事件(started/progress/completed/failed/cancelled)
├── serve.py             ★ 常驻推理协议(§3):stdin/stdout JSON lines + abort
├── inspection.py        环境/数据/state 预检(含 inspect_standard_jsonl 给导入产物)
└── cli.py               train/eval/export/preview/chat/serve/import,带 --template 开关

tests/                   回归测试(改 src 必跑)
├── fixtures/              NekoQA 基准 state(nekoqa_04b_s42.npz,产品 CLI 训练)
│   └── import/            导入器测试 fixtures(alpaca/sharegpt/messages/bare_qa/dpo 各一份)
└── golden/                推理 golden 快照(nekoqa_*.json)
tools/                   模型转换兼容入口(convert_rwkv7_to_hf.py,实现已下沉正式包)+ 内存探针
├── fixtures/              转换校验模板(rwkv7_hf_template.json,从 fla-hub 0.1B 生成)
│                          gen_convert_fixture.py = 一次性生成脚本(上游 schema 漂移时重跑)
└── ...
assets/
└── rwkv_world_tokenizer/  vendor 的 World tokenizer 5 文件(转换器缺省 --tokenizer-src)
                          + SOURCE.md(来源仓库 + 同步说明)
experiments/p0_translate/  ★ P0 历史归档,不要动(含已废弃翻译路径,保留可复现性)
train_data/NekoQA_10k/   NekoQA 数据集(Apache-2.0,见 NOTICE.md)
```

**架构铁律:**
- `templates.py` 是**全仓唯一**允许持有 `\n`-拼接格式字面量的地方。其他文件一律 `TaskTemplate.format_prefix/format_target` 派生。改前 grep 确认。
- 训练循环主体(`train.py` 的 `Trainer`)是禁区——不调超参默认值、不动 loss 计算。
- `data.py` 的 `encode_template_sample`(通用)是基础,`load_qa_dataset`(遗留 instruction/output)/ `load_standard_jsonl`(导入产物 prompt/response)派生自它。新任务加 loader,别 fork。
- **数据分流(service.run_training)**:数据文件旁若有 `<name>.import.json` sidecar → 走 importer 标准路径(`inspect_standard_jsonl` + `load_standard_jsonl`);否则走遗留 `inspect_data` + `load_qa_dataset`。两条路径都有截断检查。
- **serve 多轮复用 ChatSession**(§2 已完成续传/重放),serve 只做协议桥接,不重复实现多轮逻辑。abort 通过 `ChatSession.abort_checker` → `generate(should_abort=)` 透传。
- state 导出方向:x070(RWKV-7)**原样不 transpose**;v5/v6 才 swapaxes。见 export.py docstring。

---

## 常用命令

```bash
# 跑测试(改完代码必跑)
.venv/bin/python -m pytest -q                    # 快测(~22s,不加载模型做训练)
.venv/bin/python -m pytest --slow -q             # 含训练行为断言(~5min,会加载 0.4B)
.venv/bin/python -m pytest tests/test_nekoqa.py -q  # 只跑 NekoQA 管线快测

# 训练(PYTHONPATH=src 是必须的,src layout)
PYTHONPATH=src .venv/bin/python -m statetuner.cli train \
  --model models/converted/rwkv7-g1d-0.4b \
  --data train_data/NekoQA_10k/nekoqa_smoke_200.json \
  --template qa \
  --out state.npz --events-file events.jsonl \
  --lr 0.01 --epochs 3 --ctx-len 512 --no-early-stop --seed 42

# 生成/预览(--template qa|instruction|raw;G1 系列 reasoning 模型加 --reasoning --think fast)
PYTHONPATH=src .venv/bin/python -m statetuner.cli preview \
  --model models/converted/rwkv7-g1d-0.4b --state state.npz \
  --prompt "你好" --template qa --ab

# 数据导入(§4):Alpaca/ShareGPT/Messages/裸 QA 自动探测 → 标准 jsonl + sidecar
PYTHONPATH=src .venv/bin/python -m statetuner.cli import \
  --data external_dataset.jsonl --out imported.jsonl --turn-policy first
# 导入产物可直接喂 train(数据旁的 .import.json sidecar 让 service 走标准 loader)

# serve(§3):常驻推理进程,SwiftUI/SidecarClient 通过 stdin/stdout JSON lines 会话
PYTHONPATH=src .venv/bin/python -m statetuner.cli serve \
  --model models/converted/rwkv7-g1d-0.4b
# 协议:每行一个 JSON 请求 {"id","cmd",...} → stdout JSON 事件流(ready/text_chunk/turn_end/ok/error)
# 指令集:hello/new_session/send/abort/set_state/set_config/rewind/reset/close_session/preview/shutdown
```

`--template`(qa / instruction / raw)决定数据 loader 和 prompt 渲染(训练/推理必须同模板,保证编码同构)。
reasoning 方言(bos 前缀 + think 标签)由正交开关 `--reasoning` + `--think off|fast|on` 控制(仅 qa 模板合法),不写死模型版本号(G1g/G1h/... 会迭代)。NekoQA smoke 全流程脚本:`scripts/nekoqa_smoke.sh`。

**serve 协议要点(§3,Swift spike 前必读):**
- 单进程单模型常驻;换模型 = 重启 serve。同时至多一个 in-flight send/preview(busy 锁)。
- 每个请求恰好一个终结事件(`ok` / `error`),id 原样透传;`ready` 是进程级无 id 事件。
- abort:读线程内联处理 `abort` 指令 → `threading.Event.set()` → generate 下一步抛 `GenerationAborted`。延迟到下一个 MLX step 边界(~50-200ms)。
- stdout **只有** JSON 行;人类日志走 stderr。任何输入行不能让进程崩(全包 try/except → `error{bad_request/not_found/busy/aborted/internal}`)。

---

## 📝 CHANGELOG 纪律(写死,每次有更改必做)

**首个预览版 [v0.1.0-beta.1] 已发布。从此以后,凡是面向用户可见的更改,必须同步更新根目录 [CHANGELOG.md](CHANGELOG.md) 的 `[未发布]` 段。** 不写完不算任务完成。

格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/),分 `新增` / `变更` / `修复` / `移除` 小节,中文。

**详略规矩:**
- **一连串同主题改动**(如纯 UI 体验优化、文案统一、样式微调)→ **合并写一行即可**,不要逐条罗列。例:`- 优化对话面板 toolbar 视觉与交互。`
- **涉及 API / 后端 / 协议 / 数据格式 / 训练口径的改动** → **必须展开**:写了什么、为什么、影响范围。例:`- \`ChatStore\` 新增 \`clearState()\`:已连接走 \`set_state(nil)\` 重置会话,未连接只清本地。切模型时自动调用,避免旧 State 继承给新模型。`

**判断"要不要记"的标准:用户能感知到 = 要记。** 内部重构、测试、文档、依赖升级(无行为变化)不用记,除非修了用户会撞到的 bug。

发版时把 `[未发布]` 改成版本号 + 日期,新开空的 `[未发布]`。

---

## 已验证为真的技术知识

### 编码管线
- **拆分编码同构性**:`encode(prefix) + encode(target) == encode(prefix+target)`,在 0.4B 和 1.5B 的 World tokenizer 上都实测成立。所以拆分编码不改 token 序列,golden 可零回退。
- **终止符**:token 0 = World tokenizer eos(`<|rwkv_tokenizer_end_of_text|>`)。`encode_template_sample` 把它追加到 full_ids 末尾,mask 覆盖 target+stop 段。`core.generate` 遇 token 0 停下且不 decode 进输出(已修,eos 显形 bug 历史)。
- **mask 边界**:从 `prefix_len-1` 起 mask=1(预测第一个 target token 的位置),不是 `prefix_len`。`mask[i]=1 当 (i+1)>=prefix_len`。

### 训练
- **lr 默认 0.01**(不是 RWKV-PEFT 老教程的 1.0;1.0 会爆炸)。cosine 衰减到 lr_floor/100。
- **state std 健康区间未标定**(旧 >1.0 预警线已作废,官方 roleplay state std 1.385 工作正常)。目前只记录不报警。
- 0.4B NekoQA 200条×3epoch:loss 3.97→3.12→2.36,std 0.11→0.12。
- 1.5B NekoQA 200条×2epoch:loss 2.48→1.71,std 0.12→0.12。
- **风格注入实测成立**:无 state 普通助手 → 有 state 猫娘(括号动作 + 喵/主人)。state tuning **学风格不学事实**,内容映射(翻译)实测不可用——任务模板按此边界设计。
- **0.4B 基座固有重复缺陷**:有/无 state 都重复,这不是 state 引入的。

### 导出
- 键名 `blocks.{i}.att.time_state`,形状 (H,D,D) fp32,x070 原样不 swapaxes。
- Windows RWKV Runner rwkv.py:843 version>=7 不 transpose。
- **无 torch 依赖**(2026-07 移除)。读写 `.pth` 全在 `pth_io.py`:torch 的 `.pth` 是 zip+pickle+裸 storage,`read_pth` 拦 `persistent_load`/`_rebuild_tensor_v2` 复刻,`write_pth` 用纯 Python `pickle._Pickler`(非 C 版,才能覆写 `save` 手写 GLOBAL opcode)发 torch 符号引用。bf16 靠 `ml_dtypes`。改这块必跑 `test_pth_io.py`;有 torch 时 `test_torch_can_load` 会做外部 oracle 交叉验证(缺省 skip)。

---

## ⚠️ 内存事实(已归因,精度方案已锁定)

精度实验(`exp/precision` 分支)已完成内存归因 + 红线标定 + 多 seed 矩阵验证。**结论已固化,方案已锁定。** 关键事实:

### 精度方案(已锁定)
- **权重 bf16 + state fp32 训练**——即 main 现有实现(`make_state_params(dtype=mx.float32)`),不改。
- D 方案(state cast bf16、循环全程 bf16)**未采纳**:配对判据 4 红 1 绿,15b_s42 有确凿退化单例(Q8 连续"啊"121 字)。详见 `docs/decision-precision.md` 和 `experiments/mixed_precision/report_matrix.md`。
- 官方惯例是 bf16 权重 + kernel 内 fp32 state 累加;MLX 无可定制 kernel,state 保持 fp32 是我们能做到的最接近官方精神的方案。

### 内存口径(三口径)
- `mx.get_peak_memory()` **严重漏报**——只算 MLX allocator 分配,不含 Metal wired memory。**永远以 RSS(activity monitor / `ps`)为准。**
- **全仓内存单位统一 GB(÷10⁹)**,禁止 GiB(/1024³)混用(见下方「内存单位」节)。

### 红线标定(16GB 机器,bf16+c4G)
- **安全档 L600**(均 591 token / max 644):step_peak ~11.7G,削顶线 12.07G(working_set 95%)。
- **断点 L650**(均 636 token):step_peak 12.22G 顶到削顶线。
- fp32 比 bf16 step_peak 高 ~1.7G,fp32 红线更紧。
- 完整数据见 `experiments/mixed_precision/report_matrix.md`(本地留档,不进 git)。
- **产品 CLI 已接入 `--cache-limit-gb`**(train/eval/chat),**缺省 = `auto`(物理内存 × 25%,16G 机 ≈ 4.3G,c4G 同档)**。可显式传 GB 数(如 `--cache-limit-gb 4`)或 `auto` 覆盖。该参数必须在 load_model 前生效,`cli.py` 的 `_apply_cache_limit` helper 已保证时序;时序铁律来源见 `tools/mem_probe.py:106-117`(set_cache_limit 在 load_model 之前)。
- **⚠️ auto 默认改变了训练口径**:历史训练数据(0.4B/1.5B 的 loss/std 曲线)是在不设 cache_limit 下跑的;auto 默认开启后 cache 命中率下降,训练数据与历史不再严格可比。红线档内存数据仍适用(4.3G ≈ 4G)。

### 历史观察(已解释)
- 0.4B 训练实测 RSS ≈ 11~12GB:大头是 fp32 state 在 bf16 权重的 wkv 循环里把整个循环提升成 fp32(MLX 类型提升规则 bf16+fp32=fp32)。这是机制事实,非 bug。
- ctx_len 不是大头:ctx 512→192 RSS 几乎不变,因大头在不随 ctx 变化的 state 张量提升。
- 0.4B 和 1.5B 层数都是 24,state 参数量接近;内存问题两者同源。

排查技巧:`PYTHONPATH=src .venv/bin/python -c "..."` 里逐组件 `load_model` / `make_state_params` / `value_and_grad`,每个之后用 `resource.getrusage(RUSAGE_SELF).ru_maxrss` 读真实 RSS。

---

## g1g 推理对齐(已验证)

### reasoning prompt 格式 = 降智/错乱的分水岭(原 g1g 实测,Phase 3 §1 重构后 API 变更)

> **API 迁移说明(2026-07,Spec §1.2)**:本节实测时的旧 API 用 `templates.G1G` 整包模板 + `--template g1g`。
> Phase 3 §1 拆解后:旧 G1G == 新 `render_prompt(p, "qa", reasoning=True, think="fast")`,
> 即 qa 模板 + reasoning 方言(bos 前缀)+ think fast 档位。**下述机制事实不变,只是参数面换了。**

- **RWKV7-G1 是 reasoning 模型**,prompt 格式必须带 `<think>` 标签。新版等价渲染 =
  `REASONING_BOS + QA.format_prefix(q=p) + ThinkSuffix["fast"]`
  = `<|bos|>User: {q}\n\nAssistant: <think>\n</think>`,对齐官方 chat_template 的 `enable_thinking=False` 渲染。
- **实测铁证**:raw 格式(`User: 你好\n\nAssistant:`)→ 120 token 跑满、幻觉自报"ChatGPT";reasoning 方言 → 39 token 正常 eos、自然回答。**降智根因是缺 bos + 空 think 标签,不是模型或框架问题。**
- 两处关键段缺一不可:
  - 开头 `<|rwkv_tokenizer_end_of_text|>`(token 0 / bos):RWKV 训练每轮都以此起始,缺它 state 初始化偏离分布。
  - 结尾 `Assistant: <think>\n</think>`:空 think 标签告诉模型"跳过思考直接答"。缺它模型续写时不知该思考还是直答。
- chat 命令默认模板**改为裸 qa,默认不开 reasoning**(Spec §1.X 裁决:不写死模型版本号,用户知道自己的模型是否 reasoning 类)。G1 系列用户需显式 `--reasoning --think fast`,否则降智;启动 banner 已提示。
- reasoning 方言输出开头常有 `\n`(空 think 标签后的自然换行),`ChatSession._has_reasoning_dialect` 时显示已 `lstrip('\n')`。
- 详细 token 级数值对齐实验(贪心/logprobs/采样分歧)见 **[docs/g1g-decode-alignment.md](docs/g1g-decode-alignment.md)**(历史文档,不回溯改文件名)。

### llama.cpp 对比时的 tokenize 陷阱(踩过)
**llama.cpp 的 `-f` 文件模式、`/tokenize` 端点、`/completion` 端点都不识别 RWKV 的特殊 token** —— 把 `<|rwkv_tokenizer_end_of_text|>` 当普通文本逐字符切成 14 个 token,而不是单个 token 0。**只有 `/v1/chat/completions`(走 jinja chat template)正确处理。**

- 踩坑表现:`-no-cnv -f prompt.txt` 模式下,prompt 被错切成 37 token(应 24),模型续写完全跑飞,每次都生成 `<think>...` 英文思考链。
- 验证方法:看返回的 `prompt_tokens` / `tokens_evaluated` 字段,和 MLX 的 `len(tok.encode(prompt))` 对比。对齐时 "你好" 应都是 17 token、"中国首都" 应都是 24。
- **对比 llama.cpp 推理效果时,务必用 `/v1/chat/completions` + `chat_template_kwargs.enable_thinking=false`,不要用 `-no-cnv -f` 或 `/completion`。**
- server 启动:`./models/llama-b9939/llama-server -m MODEL.gguf -c 2048 -ngl 99 --host 127.0.0.1 --port 8876`,模型只加载一次,比反复重启 `llama-cli`(每次 10s)高效得多。

### 多轮会话 cache 续传 vs 重放(Phase 3 §2,已实施)

InferenceEngine + ChatSession 的多轮改造(2026-07 落地)。核心机制事实:

- **续传 vs 重放的分级(单一判定)**:`continuation_safe = (template == "qa" and not reasoning)`。
  纯 qa 走续传(轮间保留 cache);所有 reasoning 组合(off/fast/on)走重放。
  裁决依据:`docs/g1g-decode-alignment.md §8` 的 g1g 多轮 token 实测 + 附录 D.2。
- **为什么 reasoning 不能续传**:官方 `chat_template`(g1g/g1d/g1h 三模型完全一致)里
  历史 assistant 是**裸内容**(无 think 标签),think 只在当前生成轮。续传会固化首轮的
  think 标签(`<think>\n</think>`),导致历史段 token 偏离训练分布。这是 reasoning 模型
  品类结构性属性(DeepSeek-R1/Qwen3 同惯例),不是 bug。
- **bos 每对话一次**:jinja 里 bos 在 `{% for %}` 之前,多轮不重复。
- **cache 洁净性**:eos/max_tokens 干净(可续传);stop_sequence 脏(token 先喂入前向再检测,
  `\nUser:` 的若干 token 已进 cache)。脏 cache 下轮自动走重放,不修补不回滚。
- **token 账本**:`GenerationResult.display_token_ids`(干净展示文本 token,旧 `token_ids` 改名)
  vs `fed_token_ids`(实际喂入前向的完整 token,含污染)。eos 路径两者相等,stop_sequence
  路径 fed ⊃ display。重放以 history **文本**为准(文本是唯一事实源),不用 fed_token_ids 拼接。
- **官方 chat_template 只有 fast/on 两档,无 off 档**:本产品 `think=off`(裸 `Assistant:`)
  是自定义档,偏离训练分布。官方"不思考"实为 fast 档(空 think 标签)。
- **ChatSession 状态机**:`history: list[Turn]`(按 [user, assistant] 分别记录)、`cache`、
  `cache_clean`。`/rewind [n]` 每轮=2 turn,截断 n 轮=删 2n turn,触发重放。
  `/state` 中途切换=换 S₀=换人设,清空会话。
- **上游生态裁决**:RWKV Runner/Ai00/llama.cpp/HF chat_template 都是"渲染文本是真相,
  cache 是前缀优化"的重放语义。RWKV 的 state 无法按位置截断(只能有限 checkpoint),
  但语义相同。我们 v1 不做 state snapshot,reasoning 多轮每轮全量重放。

---

## ⚠️ 内存单位:全仓统一 GB(÷10⁹),禁止混用 GiB

踩坑:`bytes/1024³`(GiB)和 `bytes/1e9`(GB)混用,会让同一块内存看起来不一致——例如 `12713115648 bytes` 算出来 `11.84`(GiB)vs `12.71`(GB),一旦两套口径同表对比,就出现"active+cache 12.06 超过 working_set 11.84"的假象(其实同口径没超)。

**统一规矩:**
- **一律 GB(÷10⁹)**。换算统一写 `x / 1e9`,不要 `/1024³`。
- 字段命名带单位:`_gb` 后缀 = GB 口径。**禁止**把 GiB 值放进叫 `_gb` 的字段。
- `mx.metal.device_info()` 返回的是 bytes,字段名里若加 `_gb` 必须 `/1e9`。
- `vm_stat` compressor 的 `×16384/1024³` 是历史遗留(接近 GiB),新代码改 `×16384/1e9` 对齐 GB。
- 报告/表头里数字统一标 "G"(意指 GB),不要写 "GiB"。
- **对比 working_set 上限时,active/cache/sum 和 working_set 必须同口径**(都用 GB),否则削顶判定失效。

历史代码里 GiB 口径的(`tools/mem_probe_v2.py` 的 compressor、device_info 字段)暂不回头改(数据已留档),但**新写的内存汇报脚本一律 GB**。

---

## ⏱ 执行环境超时约束(重要)

**单条 Bash 命令最长约 10 分钟硬超时。超时会被 kill,且可能波及后台训练进程。**

实战教训:
- `sleep` 轮询训练进度会触发超时级联,**不要用**。训练放后台后,要么等 task 完成通知,要么让用户手动跑。
- 连续多次 `load_model`(训练+eval+preview 串跑)会让 Metal 内存池累积不释放,叠加出更高峰值。**一次只跑一个重任务。**
- 估算训练时长决定执行方式:
  - **0.4B @ 200条 @ ctx512**:~8 分钟,单条命令内可完成(留余量)。
  - **0.4B @ 200条 @ ctx192**:~8 分钟(步数不变,ctx 只影响单步内存不显著影响速度)。
  - **≥1000 条 或 1.5B 全量**:耗时超 10 分钟 → **输出命令让用户手动跑**,不要自己后台跑后 sleep 等。

**安全做法**:训练命令 + events-file 交给用户跑;用户贴回 stdout / events.jsonl / loss 曲线,你再分析。

---

## 🔒 判据纪律(实验裁决类工作)

**判决判据跑完实验后不许新增或修改。** 需求单里的判据是契约,实验是为了填判据,不是反过来用数据倒推判据。

实验中发现判据有盲点(数据显示某现象,但判据不敏感/测不到):
1. **停下来,不要自行新增判据**。不要把"我觉得应该是 X"包装成正式判据写进分析脚本和报告。
2. **报告盲点**:把"判据说 A,但数据还显示 B,B 可能影响结论"如实写出来,标"提请裁决"。
3. **由用户裁决**是否改判据后重判。裁决通过才能改,裁决前报告结论严格按原判据给。
4. 用户裁决改判据后,新判据写进需求单/报告,注明"经 X 裁决修订",留可追溯链。

**案例存档(max_buffer 事件,2026-07-11):** c4G 对照实验(`exp/precision`,report_c4g.md)中,需求单判决矩阵只看"是否崩",Task2 两版都没崩 → 按判据应判"封存"。agent 自行新增了"step_peak 越 max_buffer_length"判据,并基于 max_buffer 机制叙事把结论改成"红线可抬"——但 max_buffer 机制叙事是错误的(真正机制是 step_peak + 池子残留顶穿削顶线 → 换页,与 max_buffer 无关),且违反了"判据跑完不许改"的纪律。经用户裁决:删 max_buffer 叙事,红线判据改为削顶线反推 + ms/step 拐点法,重算红线绝对值(fp32≈485/bf16≈697);"+50%"相对结论因判据更换前后都成立而保留。教训:**发现盲点先报,判据等裁决。**

---

## 排查技巧备忘

- **grep 格式字面量**:`grep -rn 'f"{[^}]*}\\n' --include="*.py" . | grep -v .venv | grep -v __pycache__` — 应只剩 templates.py(和 experiments 归档的 data_v2.py,已约定不动)。
- **验证编码同构**:PYTHONPATH=src 跑,比对 `tok.encode(prefix)+tok.encode(target)` vs `tok.encode(prefix+target)`。
- **看 loss 曲线**:`grep '"type": "epoch_end"' events.jsonl`。
- **events 文件**:`"w"` 覆盖写(已修,旧版 "a" 追加会混入多次训练)。
- **shell 工作目录**:Bash 工具的 cwd 不持久,`cd` 进子目录后下条命令可能还在那。训练/eval 用绝对路径或先确认 `pwd`。
