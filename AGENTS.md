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
├── data.py              数据管线 (encode_template_sample 通用 / load_qa_dataset QA)
├── templates.py         ★ 格式模板单一事实源(NEKO_QA)
├── train.py             训练循环(产品化)
├── events.py            结构化事件(JSON lines,为 IPC 铺路)
├── export.py            .pth 导出(RWKV Runner 挂载)
└── cli.py               train/eval/export/preview,带 --template 开关

tests/                   回归测试(改 src 必跑)
├── fixtures/              NekoQA 基准 state(nekoqa_04b_s42.npz,产品 CLI 训练)
└── golden/                推理 golden 快照(nekoqa_*.json)
tools/                   模型转换(convert_rwkv7_to_hf.py)+ 内存探针
experiments/p0_translate/  ★ P0 历史归档,不要动(含已废弃翻译路径,保留可复现性)
train_data/NekoQA_10k/   NekoQA 数据集(Apache-2.0,见 NOTICE.md)
```

**架构铁律:**
- `templates.py` 是**全仓唯一**允许持有 `\n`-拼接格式字面量的地方。其他文件一律 `TaskTemplate.format_prefix/format_target` 派生。改前 grep 确认。
- 训练循环主体(`train.py` 的 `Trainer`)是禁区——不调超参默认值、不动 loss 计算。
- `data.py` 的 `encode_template_sample`(通用)是基础,`load_qa_dataset`(NekoQA q/a)派生自它。新任务加 loader,别 fork。
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
  --template nekoqa \
  --out state.npz --events-file events.jsonl \
  --lr 0.01 --epochs 3 --ctx-len 512 --no-early-stop --seed 42

# 生成/预览(--template nekoqa|raw)
PYTHONPATH=src .venv/bin/python -m statetuner.cli preview \
  --model models/converted/rwkv7-g1d-0.4b --state state.npz \
  --prompt "你好" --template nekoqa --ab
```

`--template` 决定数据 loader 和 prompt 渲染(训练/推理必须同模板,保证编码同构)。NekoQA smoke 全流程脚本:`scripts/nekoqa_smoke.sh`。

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

### 历史观察(已解释)
- 0.4B 训练实测 RSS ≈ 11~12GB:大头是 fp32 state 在 bf16 权重的 wkv 循环里把整个循环提升成 fp32(MLX 类型提升规则 bf16+fp32=fp32)。这是机制事实,非 bug。
- ctx_len 不是大头:ctx 512→192 RSS 几乎不变,因大头在不随 ctx 变化的 state 张量提升。
- 0.4B 和 1.5B 层数都是 24,state 参数量接近;内存问题两者同源。

排查技巧:`PYTHONPATH=src .venv/bin/python -c "..."` 里逐组件 `load_model` / `make_state_params` / `value_and_grad`,每个之后用 `resource.getrusage(RUSAGE_SELF).ru_maxrss` 读真实 RSS。

---

## g1g 推理对齐(已验证)

### g1g prompt 格式 = 降智/错乱的分水岭
- **RWKV7-G1 是 reasoning 模型**,prompt 格式必须带 `<think>` 标签。`templates.py` 的 `G1G` 模板 = `<|bos|>User: {q}\n\nAssistant: <think>\n</think>`,对齐官方 chat_template 的 `enable_thinking=False` 渲染。
- **实测铁证**:raw 格式(`User: 你好\n\nAssistant:`)→ 120 token 跑满、幻觉自报"ChatGPT";G1G 格式 → 39 token 正常 eos、自然回答。**降智根因是缺 bos + 空 think 标签,不是模型或框架问题。**
- 两处关键段缺一不可:
  - 开头 `<|rwkv_tokenizer_end_of_text|>`(token 0 / bos):RWKV 训练每轮都以此起始,缺它 state 初始化偏离分布。
  - 结尾 `Assistant: <think>\n</think>`:空 think 标签告诉模型"跳过思考直接答"。缺它模型续写时不知该思考还是直答。
- chat 命令默认模板已从 `nekoqa` 改为 `g1g`(产品主线是 g1g 原生对话;猫娘风格迁移用 `--template nekoqa` 显式切换)。
- g1g 输出开头常有 `\n`(空 think 标签后的自然换行),`ChatSession` 显示时已 `lstrip('\n')`。
- 详细 token 级数值对齐实验(贪心/logprobs/采样分歧)见 **[docs/g1g-decode-alignment.md](docs/g1g-decode-alignment.md)**。

### llama.cpp 对比时的 tokenize 陷阱(踩过)
**llama.cpp 的 `-f` 文件模式、`/tokenize` 端点、`/completion` 端点都不识别 RWKV 的特殊 token** —— 把 `<|rwkv_tokenizer_end_of_text|>` 当普通文本逐字符切成 14 个 token,而不是单个 token 0。**只有 `/v1/chat/completions`(走 jinja chat template)正确处理。**

- 踩坑表现:`-no-cnv -f prompt.txt` 模式下,prompt 被错切成 37 token(应 24),模型续写完全跑飞,每次都生成 `<think>...` 英文思考链。
- 验证方法:看返回的 `prompt_tokens` / `tokens_evaluated` 字段,和 MLX 的 `len(tok.encode(prompt))` 对比。对齐时 "你好" 应都是 17 token、"中国首都" 应都是 24。
- **对比 llama.cpp 推理效果时,务必用 `/v1/chat/completions` + `chat_template_kwargs.enable_thinking=false`,不要用 `-no-cnv -f` 或 `/completion`。**
- server 启动:`./models/llama-b9939/llama-server -m MODEL.gguf -c 2048 -ngl 99 --host 127.0.0.1 --port 8876`,模型只加载一次,比反复重启 `llama-cli`(每次 10s)高效得多。

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
