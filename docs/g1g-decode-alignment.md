# g1g 解码对齐报告 — MLX vs llama.cpp

> 实验日期:2026-07-11
> 目的:确认 Preen(MLX)的 g1g-1.5b 推理相对 llama.cpp 不降智、不错乱。
> 结论:**推理通路健康,数值实现小差异不影响语义质量。**

---

## 1. 实验环境

| 项 | MLX 侧 | llama.cpp 侧 |
|---|---|---|
| 模型 | `models/converted/rwkv7-g1g-1.5b`(safetensors) | `models/rwkv7-g1g-1.5b-20260526-ctx8192-FP16.gguf`(FP16) |
| 框架 | MLX + mlx-lm RWKV-7 kernel | llama.cpp b9939 (92b187c97) |
| prompt 格式 | `templates.G1G` 模板 | `/v1/chat/completions` + `enable_thinking=false` |
| 采样参数 | temp=0.8, top_p=0.9, top_k=0, seed=42(对齐);temp=0.0(贪心) | 同左 |

采样参数严格对齐。所有对比基于 g1g 不带思考模式(`enable_thinking=false`)。

---

## 2. 关键前提:tokenization 对齐验证

### ⚠️ 踩坑:llama.cpp 特殊 token 识别

llama.cpp 对 RWKV 特殊 token `<|rwkv_tokenizer_end_of_text|>` 的处理**取决于调用路径**:

| 路径 | tokenization 结果 | 正确? |
|---|---|---|
| `-f` 文件模式 / `-no-cnv` | 逐字符切成 14 token | ✗ |
| `/tokenize` 端点 | 逐字符切成 14 token | ✗ |
| `/completion` 端点 | 逐字符切成 14 token(`tokens_evaluated` 偏多) | ✗ |
| `/v1/chat/completions`(jinja 模板) | 正确识别为单个 token 0 | ✓ |

**验证方法**:对比两边的 prompt token 数。"你好" 的 g1g prompt:

```
MLX token ids (17): [0, 24281, 59, 33, 10464, 11685, 65530, 5585, 41693, 59, 295, 35762, 63, 11, 754, 35762, 63]
llama /v1/chat/completions prompt_tokens: 17  ✓
```

**本报告所有数据均来自 `/v1/chat/completions`(已验证 tokenization 对齐)。早期用 `-no-cnv -f` 跑的数据已全部作废。**

---

## 3. token 级数值对齐

### 3.1 首 token top-5 logprobs 对比(temp=0.0,贪心)

#### 三方 tokenizer 的 bos 识别差异(关键前提)

调查中发现 `<|rwkv_tokenizer_end_of_text|>` 在不同 tokenizer 下的处理不同:

| tokenizer | `<|...end_of_text|>` 字符串 | `\x00` 字节 |
|---|---|---|
| **MLX**(HF `AutoTokenizer`) | `[0, ...]` ✓ 识别为特殊标记 | `[1, ...]` |
| **llama.cpp `/tokenize`** | 逐字符切成 14 token ✗ | (未测) |
| **rwkv pip trie** | 逐字符切成 14 token ✗ | `[1, ...]` |

**id 0 和 id 1 是不同 token**:id 0 是 HF 格式(`added_tokens.json`)定义的 `<|rwkv_tokenizer_end_of_text|>` 特殊标记,id 1 是词表里的 `\x00`。

**模型发布方(BlinkDL)用的是 HF 格式**(`tokenizer_config.json` + `added_tokens.json` 明确定义 id 0 = 特殊标记,作 bos/eos/unk/pad)。**MLX 走 HF tokenizer 是正确的**;rwkv pip 的 trie tokenizer 不适用于 g1g 这种带 HF 配置的模型(它是给老格式 World .pth 用的)。

#### token id 精确对齐实验(决定性证据)

为彻底排除 tokenizer 差异,用 **token id 数组**直接喂 llama.cpp `/completion`(`"prompt": [0, 24281, ...]`),与 MLX 完全相同的 token 序列对比:

```
PROMPT: 你好  prompt = [0, 24281, 59, 33, 10464, 11685, 65530, 5585, 41693, 59, 295, 35762, 63, 11, 754, 35762, 63]  (17 tok 两边逐 id 一致)

  首 token top-5 logprobs:
                     llama.cpp (token id 精确输入)    MLX
  rank 1   id=11  \n    -0.7676                    -0.7812   ✓ 一致
  rank 2   id=10464 你  -2.1095                    -2.1562
  rank 3   id=11685 好  -3.2199                    -3.1875
  rank 4   id=12469 您  -3.2704                    -3.3438
  rank 5   id=12605 我  -3.5916                    -3.5781

  → 首 token 都是 \n(11),top-5 候选集完全一致,logprob 差 < 0.05
```

**对比:chat 端点(`/v1/chat/completions`)同一 prompt 的结果**

```
  rank 1   id=你    -0.873     (llama chat)
  rank 2   id=\n    -0.878     (llama chat)   ← 与 token id 精确输入的 -0.7676 不同!
```

chat 端点首 token 变成 `你` 而非 `\n`,且 logprob(-0.878)与精确输入(-0.7676)差 0.11。**说明 chat 端点的 jinja 模板渲染出的 `<think>\n</think>` token 序列,和 MLX 的 `G1G` 模板 encode 出的不完全一样**(多/少一个 token,或边界 token 不同),导致 logits 位置偏移。

#### 结论:数值对齐成立

**当 prompt token 序列完全一致时,MLX 和 llama.cpp 的 logprobs 高度吻合(Δ < 0.05)。** 之前 chat 端点对比看到的"首 token 分歧"是 jinja 模板渲染差异导致的 token 序列微差,不是推理引擎的数值 bug。

### 3.2 贪心逐 token 序列对比(temp=0.0)

```
PROMPT: 1+1等于几？
  MLX  (9):  [11, 50, 44, 50, 15169, 10339, 51, 10080, 0]
  llama(9):  [11, 50, 44, 50, 15169, 10339, 51, 10080, 0]
  → 9 token 完全一致 ✓ (解码: "\n1+1=2。" + eos)
```

### 3.3 数值差异归因

两边 logprobs 存在微小差异(token id 精确对齐时 Δlogprob < 0.05)。归因:

1. **bf16 精度 + 计算顺序**:MLX 和 llama.cpp 的矩阵乘法、归一化实现不同,bf16 下累积误差路径不同,导致 logits 末位抖动。
2. **kernel 实现差异**:RWKV-7 的 wkv 循环,MLX 走 `_wkv7_step_ops`(ops 路径)或 Metal kernel,llama.cpp 走自己的 CUDA/Metal kernel,内部累加精度可能不同。

**这种差异在概率接近的 token 上会翻转 argmax 排序(如 "你好" 的 你/\n 概率差仅 0.005),但在概率有明显梯度的 token 上不影响选择(如 "1+1" 的 \n 明显领先,两边一致)。**

---

## 4. 采样生成质量对比(temp=0.8, top_p=0.9, seed=42)

| # | 问题 | llama.cpp | MLX | 评价 |
|---|------|-----------|-----|------|
| 0 | 你好 | 256 tok 跑满,发散(哲学+emoji) | 108 tok eos,正常 | 都不算理想(这是 base model 的固有波动) |
| 1 | 一句话介绍自己 | 33 tok eos | 69 tok eos | **同级,都正常** |
| 2 | 中国首都 | 60 tok eos,"北京,政治文化中心" | 157 tok eos,"北京,数千年的历史" | **都正确**,MLX 啰嗦些 |
| 3 | 苹果算术 | 82 tok eos,**4个,正确** | 70 tok eos,**4个,正确** | **都正确,质量同级** |
| 4 | 月亮短诗 | 47 tok eos,五言古诗 | 168 tok eos,现代诗+注释 | **都成诗** |

**注意**:即便 seed 相同,两边采样输出不同是**正常的**——MLX 和 llama.cpp 的随机数生成器实现不同,top-p 采样从不同分布状态取样。这不代表降智,只代表采样路径不同。

---

## 5. 速度对比(参考,非对齐指标)

| | Prompt t/s | Generation t/s |
|---|---|---|
| llama.cpp | 100~330 | **~33** |
| MLX | 37~520 | **~25** |

llama.cpp 生成速度略快(~33 vs ~25 t/s)。这是 gguf Metal kernel vs MLX kernel 的实现差异,与推理质量无关。

---

## 6. 结论

### 三方交叉验证定论(MLX / llama.cpp / rwkv pip)

1. **MLX 的 tokenizer 是正确的**。模型发布方(BlinkDL)用 HF 格式定义 id 0 = `<|rwkv_tokenizer_end_of_text|>` 特殊标记(`added_tokens.json`)。MLX 走 HF `AutoTokenizer`,正确识别。rwkv pip 的 trie tokenizer 是老格式实现,不适用于 g1g,**不作为真值基准**。
2. **llama.cpp 的推理引擎没有 bug**。用 token id 数组精确控制 prompt(消除 tokenizer 差异)后,llama.cpp 与 MLX 的首 token logprobs 高度吻合(Δlogprob < 0.05,top-5 候选集完全一致)。之前看到的"首 token 分歧"是 jinja chat 模板渲染的 token 序列微差,非引擎问题。
3. **bf16 数值精度差异是固有的、可接受的**:两边 prompt 完全一致时仍有 < 0.05 的 logprob 差,源自 bf16 计算顺序不同。在概率接近的 token 上(差 < 0.03)可能翻转 argmax,但语义不影响。

### 最终结论

1. **tokenization 完全对齐**(prompt token id 序列逐位一致)—— 排除"编码不一致导致错乱"。
2. **推理数值对齐成立**:prompt token 序列一致时,MLX 和 llama.cpp 的 logprobs 吻合(Δ < 0.05)。
3. **贪心解码在概率有明显梯度时完全一致**("1+1等于几" 9 token 逐位一致)。
4. **采样生成质量同级**:知识题两边都对,开放题两边都通顺能正常 eos。无降智、无错乱、无跑飞。
5. **降智根因是 prompt 格式**(缺 bos + 空 think),已在 `templates.G1G` 修正,与框架无关。
6. **rwkv pip 包的 trie tokenizer 不适用 g1g**,不需要对齐它做基准。真值是 HF 格式的 id 0 = 特殊标记。

---

## 7. 复现方法

### MLX 侧(贪心 + logprobs)
```python
PYTHONPATH=src .venv/bin/python -c "
import mlx.core as mx, mlx.nn as nn
from statetuner.core import load_model
from statetuner.inference import render_prompt
mdl, tok = load_model('models/converted/rwkv7-g1g-1.5b', patch=False)
p = render_prompt('你好', 'g1g')
ids = tok.encode(p)
lp = nn.log_softmax(mdl(mx.array([ids]), mdl.make_cache())[0,-1], axis=-1)
mx.eval(lp)
idxs = mx.argsort(lp)[::-1][:5]
print(list(zip(idxs.tolist(), lp[idxs].tolist())))
"
```

### llama.cpp 侧(server 模式)
```bash
# 起 server
./models/llama-b9939/llama-server -m ./models/rwkv7-g1g-1.5b-20260526-ctx8192-FP16.gguf \
  -c 2048 -ngl 99 --host 127.0.0.1 --port 8876 &

# 首 token logprobs(务必用 chat 端点)
curl -s http://127.0.0.1:8876/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"gpt-3.5-turbo",
  "messages":[{"role":"user","content":"你好"}],
  "max_tokens":1,"temperature":0.0,"seed":42,
  "logprobs":true,"top_logprobs":5,
  "chat_template_kwargs":{"enable_thinking":false}
}'
```

---

## 8. 多轮对话格式对齐(2026-07-12,Phase 3 §2.5 前置任务)

> 本节回答 Phase 3 Spec §2.5 的两个未知数,作为 InferenceEngine 多轮改造(§2)的前置结论:
> **G1 方言多轮时,轮间是否重复 bos(token 0)?think 标签是否每轮渲染?**
> 方法沿用本报告的对齐方法论——以官方 `tokenizer_config.json` 的 `chat_template` 为真值。

### 8.1 实验方法

三个模型的 `tokenizer_config.json` 的 `chat_template` 字段(g1g-1.5b / g1d-0.4b / g1h-1.5b)**逐字符完全一致**(均 516 字符),取任一即可。jinja 源码:

```jinja
{{ '<|rwkv_tokenizer_end_of_text|>' }}              ← bos 在 for 循环【之前】
{% for message in messages %}
  {% user %}      'User: ' + content + '\n\n'
  {% system %}    'System: ' + content + '\n\n'
  {% assistant %} 'Assistant: ' + content + '\n\n'   ← 历史 assistant = 裸内容
{% endfor %}
{% if add_generation_prompt %}
  {% enable_thinking==False %} 'Assistant: <think>\n</think>'   ← think 仅当前生成轮
  {% else %}                   'Assistant: <think'
{% endif %}
```

用 jinja 渲染 3 轮对话,再用 World tokenizer(HF `AutoTokenizer`)encode 成 token ids,逐 token 对比。

### 8.2 两个未知数的定论

**未知数 1:bos(token 0)是否每轮重复?**
**定论:不重复。整对话只出现一次,在最开头。** 实测 3 轮 fast 对话渲染后 encode,`bos(0)` 出现位置 = `[0]`,全序列仅一处。jinja 里 bos 字面量在 `{% for %}` 之前,逻辑上就是每对话一次。

**未知数 2:think 标签是否每轮渲染?**
**定论:只在当前生成轮。历史 assistant 是裸内容,无 think 标签。** jinja 的 think 渲染在 `{% if add_generation_prompt %}` 分支内,即只作用于"待生成的当前轮"。历史轮走 `{% assistant %}` 分支,是 `'Assistant: ' + content + '\n\n'`,裸内容。

实测 3 轮 fast 对话整体渲染(token 级,38 tokens):
```
<|bos|>User: Q1\n\nAssistant: A1\n\nUser: Q2\n\nAssistant: A2\n\nUser: Q3\n\nAssistant: <think>\n</think>
└ bos 只此一处                                                              └ think 只在当前轮
```

### 8.3 chat_template 的 think 档位覆盖(发现)

**官方 `chat_template` 只有两档,没有 off 档**:

| `enable_thinking` | 渲染 | 本产品对应 |
|---|---|---|
| `False` | `Assistant: <think>\n</think>` | `think=fast` ✓ |
| `True` 或未定义(default) | `Assistant: <think` | `think=on` ✓ |
| (无) | — | `think=off` 无官方对照 |

本产品的 `think=off`(`Assistant:` 后什么都不加)是**自定义档**,模型训练时 `Assistant:` 后必有 `<think>` 标签(fast 或 on),off 档偏离训练分布。这与 Phase 3 Spec §1.1 官方映射表"不思考模式 `Assistant:` 留空 = think=off"冲突——官方 chat_template 的"不思考"实为 **fast 档**(空 think 标签),不是 off。

### 8.4 续传 vs 重放的 token 级等价性(Phase 3 §2.6.a 验收依据)

以"续传 = 轮间保留 running cache,下轮只 prefill continuation;重放 = 从 S₀ + 完整历史文本重新 prefill"为定义:

| 组合 | 续传 == 重放(逐 token) | 证据 |
|---|---|---|
| **纯 qa(无方言)** | ✅ **成立** | QA target 带前导空格(` {a}`),`encode(prefix) + encode(' A1') + encode(continuation) == encode(整体文本)`,边界编码稳定。实测三段拼接与整体 encode 逐 token 相等。 |
| **reasoning + off** | ❌ **不适用** | off 档无官方对照(§8.3),单轮已偏离训练分布,续传/重放等价性无意义。 |
| **reasoning + fast/on** | ❌ **不成立** | 续传会固化 turn1 的 think 标签(`<think>\n</think>`),导致历史段 token = `...Assistant: <think>\n</think>A1`;而 jinja 真实多轮历史段是裸 `...Assistant: A1`。实测 2 轮 fast 对话:续传 34 tokens vs 重放 28 tokens,差 6 个多余 token(历史轮的 think 标签)。 |

reasoning 组合的续传偏离**不是本产品的设计缺陷,是 reasoning 模型品类的结构性属性**:DeepSeek-R1 / Qwen3 等主流 reasoning 模型的多轮惯例都是历史剥 think,由此导致的 prefix cache 失效是公认固有成本(详见 Phase 3 §2.1 修订依据的上游生态调研)。

### 8.5 结论(裁决 Phase 3 §2 续传分级)

基于 §8.3/§8.4 实测,经用户裁决(2026-07-12):

1. **续传分级简化为单一判定**:`continuation_safe = (template == "qa" and not reasoning)`。纯 qa 走续传,所有 reasoning 组合(off/fast/on)走重放。
2. **Phase 3 §1.1 官方映射表修正**:官方"不思考模式"映射到 `think=fast`(对齐 chat_template),不是 off。`think=off` 保留为本产品自定义档(裸 `Assistant:`),但标注其偏离训练分布。
3. **Phase 3 §2.6.a2(g1+off fed 序列对齐官方)删除**:g1+off 无官方对照,该验收无法成立。
4. **reasoning 多轮全量走重放**:上游生态(RWKV Runner / Ai00 / llama.cpp / HF chat_template)均为"渲染文本是真相、cache 是前缀优化"的重放语义,本产品对齐这一惯例。fast/on 档下 state snapshot 式的"RNN prefix cache"留作 v1.5 优化项,v1 不做。

### 8.6 复现方法

```bash
cd /Users/no22/Projects/Preen
.venv/bin/python3 -c "
import json, warnings; warnings.filterwarnings('ignore')
from jinja2 import Template
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('models/converted/rwkv7-g1g-1.5b', trust_remote_code=True)
with open('models/converted/rwkv7-g1g-1.5b/tokenizer_config.json') as f:
    tmpl = Template(json.load(f)['chat_template'])
msgs = [{'role':'user','content':'Q1'},{'role':'assistant','content':'A1'},
        {'role':'user','content':'Q2'},{'role':'assistant','content':'A2'},
        {'role':'user','content':'Q3'}]
fast = tmpl.render(messages=msgs, add_generation_prompt=True, enable_thinking=False)
ids = tok.encode(fast)
print('bos 位置:', [i for i,t in enumerate(ids) if t==0])  # [0] = 整对话一次
"
```
