# AGENTS.md — Preen 工作指南

> 本文件给未来 ZCode agent 读。写的是**代码里看不出来、踩过坑才知道**的事。
> 整体规划见 **[docs/RWKV-StateTuner-Roadmap.md](docs/RWKV-StateTuner-Roadmap.md)**(落地路线图、技术决策、风险、内存口径说明)。
> 原理见 [docs/P0-理论指南.md](docs/P0-理论指南.md)。

---

## 这是什么项目

Mac 原生的 RWKV-7 **state tuning** 工具。冻结模型全部权重,只训练每层初始状态矩阵 S₀(64×64)。
当前阶段:P1 完成(CLI + 训练 + 导出),NekoQA 风格迁移已跑通,正在做测试化。

**git 分支注意**:`main` 只有 P0 实验;**P1 正式代码在 `p1-statetuner-cli` 分支**(`src/statetuner/` + 根 `tests/` + 根 `pyproject.toml`)。开始工作前先确认分支对。

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

## ⚠️ 内存事实(排查中,勿下结论)

**这是当前最大的未解问题。** 若排查出具体归因(哪个组件占了多少、能否优化),**记得回填更新本节**,并同步 Roadmap 的内存表与「插队任务」。关键事实:

- **0.4B 训练实测 RSS ≈ 11~12GB**(活动监视器),不是 Roadmap 旧表写的 1.39GB。
- `mx.get_peak_memory()` **严重漏报**——只算 MLX allocator 分配,不含 Metal wired memory。它报 3.35GB,实际 RSS 12GB。**永远以活动监视器 RSS 为准,别信 mx peak。**
- **ctx_len 不是大头**:ctx 从 512→192(砍 62%),RSS 几乎不变(还是 ~12G)。说明大头在某个不随 ctx 变化的常数项,尚未定位。
- 内存行为:启动 ~2-3G → 每 ~10 步增长 → 约 50 步达到**第一个高峰**(11~12G)→ 50 步后增长速度明显放缓,但**是否会持续上涨尚未观察确认**(没跑到足够长)。16GB 机器内存压力大、有 swap,但没 OOM。
- 0.4B 和 1.5B 层数都是 24,state 参数量接近;内存问题两者同源。
- **Roadmap 旧内存表已挂起**,排查结论出来前不引用它做容量结论。详见 Roadmap「⚠️ 插队任务」。

排查技巧:`PYTHONPATH=src .venv/bin/python -c "..."` 里逐组件 `load_model` / `make_state_params` / `value_and_grad`,每个之后用 `resource.getrusage(RUSAGE_SELF).ru_maxrss` 读真实 RSS。

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

## 排查技巧备忘

- **grep 格式字面量**:`grep -rn 'f"{[^}]*}\\n' --include="*.py" . | grep -v .venv | grep -v __pycache__` — 应只剩 templates.py(和 experiments 归档的 data_v2.py,已约定不动)。
- **验证编码同构**:PYTHONPATH=src 跑,比对 `tok.encode(prefix)+tok.encode(target)` vs `tok.encode(prefix+target)`。
- **看 loss 曲线**:`grep '"type": "epoch_end"' events.jsonl`。
- **events 文件**:`"w"` 覆盖写(已修,旧版 "a" 追加会混入多次训练)。
- **shell 工作目录**:Bash 工具的 cwd 不持久,`cd` 进子目录后下条命令可能还在那。训练/eval 用绝对路径或先确认 `pwd`。
