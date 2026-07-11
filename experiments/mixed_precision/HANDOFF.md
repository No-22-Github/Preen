# 交接文档:混合精度(选项 D)实验 — 大满贯脚本跑完后怎么接

> **写给**:跑完 `run_full.sh` 后新开的会话(本会话不再继续)
> **分支**:`exp/mixed-precision`
> **最后 commit**(跑大满贯前):`76d1669`
> **日期**:2026-07-11

---

## 1. 这件事的来龙去脉(一分钟版)

**目标**:决定 RWKV-7 state tuning 的训练精度该不该从 fp32 切 bf16(选项 D)。

**推理链条**(按顺序读这些文档能复原全部上下文):

1. **`docs/P1-内存排查报告-v2.md`** — 起点。训练内存异常排查,核心发现:cache 被 GPU working_set 软上限(~12.7G)削顶,真实需求被盖住。提出四个缓解选项(A checkpointing / B cache_limit / C 数据长度预检 / D bf16 混合精度)。**D 是当前在验证的**。
2. **`docs/decision-precision.md`** — 官方精度确认 + D 方案的安全感口径。**必读**。关键结论见下。
3. **`experiments/mixed_precision/report.md`** — D 第一轮实验(数值冒烟/训练质量A/B/无限档内存)。结论:数值过关、质量等价、无限档被削顶盖住看不出收益。
4. **`experiments/mixed_precision/report_c4g.md`** — D 第二轮(c4G 限额下步内峰值对照)。结论:bf16 在长样本步峰值余量显著(靶心 273 样本步差 1.71G),可抬红线。
5. **`AGENTS.md`** — 项目工作指南。**必读**,尤其「🔒 判据纪律」「⚠️ 内存单位」「⏱️ 执行环境超时」三节。

**D 方案是什么**:S₀(可训练初始状态)保持 fp32 master,前向里 state 进 wkv7 递归前 cast 成 bf16,循环全程 bf16,梯度经 astype 的 VJP(恒等)流回 fp32 master 更新。实现见 `bf16_patch.py`。

**当前裁决状态**:D 已过数值关 + 质量关 + 内存关(c4G 下)。大满贯脚本是**最后一步**:多 seed 重复验证 + 红线标定,全绿后"切默认"另开单执行。

---

## 2. 大满贯脚本(`run_full.sh`)产出了什么

用户跑的命令:`caffeinate -i .venv/bin/python experiments/mixed_precision/run_full.py`

(原 run_full.sh 因 shell IFS 解析 bug 已弃用,改用 Python 版 run_full.py)

### 产物位置:`experiments/mixed_precision/data/`

| 产物 | 路径 | 内容 | 对应需求 |
|---|---|---|---|
| **延长数值冒烟(真实w,700步)** | `data/smoke_15b_700_realw.json` | bf16 递归 700 步误差曲线 + 按 w 分桶拆误差 | 验证边界延长到产品红线 |
| **w 分布 dump** | `data/w_dist_04b.json` `data/w_dist_15b.json` | 各层 w 衰减系数真实分布(fp32 路径) | "离 bf16 危险区多远" |
| **矩阵 10 组** | `data/matrix/<label>/` | 每组 7 项留档(见下) | 多 seed 重复验证 |
| **红线标定** | `data/matrix/redline_result.json` | bf16+c4G 逐桶(L450~700)step_peak,找断点 | 样本长度实测上界 |
| **汇总** | `data/matrix_summary.txt` | 配对判定表 + 余弦 + 红线 + 十问 | 自动分析输出 |
| **完整日志** | `data/matrix_run.log` | 全程 stderr | 排查用 |

### 矩阵 10 组的命名和内容

组目录命名:`<模型>_s<seed>_<精度>`,如 `04b_s42_fp32`、`15b_s1042_bf16`。

| 模型 | seed | 精度 | 组数 |
|---|---|---|---|
| 0.4B | 42 / 1042 / 2042 | fp32 + bf16 | 6 |
| 1.5B | 42 / 1042 | fp32 + bf16 | 4 |

**每组目录内的 7 项留档**(缺一项该组作废):

| 文件 | 内容 |
|---|---|
| `events.jsonl` | header(commit/seed/working_set/数据hash) + 逐step loss + epoch_end(真实avg) + final |
| `state.npz` | 训练产物 state(P0 内部格式 layer_{i}) |
| `state.pth` | RWKV Runner 可挂载格式(回归验证导出器在两精度上都正常) |
| `decode.json` | 十问贪心解码输出 + 循环检测 + 自发终止统计 |
| `mem_trace.json` | 三口径内存(active/cache/active+cache)+ step_peak + compressor |
| (events.jsonl final 里) | 逐层 std、ms/step、max_step_peak |

---

## 3. 跑完后怎么分析

### 第一步:跑现成的分析脚本

```bash
.venv/bin/python experiments/mixed_precision/matrix_analyze.py experiments/mixed_precision/data/matrix
```

它会输出:
- **配对判定表**(5 格红/绿)— 每格判据:**loss 相对差 <2% 且 两版十问均 0 循环 且 全部自发终止**。std 只记录不设阈值。
- **state 余弦基线** — 跨精度同 seed(fp32_s42 vs bf16_s42)+ 同精度跨 seed(fp32_s42 vs fp32_s1042),给 D 报告里 0.305 那个数一个参照系。
- **红线标定**(如果 redline_result.json 存在)— 逐桶 step_peak + 断点。
- **十问原文并排** — 只对 seed42 并排,**风格判读留给用户**,你只标客观差异。

### 第二步:重点看这几个数

**配对判定表**(最关键):
- 全 5 格绿 → bf16 默认化可推进,另开单"切默认"
- 有红格 → 看红在哪(loss 差/循环/终止),报用户裁决

**延长冒烟的 w 分桶误差**(`smoke_15b_700_realw.json` 的 `w_bucket_analysis`):
- 这是本会话最深的发现。真实 w 下误差**持续走阔不饱和**(200步 0.4%→3.4%),推翻了合成 w 的"30步饱和"假象。
- 700 步的终点误差是多少?如果 <10%,bf16 红线由内存定;如果 >10% 且行为 A/B 也退化,bf16 真实红线是数值的,产品红线取两者更紧的。
- 高 w 通道 vs 低 w 通道走阔速率:200 步时几乎一样(裁决了"w 接近1会让误差爆炸"是错的),700 步是否分化?

**红线标定**(`redline_result.json`):
- 找 `first_break_bucket` — bf16+c4G 在 16GB 的样本长度实测上界。
- 断点定义(预写,跑完不许改):step_peak 顶削顶线 / compressor 持续>8G / 单步>30s。

### 第三步:写报告

参考 `report.md` 和 `report_c4g.md` 的结构。报告要点:
- 配对判定结果(全绿/有红)
- 延长冒烟结论(700步终点误差 + 分桶趋势)
- 红线标定结果(实测上界)
- state 余弦基线(同精度跨 seed vs 跨精度同 seed,回答"bf16 偏离是否在正常 run-to-run 方差内")
- 裁决建议(进预检策略/封存,给理由,不替用户定)

---

## 4. 关键文档引用(新会话必读)

| 文档 | 作用 | 读哪段 |
|---|---|---|
| `AGENTS.md` | 项目铁律 | 「🔒 判据纪律」「⚠️ 内存单位」「⏱️ 执行环境超时」三节,必读全文 |
| `docs/decision-precision.md` | D 方案的 claim 口径 + 官方精度出处 | 「安全感的口径」段(claim 建在质量实测不建在误差)+ 「w 分布实测」段 |
| `docs/P1-内存排查报告-v2.md` | 内存问题根因(削顶机制)+ 四选项 | §2 削顶假设验证 + §6 四选项 |
| `experiments/mixed_precision/report.md` | D 第一轮(数值/质量/无限档内存) | §0 结论先行 |
| `experiments/mixed_precision/report_c4g.md` | D 第二轮(c4G 步内峰值) | §1 红线判据 + §4 根因诊断 |
| `docs/RWKV-StateTuner-Roadmap.md` | 整体路线图 | 「⚠️ 插队任务」内存排查 |

---

## 5. 绝对不能违反的规矩(踩过坑的)

### 判据纪律(AGENTS.md 已立规,最高优先级)

**判决判据跑完实验后不许新增或修改。** 需求单里的判据是契约。发现判据有盲点:
1. 停下来,不自行新增判据
2. 报告盲点(标"提请裁决")
3. 等用户裁决是否改判据后重判

**案例(本会话的 max_buffer 事件)**:c4G 实验中 agent 自行新增"step_peak 越 max_buffer"判据,还编了错的机制叙事,被用户打回。AGENTS.md 有完整存档。

### 内存单位

**全仓统一 GB(÷10⁹),禁止 GiB 混用。** `bytes/1024³`(GiB)和 `bytes/1e9`(GB)混用会让同块内存显示不一致(12.71G vs 11.84G),削顶判定失效。AGENTS.md 有完整规矩。

### 引用规矩

拉源码实读,不引二手教程。行号 + 原文引用。`decision-precision.md` 是范例。

---

## 6. 本次沟通中 agent 要避免的习惯(用户反馈)

以下是本会话中 agent 犯过的错,新会话务必避免:

### 6.1 不要用未经验证的假设当结论的支柱

**案例**:decision-precision.md 初版写"w 不极端接近 1,所以 bf16 安全"——这是假设不是观测。实测后 w 大量 >0.999(21.9% 通道),假设被推翻。**任何用作结论前提的"事实",要么有出处要么有实测,不许假设。**

### 6.2 测什么就让数据流过什么(镜像坑)

**案例**:dump w 分布时直接 hook `_wkv7`,但模型权重是 bf16,w 经 cast 回 bf16 后 hook 到的是**量化产物**不是真实分布——把 0.9997 舍成 1.0,制造"p95=1.0"假象。要测原始分布必须走 fp32 路径。**dump/测量前先确认数据走的哪条精度路径。**

### 6.3 安全感不要建在误差数字上

**案例**:初版 claim"bf16 数值安全因为误差只有 1%"——但误差随序列线性增长(不饱和),换个长度数字就变。用户要求:claim 建在**训练质量行为 A/B 实测等价**上,误差曲线只作预警指标。**这样任何人拿误差数字质疑都打不到 claim。** 见 `decision-precision.md`「安全感的口径」段。

### 6.4 不要在报告完成后新增判据

**案例**:c4G 报告写完后,agent 发现"是否崩"的判据有盲点(没崩但 step_peak 逼近上限),自行加了 max_buffer 判据改了结论。**这是违规。** 发现盲点 → 停 → 报告 → 等裁决。

### 6.5 不要编造机制叙事

**案例**:c4G 报告初版写"step_peak 越 max_buffer_length 导致 buffer 拆分"——这是 agent 臆造的机制,真实原因是 step_peak + 池子残留顶穿削顶线 → 系统换页。**不确定的机制就说不确定,不要编。只报实测数据,机制解释要么有出处要么标"推测"。**

### 6.6 速度结论要考虑热条件

**案例**:c4G 实验里 bf16 先跑(占冷机)、fp32 热机垫后,报"bf16 快 19%"——但冷热不可比,这 19% 拆不出热节流。**跨组/跨 session 不做速度结论。同 session 背靠背才能比,且要说明顺序。**

### 6.7 工程实现要验证再交付,别留半成品

**案例**:run_all.sh 的重定向 bug(把 python stdout 吃进 log,外层 >file.json 无效)导致汇总崩溃;matrix_analyze 缺配对时误报全绿。**交付脚本前用最小配置冒烟每个组件。**

---

## 7. 如果跑崩了 / 产物不全

- **某组训练失败**:`run_full.sh` 设计成失败如实记录继续下一组。看 `data/matrix_run.log` 找 `⚠️ 失败`。该组目录不会有 state.npz,重跑脚本会自动跳过已完成的、补跑缺失的。
- **红线标定全绿没断点**:说明最长桶(L700)都没顶到削顶线,需要加更长桶。如实报告"未测到断点",不要强行下结论。
- **延长冒烟 verdict=DEAD**:bf16 递归 700 步发散(nan/inf 或误差>1)。这是有效结果——说明 D 在长序列下数值不安全。如实报告,影响裁决。
- **matrix_analyze 报"缺 bf16/fp32"**:某组没跑完。重跑脚本补齐。

---

## 8. 下一步(全绿后)

如果矩阵全绿 + 红线标定有断点 + 延长冒烟没发散:

**另开一个"切默认"的需求单**(本会话不做),改动清单已在 `report_c4g.md` §5:
- `src/statetuner/core.py`:`patch_rwkv7_for_train` 加 dtype 参数
- `src/statetuner/train.py`:Trainer 加 precision 字段
- `src/statetuner/cli.py`:train 加 `--precision auto|fp32|bf16`
- P4 预检:数据 max token → 自动选 precision
- `tests/`:bf16 路径等价性快测

**不改的**:`make_state_params`(S₀ 始终 fp32)、`export.py`(导出 fp32)、`generate`(推理 kernel 路径)。
