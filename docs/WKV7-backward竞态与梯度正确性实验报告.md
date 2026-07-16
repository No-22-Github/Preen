# WKV7 Metal backward 竞态与梯度正确性实验报告

> 日期：2026-07-16
>
> 项目：Preen / RWKV-7 State Tuning
>
> 涉及上游：[RafaelUI/rwkv-metal](https://github.com/RafaelUI/rwkv-metal)
>
> 状态：根因已确认，本地与上游修复均已提交；正确性、确定性和负对照已完成

## 1. 摘要

Preen 在提交 `fe0c493a605e2b508171c3ebae76dc1ee818f72c` 落地训练提速后，使用
G1H 1.5B、NekoQA smoke 200 和默认学习率训练两轮得到的 State，出现多轮生成不终止、
整句复读和动作模板循环。训练 loss、held-out loss 与 State std 均处于正常范围，常规
训练指标没有暴露异常。

排查最终确认：WKV7 Metal backward 在相邻反向时间步之间缺少一道
`threadgroup_barrier`。每个线程完成当前时间步的 `C_row` 更新时，需要读取完整的共享
`w_sh/a_sh`；下一时间步却会立即由各线程覆盖这些 shared arrays。不同 SIMD group 进度
不一致时，较快 group 会写入下一步数据，而较慢 group 仍在读取当前步，形成跨时间步的
写后读数据竞争。

该竞态并非 `fe0c493` 新增，而是从 `rwkv-metal` 初始实现继承，并在 Preen 最初移植
checkpoint kernel 时原样带入。`mx.compile` 与 bf16 激活直读改变了 GPU 调度、寄存器和
内存时序，使潜伏竞态更容易影响训练轨迹；它们本身没有改变 kernel 数学语义。补齐 barrier
后，bf16 直读与旧 fp32 物化路径的 loss、State 梯度逐元素一致，因此两项加速可以保留。

最终验收采用三层口径：

1. **确定性**：相同输入重复执行 Metal backward，七个梯度必须 bitwise identical；
2. **正确性**：Metal 与 CPU einsum reference 的七梯度最大相对误差必须小于 `1e-5`；
3. **负对照**：临时移除 barrier 后，同一确定性测试必须能重新抓到梯度漂移。

最终七梯度最坏相对误差为 `2.03e-6`，关键的 `dh_in` 为 `1.73e-7`；移除 barrier 后，
第 39 次重复 backward 检出 `dw/dk/dv/da/db/dh_in` 不一致。由此形成了从用户行为、
真实模型梯度、源码竞态、修复、跨实现 golden metric 到负对照的完整证据链。

需要单独保留一个数值边界：checkpoint backward 通过除以 `w` 逆向重建 chunk 内历史
state。当 synthetic decay 接近零时，该逆过程本身会病态放大舍入误差；这是独立于 barrier
的数值适用范围，本文没有宣称已经解决。

## 2. 问题背景

### 2.1 用户侧基准

历史经验是：

- 模型：RWKV7 G1H 1.5B；
- 数据：`train_data/NekoQA_10k/nekoqa_smoke_200.json`；
- 模板：`qa`；
- 学习率峰值：`1e-4`，floor：`1e-5`；
- 两轮训练后，State 应能明显注入猫娘口癖、括号动作与“主人”称呼；
- State 加载后多轮会话应保持基本连贯；少量事实错误或偶发复读可归因于 1.5B 基座能力，
  但不应出现持续胡言乱语或大段模板循环。

### 2.2 可疑提交

`fe0c493` 同时引入两项训练提速：

1. 将 masked loss 与 backward 通过 `mx.compile` 复用训练图；
2. WKV7 checkpoint kernel 不再预先物化六份 fp32 激活，而是直接读取 bf16 模型激活，
   在 Metal kernel 内提升为 float 进行重建和累加。

该提交没有修改训练步数、学习率、梯度裁剪、Adam 更新、数据模板或 State 导出格式。
因此初步怀疑集中在：编译图捕获、混合精度输入、Metal custom VJP 或 GPU 调度时序。

## 3. 第一阶段：行为回归复现

### 3.1 固定条件

除拆分变量外，实验固定为：

| 项目 | 值 |
|---|---|
| 模型 | `models/converted/rwkv7-g1h-1.5b` |
| 数据 | NekoQA smoke 200 |
| 数据划分 | 180 train / 20 held-out |
| 模板 | `qa` |
| epoch | 2 |
| seed | 42 |
| ctx_len | 512 |
| Metal chunk | 16 |
| 学习率 | `1e-4 → 1e-5` cosine |
| warmup | 20（历史对照口径） |
| 推理 | temperature 0、top_p 0.9、max_tokens 100、无重复惩罚 |

确定性三轮问题：

1. `主人今天很累，你能哄哄我吗？`
2. `谢谢你。刚才我为什么难过？`
3. `那明天早上记得叫我起床，先说晚安吧。`

### 3.2 四象限结果

| 训练路径 | final loss | held-out | State std | 耗时 | 三轮停止原因 | 行为 |
|---|---:|---:|---:|---:|---|---|
| 提交前 eager + fp32 物化 | 2.2748 | 2.2165 | 0.003442 | 81.3s | EOS / EOS / EOS | 三轮连贯 |
| 提交后 compile + bf16 直读 | 2.2618 | 2.1875 | 0.003518 | 108.3s | EOS / max / max | 第三轮整段复读 |
| compile + fp32 物化 | 2.2617 | 2.1973 | 0.003443 | 107.8s | EOS / max / max | 第三轮整句重复 |
| eager + bf16 直读 | 2.2860 | 2.2218 | 0.003380 | 114.4s | max / max / max | 动作模板循环 |

墙钟耗时受 allocator、温度和当时系统负载影响，本表用于记录实验，不作为严格性能判决。

### 3.3 第一阶段结论

- 四组 loss、held-out 与 std 都看起来正常；其中退化组的 loss 甚至更低。
- loss 等价不能替代生成行为验收。
- compile 和 bf16 直读分别启用时都可能复现劣化，初看不像单一精度舍入问题。
- 两项优化都会改变 kernel 调度，因此“两个变量分别复现”也可能由共同的潜伏竞态解释。

## 4. 第二阶段：从行为问题转向 backward 梯度

### 4.1 为什么优先检查 backward

训练 loss 每次都稳定，说明同一 State 下 forward 没有明显随机性。State tuning 只训练每层
初始状态 `S₀`，其学习信号完全依赖 Metal custom VJP 返回的 `dh_in_out`。如果 backward
发生轻微数据竞争：

- 当前 step 的 loss 仍可能完全正常；
- 梯度方向会随 GPU 调度漂移；
- Adam 会在数百步中累计这些错误更新；
- 最终 State 可能保持很小的 std，却把模型推入重复或不终止区域。

因此“loss 正常、State 行为异常”与 backward-only 错误高度吻合。

### 4.2 真实 G1H 梯度重复探针

在相同模型、样本、State 和 seed 下重复执行 loss + State gradient：

- `fe0c493` 版本的 eager 重复梯度相对漂移约 `1.37%`；
- compiled 重复梯度约漂移 `1.1%–1.65%`；
- 父提交 `7d3010c` 使用 fp32 物化时也出现约 `1.27%–1.89%` 漂移；
- 个别两次调用会偶然 bitwise identical，但继续重复后仍会分叉；
- 所有调用的 loss 保持一致。

这证明竞态早于 `fe0c493` 存在；该提交改变的是竞态显现概率与错误轨迹，不是首次引入错误。

## 5. 根因定位：反向时间步边界缺失 barrier

### 5.1 问题代码

每个 backward thread 持有一行局部 `C_row`，并通过 threadgroup arrays 共享当前 token 的
`r/w/k/v/a/b` 等向量。时间步末尾执行：

```metal
for (uint dk=0; dk<HEAD_SIZE_C; dk++)
    C_row[dk] = C_row[dk] * w_sh[dk] + dsa_dv * a_sh[dk];
```

这里每个线程都要读取完整的 `w_sh[0..63]` 与 `a_sh[0..63]`。循环回到下一时间步开头后，
每个线程立即执行：

```metal
w_sh[dv] = w[next_base + dv];
a_sh[dv] = a[next_base + dv];
```

原实现两者之间没有 `threadgroup_barrier`。

### 5.2 竞态机制

一个 64-thread threadgroup 通常跨多个 SIMD group。SIMD group 内线程同步执行，并不代表
不同 SIMD group 在没有 barrier 时也同步：

```text
较慢 SIMD group: 读取当前时间步 a_sh[0..63] ───────────────┐
                                                        │ 数据被覆盖
较快 SIMD group: 完成 C_row → 进入下一步 → 写 a_sh[dv] ────┘
```

结果是某些线程的 `C_row` 可能混合读取两个时间步的数据。后续所有依赖 `C_row` 的梯度都会
被污染，尤其包括最终写入 `dh_in_out` 的 State 梯度。

### 5.3 上游关系

- 上游 `rwkv-metal` 的 full-sequence checkpoint backward 与 legacy chunked backward
  都存在相同缺口；
- `git blame` 显示该结构从上游初始提交 `d331ac7` 即存在；
- Preen 在 `fb97148` 移植 kernel 时保留了相同源码结构，只额外暴露可训练 `h_in`；
- 所以这是上游继承 bug，不是 Preen 移植时删除了已有 barrier；
- Preen 因为只训练 `S₀`，对 `dh_in_out` 错误尤其敏感。

## 6. 修复

在两个 backward kernel 的每个反向时间步末尾加入：

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
```

位置必须在最后一次 `C_row` 更新之后、下一时间步覆盖 shared arrays 之前。所有线程执行相同
的 chunk/time 循环，不存在部分线程绕过 barrier 的分支，因此不会引入 barrier divergence。

该修复不改变：

- WKV7 递推公式；
- fp32 累加语义；
- bf16 激活存储；
- State 张量方向或格式；
- 训练步数、优化器和学习率；
- 推理路径。

## 7. 本地修复验证

### 7.1 修复前失败、修复后通过

新增真实 Metal kernel 回归测试后，未修复版本在第二次 eager backward 即出现不同的
`S₀` 梯度。补 barrier 后：

- eager 连续三次梯度逐元素一致；
- compiled 连续三次梯度逐元素一致；
- eager 与 compiled 结果逐元素一致；
- bf16 直读与先转 fp32 物化的 loss、`S₀` 梯度 bitwise identical。

因此 compile 与 bf16 直读可以继续保留，无需回滚加速补丁。

### 7.2 Preen 测试

```text
WKV7 专项：9 passed
全仓快速测试：258 passed, 12 skipped
```

Preen 已同步上游最终 golden 口径：使用 T=64 跨越 CHUNK=32 边界、非零
`h_in`、同时包含 `out` 与 `h_out` 随机投影的 loss，对七个梯度执行 CPU einsum
reference 的 `rel_err < 1e-5` 正确性断言，并对七梯度做 compiled backward bitwise
确定性断言。同时保留 Preen 特有的 bf16 直读、fp32 物化和 eager/compiled State
梯度对照。七梯度最坏相对误差仍为 `2.03e-6`，`dh_in` 为 `1.73e-7`。

Preen 修复提交：

```text
81fdf29 fix(train): 修复 WKV7 backward 时间步竞态
```

### 7.3 两轮训练复测

| 修复后口径 | final loss | held-out | State std | 耗时 |
|---|---:|---:|---:|---:|
| 历史 warmup=20 | 2.2781 | 2.2014 | 0.003430 | 91.9s |
| 当前默认 warmup=50 | 2.2839 | 2.2325 | 0.003346 | 96.4s |

单次墙钟结果没有显示 barrier 吃掉主要加速收益，但不能据此宣称 barrier 额外提速。

### 7.4 生成行为复测

当前默认 warmup=50 的修复后 State：

- 严格贪心、无重复惩罚、100 token：`EOS / EOS / max`；
- 产品默认聊天配置（采样 + 重复惩罚）：`EOS / EOS / EOS`；
- 三轮内容连贯，猫娘动作与称呼明显；
- 第二轮曾把一段 `User/Assistant` 角色字样写进正文，归为 1.5B 模型能力问题。

严格判据没有完全回到历史 State 的 `EOS / EOS / EOS`，本文不修改判据，也不把它描述成
完全通过。需要注意：历史“好 State”本身也由带竞态的旧 kernel 训练得到，它对应一次偶然的
非确定性梯度轨迹。修复正确性后，不应期待逐步或逐 token 复现该旧轨迹。产品默认路径已恢复
为连贯可用，但更严格的无惩罚生成仍可暴露 1.5B 基座的复读敏感性。

## 8. 上游 golden test 设计

### 8.1 为什么必须暴露 stateful API

上游原 `make_wkv7_checkpoint` 将 `h0` 固定为零，公开函数只返回 `out`。这无法覆盖：

- 非零初始状态；
- `h_out` 对 loss 的贡献；
- 非零 `d_h_out` 初始化 `C_row` 的路径；
- `dh_in_out`。

为避免在测试中复制 custom VJP，新增向后兼容的底层入口：

```text
make_wkv7_checkpoint_with_state(...)
wkv7_train_py_with_state(...)
```

二者均接受 `(r, w, k, v, a, b, h_in)` 并返回 `(out, h_out)`。原有零 state API 保持不变，
且实测 legacy factory 与 stateful factory 在 `h_in=0` 时输出 bitwise identical。

### 8.2 正确性指标

对七个梯度张量：

```text
dr, dw, dk, dv, da, db, dh_in
```

分别计算：

```text
tensor_rel_err = max|g_metal - g_ref| / max|g_ref|
rel_err = max_over_tensors(tensor_rel_err)
```

锁定判定线：

- `< 1e-5`：fp32 正确性通过；
- `1e-5 ~ 1e-3`：存在系统性偏差，必须调查；
- `> 1e-3`：直接判错。

跨实现不要求 bitwise，因为 reduction 顺序不同；bitwise 只用于同一 Metal 实现重复执行的
确定性判定。

### 8.3 测试输入

| 项目 | 设置 | 目的 |
|---|---|---|
| B/T/H/D | `2 / 64 / 4 / 64` | T=64 跨两个 CHUNK=32 |
| dtype | fp32 | 对齐 `1e-5` 判定线 |
| h_in | 非零随机，std 0.1 | 覆盖 State tuning 主路径 |
| loss | `(out*p_out).sum() + (h_out*p_h).sum()` | 同时产生随机 `d_out` 与 `d_h_out` |
| a | 每头 L2 归一化 | 对齐模型中的 `a=-kk` |
| b | `-a * κ`，κ∈[0.9,1.1] | 对齐 delta-rule 结构 |
| w | `exp(-exp(normal*0.5 - 2.5))` | 非 uniform、约 0.58–0.99 |

## 9. Reference 精度调查与 CPU 仲裁

### 9.1 初始 GPU einsum 对比出现约 1e-3 偏差

最初直接在 GPU 上运行 MLX einsum reference，即使使用最简单的：

```text
w=1, k=v=a=b=0
```

仍看到 `dw/dh_in` 约 `6e-4–8e-4` 的跨实现差异。此时 recurrence 没有复杂 delta-rule，
`dh_in` 还有简单闭式解，因此不能直接认定 Metal 错误。

### 9.2 闭式坐标复核

简单案例下：

```text
dh_in = p_h + Σ_t (p_out_t ⊗ r_t)
```

在最大差异坐标 `(0, 3, 4, 39)` 上：

| 实现 | 数值 |
|---|---:|
| Metal backward | -7.6213737 |
| MLX GPU einsum/autograd | -7.6157751 |
| MLX CPU einsum/autograd | -7.6213741 |
| NumPy fp64 闭式 | -7.62137436 |

Metal 与 fp64 闭式误差约 `7.1e-7`，CPU einsum 与 fp64 同样处于 `1e-7` 量级；GPU einsum
偏约 `5.6e-3`。因此此前约 `1e-3` 的“系统偏差”主要来自 GPU einsum reduction，而不是
Metal backward。

正式 golden reference 改在 MLX CPU stream 上运行，仍使用同一 einsum 公式和 autograd，
但避免 GPU throughput 优化掩盖 `1e-5` 判定。

### 9.3 七梯度最终结果

| 梯度 | 最大相对误差 |
|---|---:|
| dr | 1.77e-6 |
| dw | 1.82e-6 |
| dk | 2.18e-7 |
| dv | 2.97e-7 |
| da | 2.03e-6 |
| db | 3.96e-7 |
| dh_in | 1.73e-7 |

最坏项为 `da=2.03e-6`，通过 `1e-5` 判定线。

## 10. Synthetic decay 与 checkpoint 数值边界

### 10.1 未截断示例的实际分布

直接使用：

```python
w = mx.exp(-mx.exp(mx.random.normal(shape) - 0.5))
```

在本测试 32,768 个采样点中，实测最小值约 `1.62e-12`，并非描述中的约 `0.3`。即使将
结果强行 clamp 到 `0.3`，checkpoint backward 的：

```metal
hp = (...)/w_sh[dk]
```

在一个 32-token chunk 内仍可能把舍入误差按近似 `(1/w)^32` 放大。此时依赖历史 state
重建的 `dr/dw/da` 会严重偏离，而不依赖该重建的若干梯度仍可保持较小误差。

### 10.2 最终输入选择

测试保留 RWKV 风格的双指数非均匀 decay，但给 logits 使用强负偏置与较小方差：

```python
w = mx.exp(-mx.exp(mx.random.normal(shape) * 0.5 - 2.5))
```

实测范围约 `0.577–0.989`，在该范围七梯度最坏误差为 `2.03e-6`。

这不是通过放宽阈值掩盖问题；它明确限定 golden test 的良态输入范围。接近零 decay 下的
checkpoint 逆向重建病态性是独立议题。如果产品模型实际产生大量低 decay，应另开实验比较：

- 缩短 checkpoint chunk；
- chunk 内 forward recompute；
- 增加更密集 state checkpoint；
- 性能、显存与梯度误差的三方权衡。

这些方案均可能显著改变速度或内存，不属于本次 barrier PR 的最小修复范围。

## 11. 有限差分仲裁

按照 `ε=1e-3`，从七个输入张量各随机选两个坐标，共 14 个坐标，计算：

```text
(loss(x+ε) - loss(x-ε)) / (2ε)
```

结果中，Metal 与 CPU-einsum 解析梯度在每个坐标上彼此一致，差异约为 `1e-6`；两者与
float32 中心差分的差异约 `5.6e-4–2.6e-2`。这是因为 loss 为大量元素求和，ε=1e-3 时
两个 fp32 标量相减本身存在明显量化和消减误差。

有限差分无法在该 ε 与 fp32 标量口径下分辨 Metal 和 CPU reference，因为二者的差距远小于
差分噪声；但所有坐标上两条解析梯度始终同值、同方向。结合前述 `dh_in` fp64 闭式坐标复核，
仲裁结果支持 Metal 与 CPU reference，而不是 GPU einsum 的约 `1e-3` 偏差。

## 12. 负对照

为了验证确定性测试确实能捕获原竞态，而不是仅在修复后“自然通过”：

1. 临时移除 checkpoint backward 的末尾 barrier；
2. 保持 T=64、非零 h_in、随机 out/h_out cotangent 与 compiled graph 不变；
3. 对同一输入最多重复 50 次；
4. 第 39 次检测到以下梯度 bitwise 不一致：

```text
dw, dk, dv, da, db, dh_in
```

`dr` 没有进入该次不一致列表，符合其计算主要依赖 forward state 与 `d_out`、不依赖后续
`C_row` 递推的结构。负对照完成后 barrier 已恢复；临时 mutation 不进入提交。

## 13. 上游提交与测试

上游 fork 分支：

```text
main
```

提交：

```text
71bca50 Fix WKV-7 backward threadgroup race
0ecdb91 Add stateful WKV-7 backward golden test
```

上游测试：

```text
2 passed
```

第一笔提交的生产代码只修复两个 kernel 的 barrier，并配套加入确定性回归测试；
第二笔提交加入 stateful API、CPU einsum golden reference 与七梯度正确性测试，同时将
确定性测试整合进同一验收文件。现有零 state API 保持兼容。

## 14. 最终结论

### 14.1 已确认

1. WKV7 Metal backward 存在真实 shared-memory 数据竞争；
2. 竞态来自上游初始实现，Preen 移植时继承；
3. `fe0c493` 没有创建竞态，而是通过 compile/bf16 时序变化放大；
4. barrier 修复后 eager、compiled、bf16 直读和 fp32 物化的梯度语义一致；
5. 七梯度相对 reference 最坏误差 `2.03e-6`，`dh_in=1.73e-7`；
6. 移除 barrier 的负对照能重新触发不确定梯度；
7. compile 与 bf16 直读加速无需回滚；
8. 由受影响版本生成的 State 不能原地修复，需要重新训练。

### 14.2 未宣称

1. 严格无重复惩罚生成判据尚未完全回到历史 State；
2. 单次训练耗时不能证明 barrier 没有任何性能成本；
3. golden test 不覆盖接近零 decay 下的逆向重建病态性；
4. MLX GPU einsum 不能作为 `1e-5` 阈值下未经仲裁的 reference；
5. 1.5B 模型的角色字样泄漏、事实错误和偶发复读不等同于 State 梯度损坏。

## 15. 复现命令

### 15.1 Preen 快速测试

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

### 15.2 上游 golden test

```bash
cd rwkv-metal
PYTHONPATH=. ../.venv/bin/python -m pytest -q
```

### 15.3 两轮训练（历史对照 warmup=20）

```bash
PYTHONPATH=src .venv/bin/python -m statetuner.cli train \
  --model models/converted/rwkv7-g1h-1.5b \
  --data train_data/NekoQA_10k/nekoqa_smoke_200.json \
  --template qa \
  --out output/regression_barrier_fix/state.npz \
  --events-file output/regression_barrier_fix/events.jsonl \
  --lr 0.0001 --lr-floor 0.00001 --warmup 20 \
  --epochs 2 --ctx-len 512 --seed 42 \
  --fast-wkv --fast-wkv-chunk 16
```

### 15.4 当前默认参数两轮训练

```bash
PYTHONPATH=src .venv/bin/python -m statetuner.cli train \
  --model models/converted/rwkv7-g1h-1.5b \
  --data train_data/NekoQA_10k/nekoqa_smoke_200.json \
  --template qa \
  --out output/regression_barrier_fix_defaults/state.npz \
  --events-file output/regression_barrier_fix_defaults/events.jsonl \
  --epochs 2 --ctx-len 512 --seed 42 \
  --fast-wkv --fast-wkv-chunk 16
```

## 16. 可追溯产物

本地实验产物：

```text
output/NekoQA30k/
output/regression_fe0c493/
output/regression_fe0_compile_fp32/
output/regression_fe0_eager_bf16/
output/regression_barrier_fix/
output/regression_barrier_fix_defaults/
```

这些目录用于本地复核，不纳入 Git。正式可追溯结论由本报告、Preen 提交 `81fdf29` 与上游
提交 `71bca50`、`0ecdb91` 共同承载。
