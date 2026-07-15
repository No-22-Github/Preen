# 决策记录:WKV7 Metal checkpoint kernel 接入(fast path)

> **生成时间**:2026-07-15
> **关联分支**:`exp/wkv7-metal`
> **关联实验**:`test/metal-wkv7/`(上游仓库 clone)、`test/bench_wkv7.py` 等 bench 脚本
> **引用规矩**:所有结论来自实测数据,不引二手。判据跑前定死,跑完未改(AGENTS.md §判据纪律)。

---

## 结论

**采纳 `--fast-wkv --fast-wkv-chunk 16` 为训练 fast path,准备并入 main。**

将训练路径的 WKV7 递归从 Python `_wkv7_step_ops` 循环(L 次 GPU dispatch)换成
整段 Metal checkpoint kernel(forward + backward 各一次 dispatch),在 1.5B 模型
state tuning 上实测:

- **3.28× 训练加速**(528s → 161s,3 epoch);
- **内存反降 2.82GB**(12.28G → 9.46G);
- **精度数值等价**:loss 末态与 ops 基线差 0.15%,state std 差 0.6%。

结论的关键支撑是 **chunk × lr 完整 2×2 矩阵**:chunk=16 在高 lr(0.01)和
低 lr(1e-4)下都与 ops 数值等价(差 1.3% / 0.2%),证明 **chunk 是决定性变量**,
不是靠低 lr 才成立。详见下文。

---

## 背景:为什么有这个实验

Preen 的训练路径(`core.patch_rwkv7_for_train`)用 Python `_wkv7_step_ops` 循环
实现 WKV7 递归——每个 token 一次 GPU dispatch,ctx=512 就是 512 次。这是刻意的:
原生 `mlx-lm` 的 `wkv7_kernel` 是 Metal 黑盒无 VJP,梯度静默断裂,而 state tuning
唯一可训参数 S₀ 的梯度必须穿透整个递归链回到初始 state。ops 循环每步有 VJP,
保证梯度通畅,代价是慢。

朋友推荐了 RafaelUI(Alexei Goncharov / ImpulseLeap)的三个仓库,其中 `metal-wkv7`
实现了带 checkpoint 反向的 Metal WKV7 kernel(`mx.custom_function` 注册 VJP),
宣称 7.8× 加速。本实验验证它能否在不破坏 state tuning 梯度正确性的前提下提速。

---

## 上游仓库与许可证

| 仓库 | 角色 | LICENSE |
|------|------|---------|
| `rwkv-metal` | 正式发布包(v0.1.0,完整框架) | ✅ Apache-2.0 完整文本 |
| `rwkv-mlx` | 生产训练仓库(预训练+LoRA) | ⚠️ 仅 README 一行声明,无 LICENSE 文件 |
| `metal-wkv7` | R&D 隔离仓库(纯 kernel) | ⚠️ **无任何许可证声明** |

三者同作者。kernel 实现从 `rwkv-metal`(唯一带完整 Apache-2.0 文本)移植,兼容
本项目许可证。`metal-wkv7` 裸抄有法律灰色地带(无许可证 = 保留所有权利)。

kernel 核心设计(checkpoint 反向、`mx.custom_function` VJP、bf16 dtype cast)原样
保留;**唯一改动**:`make_wkv7_checkpoint` 工厂暴露 `h_in` 参数,支持透传可训练 S₀
(原实现固化成零)。VJP 本就支持对 `h_in` 求梯度(backward kernel 已算 `dh_in_out`),
改动几乎免费。

---

## 判据(跑前定死,跑完未改)

| # | 判据 | 通过线 | 依据 |
|---|------|--------|------|
| 1 | 端到端速度 | fast ≥ ops 的 1.5× | 隔离 kernel bench 13×,端到端被 matmul 摊薄;1.5× 是"值得改"最低门槛 |
| 2 | loss 末态不退化 | 同 seed 同数据,fast vs ops 差 < 5% | fwd/bwd 数值等价则应≈相等,5% 留余量 |
| 3 | state std 健康 | fast 训完 std 与 ops 偏差 < 20% | 防 kernel 改变梯度尺度导致 S₀ 长歪 |
| 4 | 内存峰值不恶化 | fast RSS peak ≤ ops RSS peak + 0.5GB | kernel 多存 h_checkpoints,应有微小增量 |

**盲点预警(判据不覆盖,如实记录):**
- 训练数值稳定性跨 epoch 表现(判据只看末态,不看中间抖动)。
- kernel JIT 首次编译耗时(判据测稳态,不含冷启动)。
- 更长 ctx(1024/2048)、0.4B 模型未覆盖。

---

## 第一轮:lr=0.01 + chunk=32 —— 否决(2/4)

> **这是踩坑轮。** 我误用了老的高 lr 参数(0.01),实际默认已改为 1e-4。
> 此轮数据**不构成有效裁决**,但根因诊断有价值,保留记录。

### 配置

模型 rwkv7-g1g-1.5B,NekoQA 200条×3epoch,**lr=0.01**(误用),ctx=512,seed=42。

### 数据

```
              耗时    加速    ep0     ep1     ep2      std    mem
  ops         470s  1.00×   2.489   1.772   1.111  0.1453  12.32G
  fast c32    138s  3.40×   4.884   3.717   2.960  0.1157   9.34G
```

| 判据 | 结果 | 裁决 |
|------|------|------|
| 速度 3.40× | ✓ | 通过 |
| loss 差 166% | ✗ | 不过 |
| std 差 20.4% | ✗ | 临界不过 |
| 内存 -2.98GB | ✓ | 通过 |

**2/4。loss 从 epoch 0 就严重偏离**(4.88 vs 2.49)。

### 逐 step 发散诊断

```
step | ops_loss | fast_loss | diff
  0  | 2.4674   | 2.4641    | -0.003  ← 完全等价
  1  | 2.0373   | 2.0289    | -0.008  ← 完全等价
  2  | 7.4381   | 10.5120   | +3.07   ← 第一次梯度更新后突变分裂
  3+ | ~2.9     | ~10.8→收敛 | 持续偏高
```

step 0-1 两路 loss 完全一致(kernel forward 数值正确),step 2(第一次梯度更新后)
分裂。fast 持续高位但缓慢收敛(不爆炸),说明不是发散,是**收敛到不同的解**。

### 根因初判(后被部分修正)

隔离测试(单层 WKV7,真实维度 H=32 D=64)测梯度相对误差:

| 激活量级 σ | forward 相对误差 | 最大梯度相对误差 |
|-----------|-----------------|-----------------|
| 0.3(初始测试用) | 8e-4 | 1.3e-3 |
| 0.5(真实模型激活) | 4e-3 | **9.2%(dk)** |

σ=0.3 掩盖了问题,σ=0.5 下单层梯度误差 9.2%,经 24 层反向 + 512 步递归累积放大。
初步归因:checkpoint 反向重构 `h_prev = (h_cur - v*k - sa*b) / w`,(1/w)^CHUNK 放大。

**但这个归因在第二轮被修正——见下文。**

---

## 第二轮:lr=1e-4 + chunk=32/16/8 —— chunk=16/8 反转通过(4/4)

用户指出 lr 默认值已改为 1e-4,并建议试小 chunk。本轮用正确参数重测。

### 配置

同上模型/数据,**lr=1e-4(新默认,不传 --lr)**,chunk 三档对照。

### 数据

```
              耗时    加速    ep0     ep1     ep2      std    mem
  ops         528s  1.00×   2.463   2.272   2.189  0.00429 12.28G
  fast c32    159s  3.31×   3.358   2.400   2.370  0.00325  9.34G
  fast c16    161s  3.28×   2.455   2.271   2.185  0.00431  9.46G
  fast c8     169s  3.13×   2.462   2.273   2.182  0.00438  9.61G
```

### 裁决

| 配置 | 速度 | loss 差 | std 差 | 内存 | 结果 |
|------|------|---------|--------|------|------|
| fast c32 | ✓ 3.31× | ✗ 8.27% | ✗ 24.1% | ✓ -2.94GB | 2/4 不过 |
| **fast c16** | ✓ 3.28× | ✓ **0.15%** | ✓ **0.6%** | ✓ -2.82GB | **4/4 通过** |
| **fast c8** | ✓ 3.13× | ✓ **0.32%** | ✓ **2.1%** | ✓ -2.67GB | **4/4 通过** |

**chunk=16 三 epoch loss 与 ops 差 <0.01**(2.455 vs 2.463、2.271 vs 2.272、
2.185 vs 2.189),std 差 0.6%——数值等价。

---

## 两轮反转的根因(核心结论)

### 完整 2×2 矩阵:chunk × lr

为分离 chunk 和 lr 两个变量的贡献,补测了**高 lr(0.01)+ chunk=16**(此前缺这格)。
完整矩阵:

```
           |            chunk=32                |            chunk=16
           |   ep0/ep1/ep2    vs ops            |   ep0/ep1/ep2    vs ops
-----------+-----------------------------------+-----------------------------------
 lr=0.01   | 4.88/3.72/2.96   166% ✗           | 2.48/1.76/1.10    1.3% ✓
   ops     | 2.49/1.77/1.11  (基线)            |
-----------+-----------------------------------+-----------------------------------
 lr=1e-4   | 3.36/2.40/2.37    8.3% ✗           | 2.46/2.27/2.19    0.2% ✓
   ops     | 2.46/2.27/2.19  (基线)            |
```

### 结论:chunk 是决定性变量,与 lr 无关

- **chunk=32**:高 lr 差 166%,低 lr 差 8.3%——**两档都不通过**。
- **chunk=16**:高 lr 差 1.3%,低 lr 差 0.2%——**两档都通过**。

chunk=16 在高 lr(0.01)和低 lr(1e-4)下都与 ops 数值等价。**第一轮的失败
根因是 chunk=32,不是 lr。** 之前"两因素叠加"的判断被这组数据修正:
高 lr 确实会放大误差(同一 chunk 下,高 lr 的差值总比低 lr 大),但只要 chunk
够小(≤16),高 lr 下仍能通过(1.3% < 5% 门槛)。

### 为什么隔离测试的预测错了

第一轮据隔离测试(σ=0.5 随机输入)预测"减小 chunk 能降梯度误差":

```
chunk=32: dk 6.6%   chunk=16: dk 10.1%   chunk=8: dk 11.5%  ← chunk 越小误差越大
```

但真实训练结果**相反**:chunk=16/8 比 chunk=32 准得多。

**隔离测试的随机输入不代表真实训练激活分布。** 真实模型经 layernorm/projection
后激活分布不同,checkpoint 反向的误差行为也不同。**隔离数值测试不能替代真实
训练对照**——这是本轮最重要的方法论教训,已更新到 `tests/test_fast_wkv7.py`
的注释(标注 σ=0.3 的局限性)。

### 修正第一轮的根因误判

第一轮把分叉归因于 `(1/w)^CHUNK` 放大,据此预测"减小 chunk 能降梯度误差"。
但隔离测试(σ=0.5 随机输入)显示相反结果:

```
chunk=32: dk 6.6%   chunk=16: dk 10.1%   chunk=8: dk 11.5%  ← chunk 越小误差越大
```

而真实训练结果**相反**:chunk=16/8 比 chunk=32 准得多。

**结论:隔离测试的随机输入不代表真实训练激活分布。** 真实模型经 layernorm/
projection 后激活分布不同,checkpoint 反向的误差行为也不同。**隔离数值测试
不能替代真实训练对照**——这是本轮最重要的方法论教训,已更新到
`tests/test_fast_wkv7.py` 的注释(标注 σ=0.3 的局限性)。

---

## chunk 选择的工程权衡

```
chunk   (1/w)^chunk   反向重构次数   实测 loss差   实测速度
  32     ~30×          1×(基准)      8.27% ✗       3.31×
  16     ~5.3×         2×            0.15% ✓       3.28×
   8     ~2.3×         4×            0.32% ✓       3.13×
```

- chunk=16 与 chunk=8 精度都等价(loss 差 <0.4%),chunk=16 速度略优(少一次重构)。
- chunk=8 更保守(误差放大更小),对极端 lr 或更长 ctx 有更多余量,但速度损失 ~5%。
- **推荐 chunk=16 为默认**,chunk=8 作为高敏感场景的可选档。

注:反向重构次数翻倍未显著拖慢速度(16 vs 32 仅差 2s),因为 WKV 在整步中
占比小(matmul 是大头),重构开销被摊薄。

---

## pad 约束的处理

checkpoint kernel 要求 `T % chunk == 0`。逐样本训练管线下样本长度不固定
(NekoQA 编码后 T=124 等),不 pad 也不对齐。

修法(`core.patch_rwkv7_for_train_fast` 闭包内):pad 序列**末尾**到 chunk 倍数,
r/k/v/a/b 补 0、w 补 1.0。递归 `h = 1.0*h + 0 + 0 = h` 不变(state 原样穿过
pad 段),因果性保证 pad 段对真实 token 的 y 零影响。算完 slice 回真实 L。

实测 pad+slice 的全局相对误差 8.0e-4,与不 pad 时一致。返回的 h_out 含 pad 段
递归但训练下游不用(S₀ 每步重新注入),故 w pad 值无副作用。

不动 `data.py`/`train.py`,pad 逻辑完全内聚在 fast patch 闭包。

---

## 性能归因

### 为什么快(3.3×)
ops 路径每个 token 一次 GPU dispatch(`_wkv7_step_ops` 是 4 个 MLX op 的 Python
循环),ctx=512 → 512 次 dispatch + Python 循环开销。Metal kernel 整段一次
dispatch,GPU 侧完成全部递归,消除 Python↔GPU 往返。

### 为什么内存反而降(2.8GB)
ops 循环在 MLX 的 lazy 计算图里堆叠 512 步的中间张量(每步 `state*w`、`v⊗k`、
`sab`、`state@r`),autograd 图持有全部中间值。Metal kernel 的中间状态只存在
GPU 寄存器,不进 MLX 计算图,只有 `h_checkpoints`(O(N_CHUNKS × D²))常驻。

### 为什么不是隔离 bench 的 13×
隔离 bench(kernel-only,无 matmul)测得 13×。端到端被 matmul(r/k/v/o_proj、FFN、
head)摊薄——这些本来就是 Metal 上的高效 op,WKV 提速只作用于递归段。3.3× 是
整步 fwd+bwd 的真实加速。

---

## 正确性验证

`tests/test_fast_wkv7.py`(5 测,σ=0.3,0.83s 全过):
- forward 全局相对误差 ~8e-4(fp32 累加顺序差异,非 bug);
- 6 梯度 + S₀ 透传梯度相对误差 < 2e-3;
- 零 state + 短序列 T=128 边界覆盖。

**局限性标注**:这些测试用 σ=0.3,在真实训练分布下梯度误差更大(见上文)。
隔离正确性 ≠ 训练数值等价,真实训练对照(本报告)才是最终裁决。

现有 253 测全过,无回归。

---

## 处置

### 采纳(chunk=16 为默认)
代码已在 `exp/wkv7-metal` 就绪。并入 main 时:
1. `--fast-wkv-chunk` 默认值从 32 改为 16;
2. 补 CHANGELOG `[未发布]` 段(按 AGENTS.md 纪律,展开写 kernel 来源、加速比、
   内存收益、chunk 选择理由);
3. 考虑是否将 `--fast-wkv` 默认设为 True(待用户裁决——默认开启改变训练口径,
   历史 events.jsonl 数据不再可比)。

### 后续可选(非阻塞)
- 更长 ctx(1024/2048)下 chunk=16 表现验证;
- 0.4B 模型验证(层数同 24 但 hidden 小,预期同质);
- 推理路径不接入(推理已用 mlx-lm 自带 Metal kernel,无 VJP 需求)。

---

## 产物清单

| 产物 | 位置 | 进库 |
|------|------|------|
| kernel 模块 | `src/statetuner/fast_wkv7.py` | ✅ 分支 |
| patch 接入 | `src/statetuner/core.py`(`patch_rwkv7_for_train_fast`) | ✅ 分支 |
| 透传链路 | `service.py`(`fast_wkv`/`fast_wkv_chunk` 字段)、`cli.py`(`--fast-wkv`/`--fast-wkv-chunk`) | ✅ 分支 |
| 正确性测试 | `tests/test_fast_wkv7.py` | ✅ 分支 |
| 本报告 | `docs/decision-fast-wkv7.md` | ✅ 本文件 |
| 上游 clone | `test/metal-wkv7/`、`test/rwkv-metal/`、`test/rwkv-mlx/` | ❌ gitignore |
| bench 脚本 | `test/bench_wkv7.py`、`test/bench_fast_vs_ops.py`、`test/diff_wkv7.py` | ❌ gitignore |
| 训练产物 | events/state npz(各轮) | ❌ 已清理,数据记入本报告 |
