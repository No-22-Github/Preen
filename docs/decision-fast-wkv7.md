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

- **6.67× 训练加速**(1678s → 252s,2 epoch,真实 320 token 序列);
- **内存显著低于 ops**(fast 6.51GB vs ops ~12GB);
- **精度数值等价**:loss 末态与 ops 基线差 0.19%,state std 差 0.44%。

### 序列长度与加速比的关系(关键)

ops 路径每 token 一次 Python dispatch,开销随序列长度线性增长;kernel 一次
dispatch 处理整段,优势随序列长度放大。**序列越长,fast path 价值越大:**

| 真实 token 长度 | ops 单步 | ops 总耗时(400步) | fast 总耗时 | 加速比 | ops 是否可行 |
|---------------|---------|-------------------|------------|--------|-------------|
| ~114(smoke_200) | 0.88s | 5.9 分钟 | 1.8 分钟 | 3.3× | 可行 |
| 320(nekoqa_320) | 4.1s | **28 分钟** | 4.2 分钟 | **6.67×** | 勉强(超时风险) |
| 512(nekoqa_512) | ~6.5s(估) | ~43 分钟 | 6.3 分钟 | ~7×(估) | 不可行 |

**fast path 不仅提速,更让长序列训练从"不现实"变成"可行"。** 产品级 state tuning
实际用的就是 320-512 token 的 NekoQA 回答,长序列加速比才是代表性数据。
(注:等价性完整验证覆盖至 320 token;512 档因 ops 基线不可行,仅有 fast 单跑的
收敛健康度证据,见第三轮。)

---

## 三轮实验概览

本实验经历三轮,每轮修正前轮盲点:

| 轮次 | 数据集(真实 token) | lr | 验证内容 | 结果 |
|------|---------------------|-----|---------|------|
| 第一轮 | smoke_200(~114 tok) | 0.01(**误用**) | 高 lr + chunk=32 | 2/4 否决(分叉 166%) |
| 第二轮 | smoke_200(~114 tok) | 1e-4(正确) | chunk×lr 2×2 矩阵 | chunk=16/8 通过 |
| **第三轮** | nekoqa_320/512(**真实长序列**) | 1e-4 | 长序列数值等价 + 加速比 | **通过,6.67×** |

**第一、二轮的数据集 smoke_200 编码后平均仅 114 token**(中位 104,最长 274),
远未达 ctx_len=512,代表性不足。第三轮用 NekoQA 30k 按实际编码 token 数筛选,
造出真实 320/512 token 数据集修正。三轮数据均保留记录。

---

## 背景:为什么有这个实验

Preen 的训练路径(`core.patch_rwkv7_for_train`)用 Python `_wkv7_step_ops` 循环
实现 WKV7 递归——每个 token 一次 GPU dispatch,ctx=512 就是 512 次。这是刻意的:
原生 `mlx-lm` 的 `wkv7_kernel` 是 Metal 黑盒无 VJP,梯度静默断裂,而 state tuning
唯一可训参数 S₀ 的梯度必须穿透整个递归链回到初始 state。ops 循环每步有 VJP,
保证梯度通畅,代价是慢。

BlinkDL 在本仓库 issue #1 中推荐了 RafaelUI(Alexei Goncharov / ImpulseLeap)的
三个仓库,其中 `metal-wkv7`
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
- 0.4B 模型未覆盖(本实验仅 1.5B;层数同为 24,预期同质但未验证)。

---

## 第一轮:smoke_200(~114 tok)+ lr=0.01 + chunk=32 —— 否决(2/4)

> **数据集**:`NekoQA_10k/nekoqa_smoke_200.json`,200 条,编码后平均 114 token
> (中位 104,min 64 / max 274)。**ctx_len 设 512 但实际序列远短于此。**
> **lr=0.01 是误用**——实际默认已改为 1e-4。此轮数据不构成有效裁决,但根因诊断有价值。

### 数据(1.5B g1g,3 epoch,seed=42)

```
              耗时    ep0     ep1     ep2      std    mem
  ops         470s  2.489   1.772   1.111  0.1453  12.32G
  fast c32    138s  4.884   3.717   2.960  0.1157   9.34G
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
分裂。fast 持续高位但缓慢收敛(不爆炸),说明是收敛到不同的解。

---

## 第二轮:smoke_200(~114 tok)+ lr=1e-4 + chunk=32/16/8 —— chunk=16/8 通过(4/4)

用户指出 lr 默认值已改为 1e-4,并建议试小 chunk。本轮用正确参数重测。

### 数据(1.5B g1g,3 epoch,seed=42,lr=1e-4)

```
              耗时    ep0     ep1     ep2      std    mem
  ops         528s  2.463   2.272   2.189  0.00429 12.28G
  fast c32    159s  3.358   2.400   2.370  0.00325  9.34G
  fast c16    161s  2.455   2.271   2.185  0.00431  9.46G
  fast c8     169s  2.462   2.273   2.182  0.00438  9.61G
```

### 裁决

| 配置 | 速度 | loss 差 | std 差 | 内存 | 结果 |
|------|------|---------|--------|------|------|
| fast c32 | ✓ 3.31× | ✗ 8.27% | ✗ 24.1% | ✓ -2.94GB | 2/4 不过 |
| **fast c16** | ✓ 3.28× | ✓ **0.15%** | ✓ **0.6%** | ✓ -2.82GB | **4/4 通过** |
| **fast c8** | ✓ 3.13× | ✓ **0.32%** | ✓ **2.1%** | ✓ -2.67GB | **4/4 通过** |

chunk=16 三 epoch loss 与 ops 差 <0.01,数值等价。

---

## 两轮反转的根因:chunk × lr 完整 2×2 矩阵

为分离 chunk 和 lr 两个变量的贡献,补测了**高 lr(0.01)+ chunk=16**:

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

**chunk 是决定性变量,与 lr 无关:**
- chunk=32:高 lr 差 166%,低 lr 差 8.3%——两档都不通过;
- chunk=16:高 lr 差 1.3%,低 lr 差 0.2%——两档都通过。

高 lr 确实放大误差(同 chunk 下高 lr 差值总更大),但 chunk ≤ 16 时高 lr 仍通过。
**第一轮失败的根因是 chunk=32,不是 lr。**

### 为什么隔离测试的预测错了

隔离测试(σ=0.5 随机输入)显示 chunk 越小梯度误差越大(chunk32 dk 6.6% / c16 10.1% /
c8 11.5%),但真实训练相反(chunk=16/8 更准)。**隔离测试的随机输入不代表真实训练
激活分布**——这是本轮最重要的方法论教训。

---

## 第三轮:真实长序列(320/512 tok)—— 通过,价值更大

> **修正前两轮盲点**:smoke_200 平均 114 token,代表性不足。本轮用 NekoQA 30k
> 按实际编码 token 数筛选(`test/make_long_datasets.py`),造真实长度数据集。

### 数据集

| 数据集 | 来源 | 条数 | 编码后 token(均/min/中位/max) |
|--------|------|------|-------------------------------|
| smoke_200 | NekoQA_10k | 200 | 114 / 64 / 104 / 274 |
| nekoqa_320_200 | NekoQA_30k | 200 | 319 / 318 / 320 / 321 |
| nekoqa_512_200 | NekoQA_30k | 200 | 511 / 510 / 512 / 513 |

### 320 token 档:fast vs ops 完整对照(4/4 通过)

配置:1.5B g1g,lr=1e-4,2 epoch(400 步),seed=42。ops 基线由用户手动跑(28 分钟)。

```
            耗时     ep0 loss   ep1 loss   ep0 std    ep1 std
  ops      1678s     1.7228     1.5049    0.00454    0.00478
  fast c16   252s     1.7293     1.5078    0.00451    0.00476
```

| 判据 | 结果 | 裁决 |
|------|------|------|
| 速度 | **6.67×**(1678s→252s) | ✓ |
| loss ep1 差 | **0.19%**(1.5049 vs 1.5078) | ✓ |
| std ep1 差 | **0.44%**(0.00478 vs 0.00476) | ✓ |
| 内存 | fast 6.51GB ≪ ops ~12GB | ✓ |

**长序列下 fast 与 ops loss 差 0.19%——比短序列的 0.15% 还接近,数值等价成立。**

### 512 token 档:fast 单跑(ops 基线不可行)

ops 基线在 320 token 已需 28 分钟(超 10 分钟硬超时),512 token 更甚。
512 档只跑 fast,看绝对收敛健康度。

```
            耗时     ep0 loss   ep1 loss   ep0 std    ep1 std    peak mem
  fast c16   379s     2.130      1.918     0.00418    0.00451    8.59GB
```

loss 降 10.0%,std 增 7.9%,单调收敛健康。peak 8.59GB 在 16GB 机器安全范围。
无 ops 对照(不可行),仅作收敛健康度参考。

---

## chunk 选择的工程权衡

```
chunk   (1/w)^chunk   反向重构次数   实测 loss差(短序列)  长序列验证
  32     ~30×          1×(基准)      8.27% ✗             未测(已知不行)
  16     ~5.3×         2×            0.15% ✓             ✓ 320tok 差0.19%
   8     ~2.3×         4×            0.32% ✓             未单独测
```

- chunk=16 与 chunk=8 短序列精度都等价(loss 差 <0.4%),chunk=16 速度略优。
- chunk=16 已在真实 320 token 长序列下验证通过。
- **推荐 chunk=16 为默认**,chunk=8 作为高敏感场景可选档。
- 反向重构次数翻倍未显著拖慢速度(16 vs 32 仅差 2s),WKV 在整步中占比小。

---

## pad 约束的处理

checkpoint kernel 要求 `T % chunk == 0`。逐样本训练管线下样本长度不固定
(NekoQA 编码后 T=319/511 等),不 pad 也不对齐。

修法(`core.patch_rwkv7_for_train_fast` 闭包内):pad 序列**末尾**到 chunk 倍数,
r/k/v/a/b 补 0、w 补 1.0。递归 `h = 1.0*h + 0 + 0 = h` 不变(state 原样穿过
pad 段),因果性保证 pad 段对真实 token 的 y 零影响。算完 slice 回真实 L。

实测 pad+slice 的全局相对误差 8.0e-4,与不 pad 时一致。返回的 h_out 含 pad 段
递归但训练下游不用(S₀ 每步重新注入),故 w pad 值无副作用。

不动 `data.py`/`train.py`,pad 逻辑完全内聚在 fast patch 闭包。

---

## 性能归因

### 为什么快
ops 路径每个 token 一次 GPU dispatch(`_wkv7_step_ops` 是 4 个 MLX op 的 Python
循环),ctx=512 → 512 次 dispatch + Python 循环开销。Metal kernel 整段一次
dispatch,GPU 侧完成全部递归,消除 Python↔GPU 往返。**序列越长,消除的 dispatch
越多,加速比越大**(114 tok 3.3× → 320 tok 6.67×)。

### 为什么内存反而降
ops 循环在 MLX 的 lazy 计算图里堆叠数百步的中间张量(每步 `state*w`、`v⊗k`、
`sab`、`state@r`),autograd 图持有全部中间值。Metal kernel 的中间状态只存在
GPU 寄存器,不进 MLX 计算图,只有 `h_checkpoints`(O(N_CHUNKS × D²))常驻。

### 为什么不是隔离 bench 的 13×
隔离 bench(kernel-only,无 matmul)测得 13×。端到端被 matmul(r/k/v/o_proj、FFN、
head)摊薄。长序列(6.67×)比短序列(3.3×)更接近 kernel 上限,因为 WKV 占比随
序列增长而上升。

---

## 推理路径:不接入(已在用更优 kernel)

推理走 `load_model(patch=False)`——即 **mlx-lm 自带的 `wkv7_kernel`**,纯 Metal
forward 无 VJP。实测它**比上游推理 kernel 快 10%**:

| kernel | T=32 耗时 | 技术 |
|--------|----------|------|
| **mlx-lm(我们现有)** | **0.302 ms** | `simd_sum` 硬件归约 + bf16 |
| 上游 `wkv7_metal.py` | 0.331 ms | 标量循环 + fp32 |

上游推理 kernel 是更老的实现(无 simd 归约 + fp32),换了反而倒退。推理优化应攻
matmul/lm_head(占 decode 82%),即量化领域,与本文 kernel 无关。

**fast path 仅用于训练,推理保持现状。**

---

## 正确性验证

`tests/test_fast_wkv7.py`(5 测,σ=0.3,0.83s 全过):
- forward 全局相对误差 ~8e-4(fp32 累加顺序差异,非 bug);
- 6 梯度 + S₀ 透传梯度相对误差 < 2e-3;
- 零 state + 短序列 T=128 边界覆盖。

**局限性标注**:这些测试用 σ=0.3,在真实训练分布下梯度误差更大。隔离正确性 ≠
训练数值等价,真实训练对照(本报告三轮实验)才是最终裁决。现有 253 测全过,无回归。

---

## 处置

### 采纳(chunk=16 为默认)
代码已在 `exp/wkv7-metal` 就绪。并入 main 时:
1. `--fast-wkv-chunk` 默认值从 32 改为 16;
2. 补 CHANGELOG `[未发布]` 段(按 AGENTS.md 纪律展开写);
3. 考虑是否将 `--fast-wkv` 默认设为 True(待裁决——默认开启改变训练口径,
   历史 events.jsonl 数据不再可比)。

### 后续可选(非阻塞)
- 0.4B 模型验证(层数同 24 但 hidden 小,预期同质);
- chunk=8 在长序列(512 token)下与 chunk=16 的自洽对照——512 档无 ops 基线,
  两种 chunk 互相吻合是该深度下最便宜的等价性旁证(fast 单跑约 6-7 分钟)。

---

## 产物清单

| 产物 | 位置 | 进库 |
|------|------|------|
| kernel 模块 | `src/statetuner/fast_wkv7.py` | ✅ 分支 |
| patch 接入 | `src/statetuner/core.py`(`patch_rwkv7_for_train_fast`) | ✅ 分支 |
| 透传链路 | `service.py`/`cli.py`(`--fast-wkv`/`--fast-wkv-chunk`) | ✅ 分支 |
| 正确性测试 | `tests/test_fast_wkv7.py` | ✅ 分支 |
| 本报告 | `docs/decision-fast-wkv7.md` | ✅ 本文件 |
| 长序列数据集 | `train_data/NekoQA_30k/nekoqa_{320,512}_200.json` | 本地 |
| 数据集生成脚本 | `test/make_long_datasets.py` | ❌ gitignore(test/) |
| 上游 clone | `test/metal-wkv7/`、`test/rwkv-metal/`、`test/rwkv-mlx/` | ❌ gitignore |
| bench 脚本 | `test/bench_wkv7.py`、`test/bench_fast_vs_ops.py` | ❌ gitignore |
| 训练产物 | events/state npz(各轮) | ❌ 已清理,数据记入本报告 |