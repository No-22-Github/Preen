# 决策记录:训练精度选择(bf16 混合精度)

> **生成时间**:2026-07-11
> **关联实验**:`exp/mixed-precision` 分支(D 实验 + 多 seed 矩阵)
> **引用规矩**:本文件所有结论来自拉源码实读,不引二手教程。行号截至 2026-07 调研时点的各仓库最新代码。

---

## 结论

**RWKV 官方训练(含 state tuning)以 bf16 为惯例**,所有官方示例脚本与 README 均标注 bf16。但有两点必须同时说明,表述要准确:

1. **惯例是 bf16,代码默认值不是。** `--precision` 的 argparse default 是 `fp16`(历史遗留),bf16 通过所有官方示例 shell 脚本显式指定来体现。
2. **wkv kernel 内部的递归 state 累加是 fp32**(硬性设计要求,因衰减系数 w 接近 1,fp32 累加防溢出/丢精度)。这是全行业 RWKV 实现的共性,而我们的 D 方案恰恰偏离了这一层——前向全程 bf16,无 kernel 内 fp32 累加保护,详见下文「对本项目的含义」对比表。

---

## 出处(RWKV-PEFT,JL-er/RWKV-PEFT)

| 文件 | 行号 | 原文 | 说明 |
|---|---|---|---|
| `train.py` | 142 | `parser.add_argument("--precision", default="fp16", type=str)` | 代码默认 fp16(非 bf16) |
| `train.py` | 227-228 | `if args.precision == "fp16": rank_zero_info("Note: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.")` | 代码本身提示 fp16 溢出,建议 bf16 |
| `rwkvt/args_types.py` | 87 | `precision: str = "fp16"` | dataclass 默认同样 fp16 |
| `scripts/state tuning.sh` | 末行 | `--accelerator gpu --precision bf16 \` | **官方 state tuning 示例用 bf16** |
| `scripts/lora.sh` | 末行 | `--precision bf16` | LoRA 示例 bf16 |
| `scripts/run_sft.sh` | 末行 | `--precision bf16` | SFT 示例 bf16 |
| `scripts/miss.sh` | 末行 | `--precision bf16` | MiSS 示例 bf16 |
| `README.md` | 60 | `- Training precision: BF16` | README 明确标注 |
| `README_zh.md` | 95 | `- 训练精度:BF16` | 中文 README 同样 |

state tuning 入口:`--peft state --op fla`(`README_zh.md:12` 推荐用 `--op fla`,即 `rwkvfla` 的 `chunk_rwkv7`,dtype 跟随上层 = bf16)。

CUDA 路径(`--op cuda`)的 state 累加:`rwkvt/operator/rwkvop.py:312` 硬断言 `assert all(i.dtype==torch.bfloat16 for i in [r,w,k,v,a,b])`,内部 state 累加用 `torch.float32`(`:316`)。

---

## 出处(BlinkDL/RWKV-LM,RWKV-v7)

| 文件 | 行号 | 原文 | 说明 |
|---|---|---|---|
| `RWKV-v7/train_temp/train.py` | 171 | `assert args.precision in ["fp32", "tf32", "fp16", "bf16"]` | 四选一 |
| `RWKV-v7/train_temp/train.py` | 176-177 | fp16 溢出警告,建议 bf16/tf32 | 同 PEFT |
| `RWKV-v7/train_temp/demo-training-run.sh` | 末行 | `--precision bf16 --strategy deepspeed_stage_2` | **官方 demo 用 bf16** |
| `RWKV-v7/train_temp/demo-training-run-v7-pile.sh` | 末行 | `--precision bf16 --strategy deepspeed_stage_2` | Pile 大规模训练 bf16 |
| `README.md` | 47 | `RWKV-7 7.2B bf16 training on 4x8xH100` | 官方吞吐基准用 bf16 |
| **`README.md`** | **251** | **`Note: In [state = kv + w * state] everything must be in fp32 because w can be very close to 1. So we can keep state and w in fp32, and convert kv to fp32.`** | **关键设计原则:state 递归累加必须 fp32** |
| `RWKV-v7/train_temp/src/model.py` | 64 | `assert all(i.dtype==torch.bfloat16 for i in [r,w,k,v,a,b])` | 优化 kernel 硬断言 bf16 输入 |
| `RWKV-v7/train_temp/src/model.py` | 67 | `s = torch.empty(..., dtype=torch.float32, ...)` | 内部 state 用 fp32 |

生产训练路径用的是 `train_temp/cuda/` 下带 `_bf16` 后缀的 fused kernel(`rwkv7_tmix_mix6_bf16_v5` 等,见 `model.py:87,119,179,466`),**没有 fp16 的等价 kernel 被编译**。

---

## 出处(ChatRWKV pip 包,推理用,交叉参考)

`rwkv_pip_package/src/rwkv/model.py`:
- `:234-243`:strategy 字符串决定 dtype,fp16/fp32/bf16 三选一,无默认
- `:210-214`:kernel 提供 `forward_fp16` 和 `forward_bf16` 两版本
- `:288`:`state[i*3+1] = torch.zeros(..., dtype=torch.float32, ...)` —— RNN 状态强制 fp32

注:这是推理包,不在"训练精度惯例"范围,但印证了"state 用 fp32"的设计共性。

---

## 对本项目(Preen / StateTuner)的含义

### 我们的 D 方案 vs 官方惯例

| 维度 | 官方(CUDA/FLA kernel) | 我们的 D 方案(MLX) |
|---|---|---|
| 模型权重/激活 | bf16 | bf16(MLX 加载即 bf16) |
| wkv 内部 state 累加 | **fp32**(kernel 内部硬编码) | **bf16**(state 进 `_wkv7_step_ops` 前 cast) |
| S₀(可训练 state) | 跟随权重 bf16,但 kernel 累加 fp32 | **fp32 master**(梯度回传后 fp32 更新) |
| 数值保护 | kernel 内 fp32 累加(w≈1 时不丢精度) | 无 kernel 内 fp32 保护,bf16 全程 |

**关键差异(诚实记录)**:官方 kernel 内部的 state 累加是 fp32,我们的 D 方案没有这层保护——前向 state 全程 bf16。bf16 仅 7 位有效数字,递归数百步会累积误差。

#### 安全感的口径(重要:claim 建在质量实测上,不建在误差数字上)

**前向误差随序列长度线性增长**(实测 200 步 ~3%,真实 w 分布),无饱和;**训练质量在 ≤273 token 实测等价**(D 实验任务2:loss 差 -0.36%、十问行为等价),任务解空间的平坦性提供了误差容忍。**bf16 适用边界由行为 A/B 的实测覆盖长度决定,数值误差曲线作为预警指标。**

这个口径是防御性的:安全感挂在"训练出来的东西行为等价"上,不挂在"前向误差小"上。将来任何人拿误差数字来质疑(比如"3% 误差怎么可能没问题"),动摇不了 claim——因为 claim 从一开始就没建在误差上。误差曲线只回答"边界在哪、什么时候该重新验证",不回答"能不能用"。

> **⚠️ 误差曲线本身的修正记录**:初版用合成 w(0.95±0.02,温和)测出"30 步饱和在 1%"——这是假象,合成 w 太温和。真实模型 w 分布(21.9% 通道 >0.999)下误差**持续走阔不饱和**:200 步从 0.4% 涨到 3.4%。按 w 分桶拆开后,高 w 通道(>0.999)和低 w 通道(<0.99)走阔速率几乎一样(200步末 3.24% vs 2.93%)——w 接近 1 并不是误差走阔的主因。这条误差曲线作为预警指标的用途不变,但"饱和"结论作废。

> **验证边界更新钩子**:当前行为 A/B 实测覆盖 ≤273 token,误差曲线覆盖 200 步(真实 w)。分桶标定配套的**延长冒烟(真实 w,700 步)**完成后,误差预警曲线随之延长;行为 A/B 若扩展到长样本(未来工作),覆盖边界随之更新。run_full.sh 阶段 0b 是固定环节。

#### w 分布实测(1.5B G1x,真实样本,fp32 路径)

上面冒烟结论省略了一个原本写过的前提"w 不极端接近 1"——**实测推翻了这个假设**(但需注意口径,见下)。

> **⚠️ 镜像坑修正**:初版 dump 直接 hook `_wkv7` 拿 w,但模型权重是 bf16,w 经 `exp(-0.606531*sigmoid(w_lora(x)))` 计算后 `.astype(bf16)` cast 回 bf16——hook 到的是 **bf16 量化产物**,把 0.9997 舍成 1.0,制造了"全层 p95=1.0、w>0.999 占 41.6%"的虚高假象。修正后走 fp32 路径(ops patch + `set_dtype(fp32)`),真实分布如下。`dump_w_dist.py` 已固定走 fp32 路径。

在 1.5B 各层、真实 NekoQA 样本(273 token)上 **fp32 路径** dump 的 w(`w_lora` 输出,参数化 `exp(-0.606531*sigmoid(...))`)分布:

```
1.5B,24 层,273 token 样本,fp32 路径逐层 dump
layer0:  w_p95=0.999997  遗忘率(1-w) p5=2.98e-06  bf16盲区(>0.999)=24.9%
layer6:  w_p95=0.999384  遗忘率 p5=6.16e-04       bf16盲区=7.1%
layer12: w_p95=0.999969  遗忘率 p5=3.12e-05       bf16盲区=36.5%
layer23: w_p95=0.998089  遗忘率 p5=1.91e-03       bf16盲区=3.8%
全层:w_p95=0.999979,bf16盲区(>0.999)平均 21.9%
```

**G1x 模型的 w 确实大量接近 1**(全层 21.9% 通道 >0.999,浅层高达 36%),遗忘率最小的通道 `1-w ≈ 3e-6`(本质永久记忆)。这正是官方坚持 kernel 内 fp32 累加的原因。**但分桶误差分析(见上修正段)表明,w 接近 1 并不是 bf16 误差走阔的主因**——高 w 和低 w 通道走阔速率一致。完整 dump(0.4B/1.5B 各层)随标定单一并产出(`dump_w_dist.py`),"为什么我们没踩官方的坑"的真正答案待 700 步实测后定论。

### 决策含义

- **bf16 作为训练默认路线有官方惯例背书** ✅(所有官方示例 + README + 生产 kernel 均 bf16)
- **S₀ 保持 fp32 master 符合官方"state fp32"的设计精神** ✅(我们的 master 在 fp32,只是前向 cast)
- **前向全程 bf16(无 kernel 内 fp32 累加)偏离了官方 kernel 设计** ⚠️(MLX 无可定制 kernel,这是平台限制)。前向误差线性增长不饱和,但 claim 的安全性建在训练质量实测等价上(≤273 token 行为 A/B),不建在误差数字上——误差曲线作为预警指标,边界由行为实测覆盖长度决定。

**最终结论:bf16 默认化的方向与官方惯例一致,可推进。** 多 seed 矩阵(本验证单任务1)提供跨 seed/跨模型档位的重复验证证据,全绿后"切默认"另开单执行。
