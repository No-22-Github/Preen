# Preen — RWKV-7 State Tuning for Mac

> Mac 原生的 RWKV-7 state tuning 工具。拖入 jsonl 数据集 → 选模型 → 训练 → 导出可挂载的 state 文件。
>
> 冻结模型全部权重,只训练每层 64×64 的初始状态矩阵 S₀,使模型从该状态启动时
> 表现出目标行为(说话风格、任务模式、翻译等)。

**当前阶段:P1 已完成**(CLI 工具就绪,`.pth` 导出 + RWKV Runner 挂载验证通过)。

- P0(技术验证):梯度穿透、收敛、泛化、ops/kernel 等价——[实验报告](experiments/p0_translate/实验报告.md)
- P1(产品化):CLI(train/eval/export/preview)、`.pth` 导出器、训练循环产品化、回归测试

---

## 这是什么

RWKV-7 是线性注意力/RNN 架构,每层维护一个矩阵值状态 S,随序列演化。
**State tuning** 把 S 的初始值 S₀ 从零变成可训练参数,用梯度下降找一个
"虚拟前缀"等价的初始状态,使模型从此启动时符合目标行为(说话风格、任务模式)。

P0 验证了五件事(详见 [实验报告](experiments/p0_translate/实验报告.md)):

| 命题 | 结果 |
|---|---|
| 梯度能穿透递归抵达每层 S₀ | ✅ |
| 优化器能把 10 条样本 loss 压到接近零 | ✅ |
| 100 条训练后对未见中文表现翻译行为 | ✅ |
| MLX 两条前向路径(ops/kernel)容差内等价 | ✅ |
| tokenizer 与 llama.cpp 一致 | ✅ |

---

## 架构与依赖

```
SwiftUI 壳 (Phase 3, 未实现)
        ↕ IPC
Python Sidecar (mlx-lm 训练/推理)  ← 本仓库当前实现
```

**核心引擎**:[ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) 的 `rwkv7.py`
(Apple 维护)。wkv7 前向有两条等价路径:Metal kernel(推理,快)和纯 ops 循环(可微)。
本项目的工作是在其之上做训练改造,详见 [docs](docs/)。

**反向传播**:由 MLX 框架的自动微分(`mx.value_and_grad`)自动完成,
本项目未实现任何反向传播代码 —— 梯度能否穿透取决于前向用可微路径(ops),
见 [P0 理论指南 §二](docs/P0-理论指南.md)。

---

## 仓库结构

```
src/statetuner/                 P1 正式包 (CLI 工具)
├── core.py                       patch ops 路径 + 可训练 state + generate
├── data.py                       数据集 (jsonl → tokenize + loss mask)
├── events.py                     结构化训练事件 (为 sidecar IPC 铺路)
├── train.py                      训练循环 (lr/std 监控/早停/checkpoint/恢复)
├── export.py                     .pth 导出器 (RWKV Runner 可挂载) + round-trip 验证
└── cli.py                        CLI: train/eval/export/preview 四子命令

tests/                          回归测试
├── test_export.py                导出 round-trip (快, ~5s)
├── test_inference.py             推理 golden (快, ~17s)
├── test_train.py                 训练行为断言 (慢, --slow, ~4min)
├── golden/                       golden 快照
└── conftest.py

docs/                           理论文档 (必读)
├── RWKV-StateTuner-Roadmap.md    落地路线图
├── P0-理论指南.md                 state tuning 原理
└── 参考仓库实现.md                依赖与参考来源

tools/                          模型转换工具
├── convert_rwkv7_to_hf.py        RWKV 原生 .pth → fla HF
└── fla_cpu_bootstrap.py          macOS 无 triton 时短路 fla.ops

experiments/p0_translate/        P0 实验 (历史归档, 保留不动)
├── 实验报告.md                    完整实验记录
└── checkpoints_v3/ep04.npz       P0 验收通过的翻译 state (测试基准)

models/fla-hub-rwkv7-0.1B-g1/   World tokenizer (转换依赖, 提交进库)
train_data/translate/           训练数据 (中英翻译对)
```

---

## 快速开始

### 环境要求

- Apple Silicon Mac (M1+, 本项目在 M5 / 16GB 上验证)
- Python 3.11 (uv 自动管理)
- [uv](https://docs.astral.sh/uv/) 包管理器

### 1. 获取模型

本项目不包含大模型文件(被 gitignore)。需下载以下文件到 `models/`:

| 文件 | 用途 | 大小 | 来源 |
|---|---|---|---|
| `rwkv7-g1d-0.4b-20260210-ctx8192.pth` | 0.4B 原始权重(待转换) | 902M | [魔搭 Blink_DL/rwkv7-g1](https://modelscope.cn/models/Blink_DL/rwkv7-g1/files) |
| `fla-hub-rwkv7-0.1B-g1/model.safetensors` | 0.1B HF 权重(转换的 ground truth 校验) | 364M | [HuggingFace fla-hub/rwkv7-0.1B-g1](https://huggingface.co/fla-hub/rwkv7-0.1B-g1) |

> tokenizer 文件已在本库 `models/fla-hub-rwkv7-0.1B-g1/`(体积小,转换必需)。
> 0.1B 的 `model.safetensors` 需另外下载,仅用于转换时的键名校验。

### 2. 转换模型

```bash
python tools/convert_rwkv7_to_hf.py \
    --rwkv7 models/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
    --output models/converted/rwkv7-g1d-0.4b \
    --reference models/fla-hub-rwkv7-0.1B-g1/model.safetensors \
    --tokenizer-src models/fla-hub-rwkv7-0.1B-g1 \
    --precision bf16
```

### 3. 安装 statetuner CLI

```bash
uv sync                    # 安装依赖 (mlx-lm + torch + typer 等)
uv run statetuner --help   # 四个子命令: train / eval / export / preview
```

### 4. 训练 + 导出

```bash
# 训练 state tuning, 训完直接导出 RWKV Runner 可挂载的 .pth
uv run statetuner train \
    --model models/converted/rwkv7-g1d-0.4b \
    --data train_data/translate/data_100.jsonl \
    --test-data train_data/translate/test_10.jsonl \
    --out state.npz \
    --lr 0.01 --epochs 20 \
    --export-pth --pth-out state.pth

# 训练事件以 JSON lines 输出到 stdout (loss/std/lr/epoch),
# 未来 sidecar 直接消费此事件流驱动进度面板
```

### 5. 预览 + 评估

```bash
# A/B 预览: 有 state vs 无 state
uv run statetuner preview \
    --model models/converted/rwkv7-g1d-0.4b \
    --state state.pth \
    --prompt "由于连续降雨，部分地区出现了内涝" --ab

# 单独导出 npz → pth (也可在 train 时 --export-pth 一步完成)
uv run statetuner export --state state.npz --out state.pth
```

### 6. 在 RWKV Runner 挂载 (Windows)

导出的 `.pth` 可直接在 RWKV Runner 中作为模型的初始 state 加载。
Runner 检测到 `blocks.{i}.att.time_state` 键后自动启用 tuned-state 路径。
详见 [RWKV Runner 挂载验收指南](docs/Runner挂载验收.md)。

### 7. 回归测试

```bash
uv run pytest -q                     # 快测 (导出 round-trip + 推理 golden + 单元, ~17s)
uv run pytest --slow -q              # 全测 (含训练行为断言, ~4min)
```

- 快测:导出器 round-trip / 键名形状 / 转置方向 / 推理 golden / 数据与事件单元测试
- 慢测:梯度冒烟(24层 grad 非零)/ 过拟合(loss<0.5)/ 全量收敛 + 翻译

模型或 state 缺失时相关测试自动 skip 并提示获取方式。

---

## 关键技术决策

**为什么脱离 fla 自己写转换器**:官方 `convert_from_rwkv7.py` 依赖
`flash-linear-attention`,后者顶层 import 拉起 `fla.ops` → triton,而
triton 无 macOS wheel。本项目用 0.1B safetensors 作 ground truth 模板,
独立实现键名映射,脱离 fla 依赖。

**为什么 lr=0.01 而非 RWKV-PEFT 的 1.0**:实测 lr=1.0 导致 state 数值
爆炸(std 50~100 倍于正常值),变成无条件偏置。lr=0.01 让 state 温和
生长,保留对输入的条件响应。详见 [实验报告 §三](experiments/p0_translate/实验报告.md)。

**为什么训练用 ops、推理用 kernel**:ops 路径可微(每步有 VJP),
kernel 路径快但无 VJP。两者已验证在容差内等价,详见
[P0 理论指南 §二/§五](docs/P0-理论指南.md)。

---

## 致谢

- [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) — wkv7 前向实现(Apple)
- [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) — 转换规则参考
- [Joluck/RWKV-PEFT](https://github.com/Joluck/RWKV-PEFT) — state tuning 超参配方参考
- [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) — 原始权重与参考实现

本项目核心引擎是 Apple 的 mlx-lm,贡献在于 state tuning 的训练改造与工具链,
未重新实现 RWKV-7 内核或反向传播。

---

## License

<!-- TODO: 确定 license -->
