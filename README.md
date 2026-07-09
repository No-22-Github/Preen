# Preen — RWKV-7 State Tuning for Mac

> 验证 RWKV-7 state tuning 技术路径在 Apple Silicon (MLX) 上跑通的 P0 实验。
>
> 目标:冻结模型全部权重,只训练每层 64×64 的初始状态矩阵 S₀,
> 让 0.4B 模型从该状态启动时表现出"中文→英文翻译"行为。

**当前阶段:P0 已完成**(技术风险验证通过,可进入 Phase 1 产品工程)。

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
docs/                           理论文档 (必读)
├── RWKVV-StateTuner-Roadmap.md   落地路线图
├── P0-理论指南.md                state tuning 原理 (判断代码对错的依据)
└── 参考仓库实现.md               依赖与参考来源

tools/                          模型转换工具
├── convert_rwkv7_to_hf.py        RWKV 原生 .pth → fla HF (独立, 不依赖 fla)
└── fla_cpu_bootstrap.py          macOS 无 triton 时短路 fla.ops (转换用)

experiments/p0_translate/        P0 实验 (uv 环境)
├── state_tuner.py                核心: patch ops 路径 + 可训练 state + generate
├── data_v2.py                    数据准备 (裸格式 + loss mask)
├── train_v3.py                   训练 (lr=0.01, 带 checkpoint)
├── final_eval.py                 held-out 验收
├── cross_check.py                ops vs kernel 自对照
├── report_examples.py            报告推理示例
├── 实验报告.md                    完整实验记录
├── tests/                        回归测试 (推理 golden + 训练行为断言)
│   ├── test_inference.py           快 (~17s)
│   └── test_train.py               慢 (~3.5min, --slow)
└── pyproject.toml

models/fla-hub-rwkv7-0.1B-g1/   World tokenizer (转换依赖, 提交进库)
train_data/translate/           训练数据 (中英翻译对, 自建无版权问题)
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

**魔搭仓库说明**:[Blink_DL/rwkv7-g1](https://modelscope.cn/models/Blink_DL/rwkv7-g1) 是 RWKV-7 G1
系列原始权重的合集,含 0.1B / 0.4B / 1.5B / 2.9B / 7.2B / 13.3B 多个规格的 `.pth` 文件。
本项目 P0 使用其中的 0.4B(`rwkv7-g1d-0.4b-20260210-ctx8192.pth`)。
也可用 `modelscope` CLI 下载:

```bash
pip install modelscope
modelscope download --model Blink_DL/rwkv7-g1 \
    rwkv7-g1d-0.4b-20260210-ctx8192.pth --local_dir models/
```

> tokenizer 文件已在本库 `models/fla-hub-rwkv7-0.1B-g1/`(体积小,转换必需)。
> 0.1B 的 `model.safetensors` 需另外下载,仅用于转换时的键名校验。

### 2. 转换模型

```bash
# 转换 0.4B: RWKV 原生 .pth → fla HF (safetensors)
python tools/convert_rwkv7_to_hf.py \
    --rwkv7 models/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
    --output models/converted/rwkv7-g1d-0.4b \
    --reference models/fla-hub-rwkv7-0.1B-g1/model.safetensors \
    --tokenizer-src models/fla-hub-rwkv7-0.1B-g1 \
    --precision bf16

# 验证转换正确 (应输出连贯中文)
python -c "from mlx_lm import load, generate; \
  m,t = load('models/converted/rwkv7-g1d-0.4b', \
  tokenizer_config={'trust_remote_code':True}); \
  print(generate(m,t,'User: 你好\n\nAssistant:',max_tokens=30))"
```

### 3. 训练

```bash
cd experiments/p0_translate
uv sync                              # 安装依赖
uv run python train_v3.py            # 训练 (~5min, lr=0.01)
cp checkpoints_v3/ep04.npz final_state_v3.npz   # 取 epoch4 作最终 state
```

### 4. 验收

```bash
uv run python final_eval.py          # held-out 翻译验收 + 条件性对照
uv run python report_examples.py     # 10 个完整推理示例
```

### 5. 回归测试

```bash
uv run pytest                        # 快测试 (推理 golden, ~17s)
uv run pytest --slow                 # 含训练测试 (~3.5min)
```

模型缺失时测试自动 skip 并提示转换命令。

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
