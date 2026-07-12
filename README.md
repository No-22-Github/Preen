# Preen — RWKV-7 State Tuning for Mac

> Mac 原生的 RWKV-7 state tuning 工具。拖入 jsonl 数据集 → 选模型 → 训练 → 导出可挂载的 state 文件。
>
> 冻结模型全部权重,只训练每层 64×64 的初始状态矩阵 S₀,使模型从该状态启动时
> 表现出目标行为(说话风格、人设、输出格式等)。

**当前阶段:P2 已完成**(CLI + 独立推理引擎就绪,`.pth` 导出与 Runner 挂载已验证)。

- P0(技术验证):梯度穿透、收敛、泛化、ops/kernel 等价——[实验报告](experiments/p0_translate/实验报告.md)
- P1(产品化):CLI(train/eval/export/preview)、`.pth` 导出器、训练循环产品化、回归测试——[实验报告](docs/P1-实验报告.md)

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
| 100 条翻译实验能压低训练 loss | ✅，但后续实测不具备可靠内容映射能力 |
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
src/statetuner/                 正式包 (CLI 工具)
├── core.py                       patch ops 路径 + 可训练 state + generate
├── inference.py                  独立推理引擎 (采样/A-B/结构化结果)
├── data.py                       数据集 (jsonl → tokenize + loss mask)
├── templates.py                  ★ 格式模板单一事实源 (NEKO_QA / G1G)
├── chat.py                       交互式会话 (动态 state 切换 / A-B / 流式)
├── inspection.py                 环境/数据/state 预检 + 校验
├── metadata.py                   训练产物旁挂元数据
├── service.py                    应用用例编排 (CLI/未来 sidecar 共用)
├── events.py                     结构化训练事件 (为 sidecar IPC 铺路)
├── train.py                      训练循环 (lr/std 监控/早停/checkpoint/恢复)
├── export.py                     .pth 导出器 (RWKV Runner 可挂载) + round-trip 验证
└── cli.py                        CLI: train/eval/preview/chat/export + doctor/data-info/state-info

tests/                          回归测试 (改 src 必跑)
├── fixtures/                     NekoQA 基准 state (nekoqa_04b_s42.npz, 产品 CLI 训练)
├── golden/                       推理 golden 快照
└── ...                           10 个测试模块 (含 --slow 训练行为断言)

docs/                           文档
├── 快速上手.md                    ★ 分步教程 (首次微调必读)
├── RWKV-StateTuner-Roadmap.md    落地路线图
├── P0-理论指南.md                 state tuning 原理
├── P1-实验报告.md                 P1 产品化落地记录
├── P2-CLI收尾报告.md              CLI 收尾 (eval/chat 编排下沉)
├── 转换器零依赖化报告.md           转换器 fixture + tokenizer vendor
├── g1g-decode-alignment.md        g1g prompt 格式 token 级对齐
├── decision-precision.md          精度方案 + 内存红线标定 (裁决)
├── Runner挂载验收.md              Windows RWKV Runner 挂载步骤
└── 参考仓库实现.md                依赖与参考来源

tools/                          模型转换工具
├── convert_rwkv7_to_hf.py        RWKV 原生 .pth → fla HF (零依赖, 内置 fixture + tokenizer)
├── gen_convert_fixture.py        一次性生成 fixture (上游 schema 漂移时重跑)
├── fixtures/                     转换校验模板 (rwkv7_hf_template.json)
├── fla_cpu_bootstrap.py          macOS 无 triton 时短路 fla.ops (历史保留)
└── mem_probe*.py                 内存探针 (debug 用)

assets/
└── rwkv_world_tokenizer/         vendor 的 World tokenizer 5 文件 (转换器缺省 --tokenizer-src)
                                  + SOURCE.md (来源仓库 + 同步说明)

scripts/
└── nekoqa_smoke.sh               NekoQA × 1.5B smoke 全流程脚本

experiments/                     历史归档 (保留不动, 可复现性)
├── p0_translate/                  P0 翻译实验 (已废弃路径)
└── mixed_precision/               混合精度实验 (精度方案裁决依据)

train_data/NekoQA_10k/          NekoQA 数据集 (Apache-2.0, 见目录内 NOTICE.md)
```

---

## 快速开始

> 完整分步教程(含参数解释、预期 loss 曲线、FAQ)见
> **[docs/快速上手.md](docs/快速上手.md)**。

### 环境

- Apple Silicon Mac (M1+, 本项目在 M5 / 16GB 上验证)
- Python 3.11 (uv 自动管理)
- [uv](https://docs.astral.sh/uv/) 包管理器

```bash
uv sync                    # 安装依赖 (mlx-lm + torch + typer 等)
uv run statetuner --help   # 8 个子命令: train/eval/preview/chat/export + doctor/data-info/state-info
```

### 转换 + 训练 + 预览(三步最小流程)

```bash
# 1. 转换: RWKV 原生 .pth → fla HF (零依赖, fixture + tokenizer 已内置仓库)
uv run python tools/convert_rwkv7_to_hf.py \
    --rwkv7 models/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
    --output models/converted/rwkv7-g1d-0.4b --precision bf16

# 2. 训练 state tuning, 训完直接导出 RWKV Runner 可挂载的 .pth
uv run statetuner train \
    --model models/converted/rwkv7-g1d-0.4b \
    --data train_data/NekoQA_10k/nekoqa_smoke_200.json --template nekoqa \
    --out state.npz \
    --lr 0.01 --epochs 3 --ctx-len 512 --no-early-stop --seed 42 \
    --export-pth --pth-out state.pth

# 3. A/B 预览: 有 state vs 无 state, 直观看风格注入效果
uv run statetuner preview \
    --model models/converted/rwkv7-g1d-0.4b --state state.npz \
    --prompt "你好呀，今天想做什么？" --template nekoqa --ab
```

> **`--cache-limit-gb`**(train/eval/chat):默认 `auto` = 物理内存 × 25%
> (16G 机 ≈ 4.3G,c4G 同档)。可显式给 GB 数(如 `--cache-limit-gb 4`)
> 或 `auto` 覆盖;设小可降 RSS。该参数在模型加载前生效。

### 其他常用命令

```bash
# 模型常驻交互;运行中可用 /state 动态切换 state (默认 template=g1g, 猫娘风格迁移用 --template nekoqa)
uv run statetuner chat \
    --model models/converted/rwkv7-g1d-0.4b --state state.npz \
    --template nekoqa --max-tokens 200 --temperature 0.6 --top-p 0.7
# /state PATH | /state off | /ab on | /config | /help | /quit

# held-out 评估
uv run statetuner eval \
    --model models/converted/rwkv7-g1d-0.4b --state state.npz --template nekoqa \
    --data train_data/NekoQA_10k/nekoqa_smoke_200.json --limit 5

# 单独导出 npz → pth (也可在 train 时 --export-pth 一步完成)
uv run statetuner export --state state.npz --out state.pth

# clone 后先做环境/数据/state 自检
uv run statetuner doctor
uv run statetuner data-info --model models/converted/rwkv7-g1d-0.4b \
    --data train_data/NekoQA_10k/nekoqa_smoke_200.json --ctx-len 512
uv run statetuner state-info --state state.npz
```

### 在 RWKV Runner 挂载 (Windows)

导出的 `.pth` 可直接在 RWKV Runner 中作为模型的初始 state 加载。
Runner 检测到 `blocks.{i}.att.time_state` 键后自动启用 tuned-state 路径。
详见 [RWKV Runner 挂载验收指南](docs/Runner挂载验收.md)。

### 回归测试

```bash
uv run pytest -q                     # 快测 (导出 round-trip + 推理 golden + 单元, ~22s)
uv run pytest --slow -q              # 全测 (含训练行为断言, ~5min)
```

- 快测:导出器 round-trip / 键名形状 / 转置方向 / 推理 golden / 数据与事件单元测试 / 交互 session
- 慢测:梯度冒烟(24层 grad 非零)/ 过拟合(loss<0.5)/ 全量收敛 + NekoQA 风格注入

模型或 state 缺失时相关测试自动 skip 并提示获取方式。

---

## 关键技术决策

**为什么脱离 fla 自己写转换器**:官方 `convert_from_rwkv7.py` 依赖
`flash-linear-attention`,后者顶层 import 拉起 `fla.ops` → triton,而
triton 无 macOS wheel。本项目独立实现键名映射,并用 0.1B safetensors
生成的校验 fixture + vendor 的 World tokenizer 作为内置模板,
**转换全程零外部下载**。详见
[转换器零依赖化报告.md](docs/转换器零依赖化报告.md)。

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
- [fla-hub/rwkv7-0.1B-g1](https://huggingface.co/fla-hub/rwkv7-0.1B-g1) — World tokenizer 文件与转换校验模板来源(vendor 自本仓库 `assets/rwkv_world_tokenizer/` 与 `tools/fixtures/`)

本项目核心引擎是 Apple 的 mlx-lm,贡献在于 state tuning 的训练改造与工具链,
未重新实现 RWKV-7 内核或反向传播。

---

## License

<!-- TODO: 确定 license -->
