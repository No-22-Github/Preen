<p align="center">
  <img src="assets/preen_title.png" alt="Preen — RWKV Fine-Tuning Studio" width="720">
</p>

# Preen — RWKV-7 State Tuning for Mac

> 在 Mac 上给 RWKV-7 做 state tuning。一个 SwiftUI 应用:
> 拖入 jsonl 数据集,选模型,训练,导出一个可挂载的 state 文件。
>
> 做法是冻结全部权重,只训练每层 64×64 的初始状态 S₀,替代默认的零初始状态。
> 原理和验证结果见[这是什么](#这是什么)。

首个预览版 [v0.1.0-beta.1](https://github.com/No-22-Github/Preen/releases/latest) 已发布,见[下载与安装](#下载与安装)。

[![CI](https://github.com/No-22-Github/Preen/actions/workflows/ci.yml/badge.svg)](https://github.com/No-22-Github/Preen/actions/workflows/ci.yml)
[![Build app](https://github.com/No-22-Github/Preen/actions/workflows/build-app.yml/badge.svg)](https://github.com/No-22-Github/Preen/actions/workflows/build-app.yml)
[![Release](https://img.shields.io/github/v/release/No-22-Github/Preen?include_prereleases&sort=semver)](https://github.com/No-22-Github/Preen/releases/latest)

App 底层是命令行工具 `statetuner`,同一套训练/推理引擎,可脚本化,自动化和排障走它。

- P0 技术验证:梯度穿透、收敛、泛化、ops/kernel 等价,见[实验报告](experiments/p0_translate/实验报告.md)
- P1 产品化:训练循环、`.pth` 导出器、回归测试
- P2 命令行:train / eval / preview / chat / export 全流程
- P3 图形界面:SwiftUI App,训练 / 对话 / 工具箱 / 导出

P1/P2 的实测数据与技术裁决在[工程实测数据](docs/工程实测数据.md)。

---

## 下载与安装

到 [Releases](https://github.com/No-22-Github/Preen/releases/latest) 下载,两个包都是 Apple Silicon 专用:

| 文件 | 最低系统 | |
|---|---|---|
| `Preen-macos26-arm64.zip` | macOS 26.2+ | 一般下这个 |
| `Preen-macos14-arm64.zip` | macOS 14.6+ | 老系统用这个 |

功能一样,差别只在 MLX wheel。26 版快不少:同一台 M5 上 1.5B 推理 prefill 快约 95%,
训练吞吐快 5%~17%(ctx64/256),系统够新就别下 14 版。

App 未签名未公证,首次打开会被 Gatekeeper 拦。解压后跑一次解除隔离再双击:

```bash
xattr -dr com.apple.quarantine /path/to/Preen.app
```

App 内嵌完整 Python + MLX 运行时,不用装依赖,准备一个 RWKV-7 模型就能开始。
手头没有的话,App 里「工具箱 → 模型转换」能直接从 BlinkDL / HF 权重转。首次启动有引导。

---

## 这是什么

RWKV-7 是线性注意力架构,每层维护一个矩阵值状态 S,随序列演化。常规推理里 S 从零开始;
state tuning 把初始值 S₀ 变成可训练参数,用梯度下降找一个等价于"虚拟前缀"的初始状态,
相当于把一段长 prompt 固化进模型,又不占上下文。

P0 阶段先验证了这条路走不走得通,详见[实验报告](experiments/p0_translate/实验报告.md):

| 命题 | 结果 |
|---|---|
| 梯度能穿透递归抵达每层 S₀ | 成立 |
| 优化器能把 10 条样本 loss 压到接近零 | 成立 |
| 100 条翻译实验能压低训练 loss | 成立,但实测不具备可靠的内容映射能力 |
| MLX 两条前向路径(ops/kernel)在容差内等价 | 成立 |
| tokenizer 与 llama.cpp 一致 | 成立 |

---

## 架构与依赖

整体两层:

```
SwiftUI App  (训练/对话/记录/工具箱)
      ↕ 常驻推理协议 + 一次性工具任务 JSON Lines
Python 引擎   (mlx-lm 训练/推理 · 本仓库)
```

训练、推理、导出都在 Python 引擎里,App 通过 IPC 调它,`statetuner` CLI 直接驱动它。

**核心引擎**是 [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) 里的 `rwkv7.py`(Apple 维护)。
wkv7 前向有两条等价路径:Metal kernel 快,推理用;纯 ops 循环可微,训练用。
这个仓库做的是在其上的训练改造,细节见 [docs](docs/)。

**反向传播**全部交给 MLX 自动微分(`mx.value_and_grad`),没有手写任何反向代码。
梯度能不能穿透,取决于前向是否走可微的 ops 路径,见[P0 理论指南 §二](docs/P0-理论指南.md)。

### 构建自包含 macOS App

```bash
python3 scripts/build_app.py
```

一次产出 `dist/Preen-macos14-arm64.app`(macOS 14.6+)和 `dist/Preen-macos26-arm64.app`(macOS 26.2+),
都内嵌 Apple Silicon CPython 3.11.15 和全部运行依赖,不含模型。
两版的性能差异见上面[下载与安装](#下载与安装)的实测数字。

构建机需要 Apple Silicon、Xcode Command Line Tools、`uv` 和网络。脚本会校验 PBS 归档哈希、
按精确 platform 下载 MLX wheel、构建两次 Release、跑隔离的 Python/Metal smoke test,
最后做 ad-hoc 签名校验——没有 Developer ID 签名和公证,所以用户侧才需要那步解除隔离。

---

## 仓库结构

<details>
<summary>展开目录树</summary>

```
src/statetuner/                 训练/推理引擎 + CLI 入口
├── core.py                       patch ops 路径 + 可训练 state + generate
├── inference.py                  独立推理引擎 (采样/A-B/结构化结果)
├── data.py                       数据集 (jsonl → tokenize + loss mask)
├── templates.py                  格式模板单一事实源 (QA / INSTRUCTION)
├── chat.py                       交互式会话 (动态 state 切换 / A-B / 流式)
├── inspection.py                 环境/数据/state 预检 + 校验
├── metadata.py                   训练产物旁挂元数据
├── service.py                    应用用例编排 (CLI/sidecar 共用)
├── events.py                     结构化训练事件 (sidecar IPC 用)
├── model_converter.py            原生 RWKV-7 .pth → HF safetensors
├── tool_events.py                离线工具任务 JSON Lines 事件协议
├── train.py                      训练循环 (lr/std 监控/早停/checkpoint/恢复)
├── export.py                     .pth 导出器 (RWKV Runner 可挂载) + round-trip 验证
├── pth_io.py                     纯 Python torch .pth 读写 (无 torch, bf16 靠 ml_dtypes)
└── cli.py                        CLI:训练/推理/模型转换/数据集预览与导入/检查

tests/                          回归测试 (改 src 必跑)
├── fixtures/                     NekoQA 基准 state (nekoqa_04b_s42.npz, 产品 CLI 训练)
├── golden/                       推理 golden 快照
└── ...                           10 个测试模块 (含 --slow 训练行为断言)

docs/                           文档
├── 快速上手.md                    分步教程,首次微调先看这个
├── RWKV-StateTuner-Roadmap.md    落地路线图
├── P0-理论指南.md                 state tuning 原理
├── 工程实测数据.md                 P1/P2 实测数据 + 技术裁决汇总
├── 转换器零依赖化报告.md           转换器 fixture + tokenizer vendor
├── g1g-decode-alignment.md        g1g prompt 格式 token 级对齐
├── decision-precision.md          精度方案 + 内存红线标定
├── Runner挂载验收.md              Windows RWKV Runner 挂载步骤
└── 参考仓库实现.md                依赖与参考来源

tools/                          模型转换工具
├── convert_rwkv7_to_hf.py        模型转换兼容入口 (正式实现在 src/statetuner)
├── gen_convert_fixture.py        一次性生成 fixture (上游 schema 漂移时重跑)
├── fixtures/                     转换校验模板 (rwkv7_hf_template.json)
├── fla_cpu_bootstrap.py          macOS 无 triton 时短路 fla.ops (历史保留)
└── mem_probe*.py                 内存探针 (debug 用)

assets/
└── rwkv_world_tokenizer/         vendor 的 World tokenizer 5 文件 (转换器缺省 --tokenizer-src)
                                  + SOURCE.md (来源仓库 + 同步说明)

scripts/
├── build_app.py                   一次产出 macOS 14 / 26 两个自包含 .app
└── nekoqa_smoke.sh                NekoQA × 1.5B smoke 全流程脚本

experiments/                     历史归档 (保留不动, 可复现性)
├── p0_translate/                  P0 翻译实验 (已废弃路径)
└── mixed_precision/               混合精度实验 (精度方案裁决依据)

train_data/NekoQA_10k/          NekoQA 数据集 (Apache-2.0, 见目录内 NOTICE.md)
```

</details>

---

## 命令行

日常用 App 就够了。想脚本化、或不想开图形界面,就走 `statetuner` CLI——和 App 同一套流程。
`uv sync` 装依赖(无 torch),`uv run statetuner --help` 列全部子命令。

完整教程(参数解释、预期 loss 曲线、FAQ)见 **[docs/快速上手.md](docs/快速上手.md)**;
导出的 `.pth` 在 RWKV Runner 的挂载步骤见 [挂载验收指南](docs/Runner挂载验收.md)。

<details>
<summary>三步最小流程:转换 → 训练 → 预览</summary>

```bash
# 1. 转换: RWKV 原生 .pth → fla HF (零外部下载, fixture + tokenizer 已内置仓库)
uv run statetuner convert-model \
    --rwkv7 models/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
    --out models/converted/rwkv7-g1d-0.4b --precision bf16

# 2. 训练 state tuning, 训完直接导出 RWKV Runner 可挂载的 .pth
uv run statetuner train \
    --model models/converted/rwkv7-g1d-0.4b \
    --data train_data/NekoQA_10k/nekoqa_smoke_200.json --template qa \
    --out state.npz \
    --lr 0.01 --epochs 3 --ctx-len 512 --no-early-stop --seed 42 \
    --export-pth --pth-out state.pth

# 3. A/B 预览: 有 state vs 无 state, 直观看风格注入效果
uv run statetuner preview \
    --model models/converted/rwkv7-g1d-0.4b --state state.npz \
    --prompt "你好呀,今天想做什么?" --template qa --ab
```

`--cache-limit-gb`(train/eval/chat 通用)默认 `auto`,取物理内存的 25%(16G 机约 4.3G);
设小能降 RSS,在模型加载前生效。

</details>

<details>
<summary>其他命令:chat / eval / export / 自检 / 测试</summary>

```bash
# 模型常驻交互;运行中可用 /state 动态切换 state
# (默认裸 qa 模板;G1 系列 reasoning 模型加 --reasoning --think fast 避免降智)
uv run statetuner chat \
    --model models/converted/rwkv7-g1d-0.4b --state state.npz \
    --template qa --max-tokens 200 --temperature 0.6 --top-p 0.7
# /state PATH | /state off | /ab on | /config | /help | /quit

# held-out 评估
uv run statetuner eval \
    --model models/converted/rwkv7-g1d-0.4b --state state.npz --template qa \
    --data train_data/NekoQA_10k/nekoqa_smoke_200.json --limit 5

# 单独导出 npz → pth (也可在 train 时 --export-pth 一步完成)
uv run statetuner export --state state.npz --out state.pth

# clone 后先做环境/数据/state 自检
uv run statetuner doctor
uv run statetuner data-info --model models/converted/rwkv7-g1d-0.4b \
    --data train_data/NekoQA_10k/nekoqa_smoke_200.json --ctx-len 512
uv run statetuner state-info --state state.npz

# 回归测试:快测 ~22s / 全测含训练断言 ~5min
uv run pytest -q
uv run pytest --slow -q
```

</details>

---

## 一些取舍

几个可能反直觉的地方:脱离 fla 自写转换器、lr 用 0.01、训练走 ops 推理走 kernel、不依赖 torch。

<details>
<summary>展开</summary>

**转换器为什么脱离 fla 自己写。** 官方 `convert_from_rwkv7.py` 依赖 `flash-linear-attention`,
后者顶层 import 会拉起 `fla.ops`,进而拉起 triton,而 triton 没有 macOS wheel,这条路在 Mac 上走不通。
所以自己实现了键名映射,把 0.1B safetensors 生成的校验 fixture 和 vendor 的 World tokenizer
内置进仓库,整个转换过程零外部下载。详见[转换器零依赖化报告](docs/转换器零依赖化报告.md)。

**学习率为什么是 0.01,而不是 RWKV-PEFT 用的 1.0。** 实测 lr=1.0 会让 state 数值爆炸,
std 冲到正常值的 50~100 倍,state 退化成一个无条件偏置。lr=0.01 让 state 温和生长,
保留对输入的条件响应。详见[实验报告 §三](experiments/p0_translate/实验报告.md)。

**训练为什么用 ops、推理为什么用 kernel。** ops 路径可微,每一步都有 VJP,训练需要它;
kernel 路径快但没有 VJP。两条路径已验证在容差内等价,见
[P0 理论指南 §二/§五](docs/P0-理论指南.md)。

**为什么不依赖 torch。** RWKV 的 `.pth` 是 torch 用 zip+pickle 存的,整个项目唯一需要 torch 的地方
就是读原始权重、写导出的 state。为两个 I/O 点扛 480MB 的 torch 不划算,也和 MLX 原生的定位别扭。
所以 `pth_io.py` 用纯 Python 复刻了这套格式:读端与 `torch.load` 逐字节等价
(3 个真实模型 798/798 张量验证),写端产物 RWKV Runner 可直接挂载,与 torch 版逐字节相同。
bf16 靠 `ml_dtypes`(3.8MB)补上 numpy 缺的类型。

</details>

---

## 致谢

清单与 App「关于 Preen」里的功勋墙一致:

| 项目 | 许可证 | 作用 |
|---|---|---|
| [MLX](https://github.com/ml-explore/mlx) | MIT | Apple 机器学习框架,张量运算与自动微分 |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | MIT | 核心训练/推理引擎,提供 `rwkv7.py` 前向 |
| [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) | MIT | 线性注意力上游库,模型转换校验基准 |
| [RWKV-PEFT](https://github.com/Joluck/RWKV-PEFT) | Apache-2.0 | RWKV 参数高效微调方法参考 |
| [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) | Apache-2.0 | BlinkDL 维护的 RWKV 模型仓库,参考实现 |
| [BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1) | Apache-2.0 | RWKV-7 G1 官方权重,实际下载与转换的来源 |
| [RWKV Runner](https://github.com/josStorer/RWKV-Runner) | MIT | 导出 `.pth` 的挂载目标,与 RWKV 生态直连 |
| [NekoQA-10K](https://huggingface.co/datasets/liumindmind/NekoQA-10K) | Apache-2.0 | 猫娘风格 QA 数据集,风格迁移训练数据 |
| [rwkv7-0.1B-g1](https://huggingface.co/fla-hub/rwkv7-0.1B-g1) | Apache-2.0 | World Tokenizer 与转换校验模板来源 |
| [Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | 转换链路的 HF 格式基准 |
| [safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 | 转换产物张量格式 (`.pth` → HF safetensors) |
| [swift-markdown-ui](https://github.com/gonzalezreal/swift-markdown-ui) | MIT | App 内 Markdown 渲染 |
| [uv](https://docs.astral.sh/uv/) | Apache-2.0 | 构建工具链与依赖管理 |

许可证信息以各项目仓库为准。

核心引擎是 Apple 的 mlx-lm。这个项目做的是 state tuning 的训练改造和整套工具链,
没有重新实现 RWKV-7 内核,也没有重写反向传播。

---

## License

本项目以 [Apache License 2.0](LICENSE) 发布。

```
Copyright 2026 No-22-Github (https://github.com/No-22-Github/Preen)
```

依赖与参考项目各自的许可以其上游为准:mlx-lm(MIT)、flash-linear-attention(MIT)、
NekoQA-10K 数据集(Apache-2.0,见 `train_data/NekoQA_10k/NOTICE.md`)。