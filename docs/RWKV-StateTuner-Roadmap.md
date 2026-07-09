# RWKV State Tuner for Mac — 落地 Roadmap

> 目标:一个 Mac 原生的 RWKV-7 state tuning 工具。拖入 jsonl 数据集 → 选模型 → 训练 → 导出可挂载的 state 文件。
> 技术底座:mlx-lm 官方 RWKV-7 实现(Apple 维护),训练循环自研,UI 用 SwiftUI + Python sidecar。

---

## 依赖仓库与资源清单

| 用途 | 仓库 / 资源 | 说明 |
|---|---|---|
| 核心后端 | `ml-explore/mlx-lm` → `mlx_lm/models/rwkv7.py` | 完整 RWKV-7 前向,HF 权重直接加载 |
| 模型权重 | `RWKV/RWKV7-Goose-World3-1.5B-HF`、`fla-hub/rwkv7-*-world` | HF 格式,191M / 0.4B / 1.5B / 2.9B,含 World tokenizer |
| 超参 & 格式参考 | `JL-er/RWKV-PEFT` | 只抄两样:state tuning 超参配方、state 导出 `.pth` 的键名格式 |
| 数据管线参考 | `BlinkDL/RWKV-LM` → `make_data.py` | jsonl → binidx 的处理逻辑,可简化重写 |
| 下游验证 | RWKV Runner / Ai00 | 训出的 state 文件在这两个里挂载验证兼容性 |

**关键技术事实(已确认):**
- `_wkv7` 在 Mac 上默认走手写 Metal kernel(`mx.fast.metal_kernel`),**无反向传播**;训练必须 patch 掉分发逻辑,强制走 `_wkv7_step_ops` 纯 ops 路径(可微,已 `mx.compile`)
- state 注入点:每层 `ArraysCache(size=3)` 的 `cache[1]`,形状 `(B, H, D, D)`,None 时零初始化
- 可训练参数量:1.5B 模型约 12MB(24 层 × 32 头 × 64 × 64),优化器开销可忽略

---

## Phase 0 — 脚本级验证(1~2 个晚上)

**目标:回答唯一的技术风险问题——梯度能否穿透 ops 路径,loss 是否收敛。**

- [ ] 环境:`pip install mlx mlx-lm`,Python 3.11+,venv 隔离
- [ ] `mlx_lm.load("fla-hub/rwkv7-0.4B-world")` 加载模型,先跑通推理确认权重正常
- [ ] Monkeypatch `Rwkv7TimeMixing._wkv7`,强制走 `_wkv7_step_ops` 循环路径
- [ ] 构造 per-layer 可训练 state 参数(dict of `mx.array`),注入 `cache[1]`
- [ ] 冻结全部模型权重,`mx.value_and_grad` 只对 state 求导
- [ ] **梯度冒烟测试**:单 batch 跑一步,检查每层 state 的 grad 非零、无 NaN
- [ ] **过拟合测试**:10 条固定样本训 200 步,loss 应显著下降(能过拟合 = 管线正确)
- [ ] 记录:0.4B @ ctx512 @ bsz1 的每步耗时和峰值内存(Activity Monitor / `mx.get_peak_memory()`)

**通过标准:** loss 从 ~3.x 降到 <0.5(过拟合场景)。不通过则排查 patch 是否生效、state 是否真的进了计算图。

**超参起点(抄 RWKV-PEFT):** lr 1.0 → 0.01 cosine 衰减、warmup 10 步、ctx_len 512、bsz 1、Adam(0.9, 0.99)。state tuning 的学习率就是这么大,不是笔误。

---

## Phase 1 — 训练管线完整化(1 周)✅

**目标:从「能跑」到「训出真正可用的 state 文件」。**

- [x] 数据管线:jsonl(`{"text": "User: ...\n\nAssistant: ..."}` 或裸格式)→ World tokenizer → padding + loss mask(只对 Assistant 段算 loss)
- [x] 训练循环产品化:epoch / step 控制、loss 曲线记录、checkpoint 保存、中断恢复、state std 监控(>1.0 预警)、held-out 早停、结构化事件输出(为 IPC 铺路)
- [x] **State 导出器**:MLX array → PyTorch `.pth`,键名 `blocks.{i}.att.time_state`、shape `(H,D,D)`、fp32,转置方向已验证(round-trip max diff 0.0)
- [x] **端到端验证**(Mac 侧闭环):挂载等价性——pth 注入 MLX generate 输出 == 训练 state 直接注入(逐字符一致);Runner 真实挂载验收由 Windows 环境完成(见 [Runner挂载验收.md](Runner挂载验收.md))
- [x] CLI 收口:`statetuner train/eval/export/preview` 四子命令(`src/statetuner/`)
- [x] 回归测试固化:一条命令(`pytest`)跑完快测(~17s),`--slow` 含训练(~4min)
- [x] 1.5B 内存边界实测:峰值 4.00GB(M5 16GB 舒适运行,无需 checkpointing);详见下方内存表

**产出:** `statetuner` CLI 工具(`pip install -e .` 后可用),可发布给社区尝鲜。

**关键技术结论(P1 新增):**
- **lr 默认 0.01 而非 1.0**(P0 实测修正):1.0 导致 state 爆炸(std 7~13),0.01 温和生长(std 0.1~0.2)
- **转置暗坑**:MLX state 与 BlinkDL CUDA kernel 同向,Runner 加载统一 `.transpose(1,2)`,故导出前必须 transpose(数值已验证)
- **torch 是必选依赖**(导出器用 `torch.save`);未来切割 torch 减少打包体积作为优化项(手搓零依赖 pickle 导出器)

---

## Phase 2 — 内置推理预览(2~3 天)

**结论:不内嵌 llama.cpp,用已有的 MLX 模型自己做预览。理由:**

1. 训练进程里模型本来就加载着,预览 = 把训好的 state 塞进 `cache[1]` 然后 `generate()`,推理走 Metal kernel 路径(快),**零额外依赖、零格式转换**
2. llama.cpp 虽支持 RWKV-7 GGUF 推理,但**不支持挂载独立训练的 init state 文件**,嵌进来根本实现不了「预览我刚训的 state」这个需求
3. 内嵌 llama.cpp 意味着多维护一套 GGUF 转换管线和 C++ 依赖,违背「工具要简单」的原则

**推理的分层策略:**
- 应用内:轻量对话预览(单会话、无历史管理),目的只是「训完立刻试效果」,可做 A/B(挂 state vs 不挂)对比
- 日常使用:导出 `.pth` 后引导用户去 RWKV Runner / Ai00,不重复造轮子

- [ ] 实现 state 注入的 `generate()` 封装
- [ ] A/B 对比预览(同 prompt,有/无 state 双输出)

---

## Phase 3 — Mac 原生 UI(2~3 周)

**架构:SwiftUI 壳 + Python sidecar。**

```
┌─────────────────────────────┐
│  SwiftUI App                │
│  拖放数据集 / 模型选择 /     │
│  训练面板(loss 曲线)/      │
│  预览对话 / 导出            │
└──────────┬──────────────────┘
           │ 本地 IPC(HTTP localhost 或 stdin/stdout JSON lines)
┌──────────▼──────────────────┐
│  Python Sidecar(打包进 .app)│
│  mlx-lm + 训练循环 + 推理    │
└─────────────────────────────┘
```

- [ ] Sidecar 协议设计:start_train / progress 事件流 / cancel / preview / export
- [ ] Python 环境打包:python-build-standalone 嵌入 .app,或首启动自动建 venv(前者体验好,后者包小,建议前者)
- [ ] 核心界面:模型管理(HF 下载 + 本地缓存)、数据集导入(jsonl 校验 + 预览)、训练面板(实时 loss 曲线、剩余时间估计)、预览对话、导出
- [ ] 「低内存模式」开关(= gradient checkpointing),文案说人话:"训练慢约 30%,内存省约 80%,8/16GB 机型推荐开启"
- [ ] 训练中断 / 恢复 / 后台运行(App Nap 处理)

**明确不做(v1 范围外):** LoRA/Bone 等其他微调方法、多卡/分布式、数据标注功能、模型量化训练。

---

## Phase 4 — 打磨与发布(1 周)

- [ ] 内存预检:根据用户机器内存 + 所选模型,训练前预估并提示(拿 Phase 1 实测数据做查表)
- [ ] 预设模板:「说话风格」「角色扮演」「翻译」等场景的推荐超参 + 数据格式示例
- [ ] 签名公证(Apple Developer $99/年,做 GitHub 分发就必须要)
- [ ] README + 演示视频,发 RWKV Discord / 社区群 / 即刻·V2EX 类渠道

---

## 风险清单

| 风险 | 概率 | 应对 |
|---|---|---|
| ops 路径梯度有隐性问题(mx.compile 与 grad 交互) | 低 | Phase 0 首要验证项;兜底是去掉 `@mx.compile` 装饰器裸跑 |
| state `.pth` 键名/形状与下游不兼容 | 中 | 对照 RWKV-PEFT 源码 + Runner 实测,Phase 1 验收标准 |
| ops 循环太慢(逐 token 串行) | 中 | ctx512 场景可接受;后续可写 chunked 并行版或给 Metal kernel 补 custom VJP(v2 优化项) |
| 16GB 上 1.5B 训不动 | 中 | `mx.checkpoint` + `iogpu.wired_limit_mb` 放宽;实在不行 v1 只支持到 0.4B,1.5B 标注"需 24GB+" |
| mlx-lm 上游改动破坏 patch | 低 | 锁定依赖版本;长期可把 rwkv7.py 复制进项目自己维护训练版 |
| torch 依赖体积大(~2GB),拖累打包 | 中 | P1 用 torch.save 导出(成熟可靠);未来切割 torch 改手搓零依赖 pickle 导出器(legacy tar 格式),减少打包体积 |

## 内存参考(ctx 512, bsz 1, bf16)— P1 实测

| 模型 | 权重 | 训练峰值(实测) | 每步耗时 | 16GB M5 |
|---|---|---|---|---|
| 0.4B | ~0.9GB | **1.39GB** | 0.18s | ✅ 舒适 |
| 1.5B | ~3.0GB | **4.00GB** | 0.29s | ✅ 舒适(无需 checkpointing) |
| 2.9B | ~6GB | >10GB(估) | - | ⚠️ 需实测;16GB 边缘,24GB 安全 |

**P1 实测结论(M5 / 16GB):**
- 0.4B 和 1.5B 都在 16GB 上**舒适运行**,无需 gradient checkpointing
- 1.5B 峰值仅 4GB,远低于 P0 预估的 8~11GB(预估偏保守)
- **v1 支持档位:0.4B + 1.5B 都官方支持**;2.9B 标注"建议 24GB+"
- 每步耗时:1.5B 比 0.4B 慢约 60%(0.29s vs 0.18s),可接受
