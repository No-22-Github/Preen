# RWKV State Tuner for Mac — 落地 Roadmap(2026-07-10 修订)

> 目标:一个 Mac 原生的 RWKV-7 state tuning 工具。拖入 jsonl 数据集 → 选模型 → 训练 → 导出可挂载的 state 文件。
> 技术底座:mlx-lm 官方 RWKV-7 实现(Apple 维护),训练循环自研,UI 用 SwiftUI + Python sidecar。
> 本版修订:内存数据标口径并挂起(排查中)、torch 体积错误修正、lr 老配方注记、演示案例定为 NekoQA、P4 模板品类按任务边界结论收窄。

---

## 依赖仓库与资源清单

| 用途 | 仓库 / 资源 | 说明 |
|---|---|---|
| 核心后端 | `ml-explore/mlx-lm` → `mlx_lm/models/rwkv7.py` | 完整 RWKV-7 前向,HF 权重直接加载 |
| 模型权重 | BlinkDL G1x 系列原生 .pth + 自研转换器 `tools/convert_rwkv7_to_hf.py` | fla-hub 现成 HF 仓库落后数代,不用;tokenizer 从 fla-hub 拷贝 |
| 格式参考 | `JL-er/RWKV-PEFT` | state 导出 `.pth` 键名格式。⚠️ 其老教程 lr=1.0 配方已废弃,现行官方建议 1e-3~1e-2 + cos_decay |
| 数据管线参考 | `BlinkDL/RWKV-LM(-V7)` → `make_data.py` | 终止符约定锚点(make_data.py:89,每样本尾 token 0);binidx 逻辑 |
| 推荐测试数据集 | HF `liumindmind/NekoQA-10K`(Apache 2.0) | 官方演示案例；v1.1 经内容与归属复核后内嵌固定 200 条子集，带 manifest / LICENSE / NOTICE，不内嵌完整 10K |
| 下游验证 | RWKV Runner(Windows CUDA 真机)/ Ai00 | Runner x070 路径加载 state 不 transpose(rwkv.py:843) |

**关键技术事实(已确认):**
- `_wkv7` 在 Mac 上默认走 Metal kernel,无反向传播;训练 patch 强制 `_wkv7_step_ops` ops 路径(可微)
- state 注入点:每层 `ArraysCache(size=3)` 的 `cache[1]`,形状 `(B, H, D, D)`
- 可训练参数量:1.5B 约 12MB,优化器开销可忽略
- **任务类型边界**:state tuning 擅长风格/人设/格式定向(NekoQA 200 条冒烟即成),不擅长内容映射(翻译实测不可用)。产品模板与文档按此设计

---

## Phase 0 — 脚本级验证 ✅(历史注记)

原始超参起点"lr 1.0 → 0.01 cosine"来自 v6 时代老教程,**已废弃勿再引用**;P0 实测 lr=1.0 导致 state 爆炸。产品现行默认峰值为 1e-4,cosine 衰减至 1e-5;历史 P0/P1 曲线仍按当时显式传入的 1e-2 配方解读。其余验证项(梯度穿透/过拟合/等价性)全部通过,详见前置上下文文档。

---

## Phase 1 — 训练管线完整化 ✅

- [x] 数据管线:jsonl → World tokenizer → loss mask(目标段+终止符)
- [x] 训练循环产品化:epoch/step 控制、loss 记录、checkpoint 断点续训、state std 监控(**只记录不报警**——旧 >1.0 预警线已作废,官方 roleplay state std 1.385 工作正常,健康区间未标定)、held-out 早停、JSON lines 事件流
- [x] State 导出器:键名 `blocks.{i}.att.time_state`、(H,D,D)、fp32,**x070 原样方向不 swapaxes**(Windows 真机 + Runner 源码 rwkv.py:843 确认)
- [x] 端到端验证:pth 注入 == 训练 state 直注,逐字符一致;Windows Runner 真机挂载行为一致
- [x] CLI:`statetuner train/eval/export/preview`;回归 pytest 快测/--slow 全测
- [x] **管线对齐补丁**:终止符(token 0 进样本进 loss,修复循环输出根因)+ templates.py 模板单一事实源(NEKO_QA;P0_BARE/ZH_EN_LABELED 为 P0 翻译实验遗留,已废弃)+ 训推同源分段编码 + L1 断言
- [x] NekoQA 冒烟:1.5B + 200 条,风格迁移完整成立,A/B 悬殊,自发终止正常

**关键技术结论:**
- 产品默认 lr=1e-4、lr_floor=1e-5 + cos_decay;历史验证配方与产品默认分开记录
- x070 导出方向:原样,不 transpose;v5/v6 才 transpose(1,2)
- torch 依赖已移除(2026-07):读写 `.pth` 改为纯 Python(`src/statetuner/pth_io.py` + `ml_dtypes`),转换产物与 torch 版逐字节相同。省 ~480MB,消除"MLX 原生却拖 torch"的观感

---

## ✅ 插队任务 — 训练内存异常排查(已完成,结论已固化)

**状态:已完成,阻塞解除。** 精度实验(`exp/precision` 分支)完成了内存归因 + 红线标定 + 多 seed 矩阵验证。

**结论:**
- **精度方案锁定:权重 bf16 + state fp32 训练**(即现有实现)。D 方案(state cast bf16 全程)未采纳——配对判据 4 红 1 绿,有确凿退化单例。详见 `docs/decision-precision.md`。
- **内存大头归因**:fp32 state 在 bf16 权重的 wkv 循环里,因 MLX 类型提升(bf16+fp32=fp32)把整个循环提升成 fp32。这是机制事实,非 bug。
- **红线标定**(16GB,bf16+c4G):安全档 L600(均 591 token),断点 L650(均 636 token,step_peak 顶削顶线 12.07G)。
- **实验脚本**归档于 `experiments/mixed_precision/`(裁决报告 `report_*.md` 本地留档不进 git)。

**对后续阶段的影响:**
- Phase 2 已完成，CLI 与未来 sidecar 共用独立推理 API。
- Phase 3「低内存模式」(gradient checkpointing):当前非必须——16GB 机器在 L600 安全档内可跑,checkpointing 降优先级。若未来要支持更长样本或更低端机器再启用。

---

## Phase 2 — 内置推理预览 ✅

不内嵌 llama.cpp(不支持挂载外部 init state),用 MLX 模型自己做预览。

- [x] 独立 `inference.py`:生成配置、state 注入、停止原因、结构化结果；不感知 CLI/UI
- [x] 采样参数:temperature / top-p / seed；temperature=0 保留贪心回归路径
- [x] A/B 对比:同配置/seed 下 tuned state vs 零 state，支持人类文本和 JSON 输出
- [x] 模型常驻 `chat`:每轮 fresh cache，运行中 `/state` 动态切换 S₀，支持 `/ab`
- [x] 模板级停止序列:NekoQA 在下一轮 `User:` 角色边界前停止，避免裸生成自问自答
- [x] stop-aware 流式输出:按解码文本检测并缓冲潜在角色边界前缀，不依赖上下文分词；`chat` 默认流式
- [x] CLI 检查入口:`doctor` / `data-info` / `state-info`

`core.generate()` 保留为兼容薄包装，现有 NekoQA golden 继续走贪心；CLI 与未来
sidecar 直接消费 `InferenceEngine` 的结构化结果。

---

## Phase 3 — Mac 原生 UI(2~3 周)

SwiftUI 壳 + Python sidecar(python-build-standalone 打包),IPC 走 JSON lines。

- [ ] Sidecar 协议:start_train / progress 事件流 / cancel / preview / export(直接消费 events.py)
- [ ] 核心界面:模型管理(下载+转换向导)、数据导入(jsonl 校验+预览;binidx 导入可选)、训练面板(loss+std 曲线)、预览对话、导出
- [ ] 任务模板系统:templates.py 是种子;**内置模板全部为风格/人设/格式类**;state 导出旁挂模板元数据 JSON(用户不该靠猜前缀)
- [ ] 「低内存模式」(gradient checkpointing)——**是否升级为 v1 必做,以内存排查结论定**
- [ ] 训练中断/恢复/App Nap 处理

**明确不做(v1 范围外):** LoRA/Bone、多卡、数据标注、量化训练。

---

## Phase 4 — 打磨与发布(1 周)

- [ ] 内存预检:按排查产出的经验公式 f(ctx, 样本长度, 模型档位) 查表预估
- [ ] 默认配方标定:lr × 数据量矩阵扫描,产出推荐最低数据量/默认 lr 实测数字;std 健康区间标定(官方 state 做参照)
- [ ] 预设模板:「人设对话」「文风迁移」「输出格式约束」等风格/格式类场景(**翻译不做模板**,作为文档能力边界示例)
- [ ] 签名公证(Apple Developer $99/年)
- [ ] README + 演示视频:**演示案例 = NekoQA 猫娘 state**;能力边界一节(翻译实测素材);附 v7 state .pth 格式规范(键名/形状/方向/终止符——可能是社区第一份成文规范);NekoBench 可用则接入
- [ ] 发 RWKV Discord / 社区群 / 即刻·V2EX

---

## 风险清单

| 风险 | 概率 | 应对 |
|---|---|---|
| 训练内存 O(ctx)(fp32 state 提升 wkv 循环,机制事实) | **已归因** | 16GB 机器 L600 安全档可跑;更长样本需 cache_limit 或降 ctx。checkpointing 降优先级 |
| ops 循环太慢(逐 token 串行) | 中 | ctx512 可接受;chunked 并行或 Metal custom VJP 列 v1.1 |
| mlx-lm 上游改动破坏 patch | 低 | 锁版本;长期可 vendor rwkv7.py 训练版 |
| torch 依赖体积 | **已解决** | 已移除 torch,读写 `.pth` 纯 Python 化(pth_io + ml_dtypes),省 ~480MB |
| NekoQA 数据个别劣质样本进演示 | 低 | 示例子集固定行号清单,人工筛过 |

## 内存参考(精度实验后,16GB 机器,bf16 权重 + fp32 state 训练)

红线标定数据(三口径,来自 `experiments/mixed_precision/`,详见 report_matrix.md 本地留档):

| 场景 | step_peak(mx) | 削顶线 | 状态 |
|---|---|---|---|
| 0.4B bf16+c4G L600(均 591 token) | ~11.7G | 12.07G | ✅ 安全档 |
| 0.4B bf16+c4G L650(均 636 token) | 12.22G | 12.07G | 🔴 断点(顶线) |
| fp32 比 bf16 step_peak 高 ~1.7G | — | — | fp32 红线更紧 |

`mx.get_peak_memory()` 不含 Metal wired memory,真实 RSS 更高(三口径同报,见 AGENTS.md「⚠️ 内存事实」)。今后所有内存汇报必须三口径同报 + 统一 GB 单位。
