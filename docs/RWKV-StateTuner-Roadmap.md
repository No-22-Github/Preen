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
| 推荐测试数据集 | HF `liumindmind/NekoQA-10K`(Apache 2.0) | 官方演示案例;不打包进 app,链接+致谢+引用 |
| 下游验证 | RWKV Runner(Windows CUDA 真机)/ Ai00 | Runner x070 路径加载 state 不 transpose(rwkv.py:843) |

**关键技术事实(已确认):**
- `_wkv7` 在 Mac 上默认走 Metal kernel,无反向传播;训练 patch 强制 `_wkv7_step_ops` ops 路径(可微)
- state 注入点:每层 `ArraysCache(size=3)` 的 `cache[1]`,形状 `(B, H, D, D)`
- 可训练参数量:1.5B 约 12MB,优化器开销可忽略
- **任务类型边界**:state tuning 擅长风格/人设/格式定向(NekoQA 200 条冒烟即成),不擅长内容映射(翻译实测不可用)。产品模板与文档按此设计

---

## Phase 0 — 脚本级验证 ✅(历史注记)

原始超参起点"lr 1.0 → 0.01 cosine"来自 v6 时代老教程,**已废弃勿再引用**;P0 实测 lr=1.0 导致 state 爆炸。现行默认 1e-2 + cos_decay(衰减至 /100)。其余验证项(梯度穿透/过拟合/等价性)全部通过,详见前置上下文文档。

---

## Phase 1 — 训练管线完整化 ✅

- [x] 数据管线:jsonl → World tokenizer → loss mask(目标段+终止符)
- [x] 训练循环产品化:epoch/step 控制、loss 记录、checkpoint 断点续训、state std 监控(**只记录不报警**——旧 >1.0 预警线已作废,官方 roleplay state std 1.385 工作正常,健康区间未标定)、held-out 早停、JSON lines 事件流
- [x] State 导出器:键名 `blocks.{i}.att.time_state`、(H,D,D)、fp32,**x070 原样方向不 swapaxes**(Windows 真机 + Runner 源码 rwkv.py:843 确认)
- [x] 端到端验证:pth 注入 == 训练 state 直注,逐字符一致;Windows Runner 真机挂载行为一致
- [x] CLI:`statetuner train/eval/export/preview`;回归 pytest 快测/--slow 全测
- [x] **管线对齐补丁**:终止符(token 0 进样本进 loss,修复循环输出根因)+ templates.py 模板单一事实源(P0_BARE/ZH_EN_LABELED/NEKO_QA)+ 训推同源分段编码 + L1 断言
- [x] NekoQA 冒烟:1.5B + 200 条,风格迁移完整成立,A/B 悬殊,自发终止正常

**关键技术结论:**
- lr 默认 1e-2 + cos_decay(官方现行配方区间 1e-3~1e-2);最佳值待 P4 前配方标定
- x070 导出方向:原样,不 transpose;v5/v6 才 transpose(1,2)
- torch 是必选依赖(导出用);**Mac 上 torch CPU 体积百 MB 量级**(旧记录"~2GB"是错的),切割手术的前提是 P3 打包后实测体积证明有必要

---

## ⚠️ 插队任务 — 训练内存异常排查(阻塞后续训练类工作)

现象:0.4B 训练 RSS 一晚从 ~4GB(翻译/ctx512)涨到 ~11GB(NekoQA),内存压力黄。
已确认:历史内存数字全部是 mx.get_peak_memory 口径,不含 wired/cache,显著低于 RSS。
待分解变量:ctx(512→1024?)/ 数据长度(NekoQA 长对话)/ 代码回归(EOS 修复动过 core)/ allocator cache。
详见《需求单-内存排查》。**结论出来前:不引用下方内存表做容量结论,不上 checkpointing,不把降 ctx 当正式方案。**

---

## Phase 2 — 内置推理预览(2~3 天,排查后开工)

不内嵌 llama.cpp(不支持挂载外部 init state),用 MLX 模型自己做预览。

- [ ] state 注入 `generate()` 封装(CLI preview --ab 已有,主要是封装)
- [ ] 采样参数(评估默认 top-p 0.9 轻采样;贪心仅用于可复现对比)
- [ ] A/B 对比预览

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
| 训练内存超预期(11GB 异常待归因;O(ctx) 是机制事实) | **排查中** | 差值分解 → 按结论决定 checkpointing 是否 v1 必做 |
| ops 循环太慢(逐 token 串行) | 中 | ctx512 可接受;chunked 并行或 Metal custom VJP 列 v1.1 |
| mlx-lm 上游改动破坏 patch | 低 | 锁版本;长期可 vendor rwkv7.py 训练版 |
| torch 依赖体积 | 低 | Mac 上百 MB 量级(旧记录 2GB 有误);P3 打包实测后再决定是否切割 |
| NekoQA 数据个别劣质样本进演示 | 低 | 示例子集固定行号清单,人工筛过 |

## 内存参考 — ⚠️ 口径修正中,暂停引用

历史数字(mx.get_peak_memory 口径,不含 wired/cache,低于真实 RSS):0.4B ctx512 1.39GB(RSS 实测 ~4GB)、1.5B ctx512 4.00GB、1.5B ctx1024 冒烟 7.87GB。
"16GB 双档舒适/无需 checkpointing"结论**挂起**,以排查产出的三口径数据(mx peak / mx cache / RSS)和经验公式为准重建本表。今后所有内存汇报必须三口径同报。
