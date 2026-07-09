# RWKV Runner 挂载验收指南

> 导出的 `.pth` state 文件在 Windows + RWKV Runner 中真实挂载验收。
> RWKV Runner 是 RWKV 官方推荐的桌面推理应用,本项目导出的 state 应在此无缝挂载。

## 背景:为什么需要真实挂载验收

本项目用 MLX(Apple)训练 state,用 torch 导出 `.pth`,但下游消费者(RWKV Runner)
用的是 BlinkDL 的 CUDA/kernel 推理路径。我们在 Mac 上做了两层验证:

1. **round-trip 数值验证**:导出 pth → `torch.load` 读回 → transpose → allclose 原始 state(max diff 0.0)
2. **挂载等价性验证**:pth 注入 MLX generate 的输出 == 直接用训练 state 的输出(逐字符一致)

但这两层都在 MLX 侧闭环。**真实环境(Runner + Windows + CUDA)的挂载是最终验收**,
由本指南记录。

## 导出格式(已对照源码确认)

| 项 | 值 | 来源 |
|---|---|---|
| 容器 | `torch.save(OrderedDict, path)` | RWKV-PEFT `trainer.py` |
| 键名 | `blocks.{i}.att.time_state` | RWKV-PEFT `att.py: RWKV_Tmix_x070_State` |
| 形状 | `(H, D, D)` = `(num_heads, head_dim, head_dim)` | 同上 |
| dtype | fp32 (`torch.float32`) | Runner 加载时 `.to(torch.float)` |
| 转置 | 导出前已 `transpose(1,2)`,Runner 加载再 `transpose(1,2)` 还原 | 双方源码一致 |

**转置暗坑**(已解决):MLX 训练的 state 与 BlinkDL CUDA kernel 同向(`S[i,j]+=v[i]k[j]`)。
Runner 加载时统一对 `.pth` 里的张量做 `.transpose(1,2)` 再喂给 kernel。因此导出时
必须先 transpose,这样 Runner 转回来恰好 = 训练时的 state。

Runner 的检测逻辑(`rwkv_pip/model.py`):扫描 state dict 的 key,只要含 `.time_state`
就自动启用 tuned-state 路径,无需任何额外配置。

## 验收步骤(Windows + RWKV Runner)

### 1. 准备文件

从 Mac 侧导出(已完成):
```bash
# 已训好的翻译 state (0.4B, 中→英翻译)
statetuner export --state experiments/p0_translate/checkpoints_v3/ep04.npz --out translate_state.pth
```

需要两个文件拷到 Windows:
- `translate_state.pth` — 训练的 state(本项目导出)
- 对应的 RWKV-7 0.4B 模型权重(GGUF 或 pth)— Runner 自带或从 BlinkDL 下载

### 2. 在 Runner 加载

1. 打开 RWKV Runner
2. 加载 RWKV-7 0.4B 模型(G1 系列)
3. **关键**:Runner 应能自动识别 `translate_state.pth` 中的 `blocks.{i}.att.time_state`
   键并加载为初始 state。如果 Runner 有显式的 "加载 State 文件" 入口,选择该 pth。

### 3. 验收测试用例

以下输入对应训练数据的格式约定(`{中文}\n` 前缀)。输入时**确保带上 `\n`**:

| 输入 | 期望行为 |
|---|---|
| `今天下午三点开会\n` | 输出英文翻译尝试(含 meeting/document 语义) |
| `由于连续降雨\n` | 输出英文(含 rain 语义) |
| `人工智能技术\n` | 输出英文(含 technology/AI 语义) |
| `The weather is nice\n`(英文) | **续写英文**,不翻译(条件性:英文输入不触发中→英) |
| `1234567890\n`(乱码) | 不产出中文翻译(条件性:方向正确) |

**通过标准**:
- 中文输入 → 英文翻译尝试(语义相关,允许笨拙错词,0.4B 能力有限)
- 英文/乱码输入 → 不反向翻译成中文

### 4. 对照:无 state 基线

同一批中文输入,不加载 state 文件时,Runner 应输出中文续写(不翻译)。
state 加载前后的行为差异 = 训练效果。

## 常见问题排查

**Q: Runner 报错 "key not found" 或 state 未生效**
→ 检查 pth 的 key 是否为 `blocks.{i}.att.time_state`。在 Mac 侧验证:
```bash
uv run python -c "import torch; d=torch.load('translate_state.pth', map_location='cpu', weights_only=True); print(list(d.keys())[:3])"
```

**Q: 翻译方向反了(英文→中文)**
→ 转置方向错。本项目已处理(导出前 transpose)。若仍出错,检查 Runner 版本
是否对 `time_state` 做了 `transpose(1,2)`。

**Q: 输出是乱码或重复碎片**
→ state 数值爆炸。检查训练时 state std(P0 经验:正常 0.01~0.3,爆炸 >1.0)。
训练时 `--max-state-std 1.0` 会发预警。

## 技术验证记录(Mac 侧)

- round-trip max diff: 0.0(导出 → 读回 → transpose → allclose 原始)
- 挂载等价性: pth 注入输出 == 训练 state 直接注入输出(95 chars 完全一致)
- 推理 golden 测试: 6 passed(翻译/条件性/基线全覆盖)

这些在 Mac 侧闭环验证通过。Windows Runner 是最终的真实环境确认。
