# RWKV Runner 挂载验收指南

> 导出的 `.pth` state 文件在 Windows + RWKV Runner 中真实挂载验收。
> RWKV Runner 是 RWKV 官方推荐的桌面推理应用,本项目导出的 state 应在此无缝挂载。
>
> **状态(2026-07-10)**:方向问题已修复并经 Windows 真机确认。导出器已改为 x070 noswap。

## 背景:为什么需要真实挂载验收

本项目用 MLX(Apple)训练 state,用 torch 导出 `.pth`,但下游消费者(RWKV Runner)
用的是 BlinkDL 的 CUDA/kernel 推理路径。Mac 侧的验证(round-trip / 挂载等价性)**只能证明
内部自洽,不能证明与 Runner 兼容**。真实环境(Runner + Windows + CUDA)的挂载是最终验收。

## 导出格式(对照 RWKV-Runner 源码确认)

| 项 | 值 | 来源 |
|---|---|---|
| 容器 | `torch.save(OrderedDict, path)` | RWKV-PEFT `trainer.py` |
| 键名 | `blocks.{i}.att.time_state` | RWKV-PEFT `att.py: RWKV_Tmix_x070_State` |
| 形状 | `(H, D, D)` = `(num_heads, head_dim, head_dim)` | 同上 |
| dtype | fp32 (`torch.float32`) | Runner 加载时 `.to(torch.float)` |
| 转置 | **x070: 存原样,不 swapaxes** | RWKV-Runner rwkv.py:843 version>=7 分支 |

**版本相关的转置约定(RWKV-Runner rwkv.py:836-857 的 `load_rwkv_state`):**

```python
if model.model.version >= 7:          # x070
    state_tuned[i*3+1] = time_state.to(float)        # 原样, 不 transpose
else:                                  # v5/v6
    state_tuned[i*3+1] = time_state.transpose(1,2)   # 只 v5/v6 转
```

我们的模型是 x070,Runner 判 version=7,**不 transpose**。故导出器存原样训练方向(`export_pth`
默认 `x070=True`)。官方 roleplay state 也是原样存储,同链路验证正常。

> ⚠️ 历史教训:初版导出器误按 v5/v6 假设做了 swapaxes,导致 x070 模型在 Runner 上注入转置方向
> 的 state,输出碎渣。已修正。详见 [P1-任务①收尾报告.md](P1-任务①收尾报告.md)。

Runner 的检测逻辑:加载 state 时校验层数和 `n_embd`,含 `.time_state` 键即启用 tuned-state 路径。

## 验收步骤(Windows + RWKV Runner)

### 1. 准备文件

从 Mac 侧导出(使用修正后的 CLI,默认 x070 noswap):
```bash
statetuner export --state experiments/p0_translate/checkpoints_v3/ep04.npz --out translate_state.pth
```

需要两个文件拷到 Windows:
- `translate_state.pth` — 训练的 state(本项目导出)
- 对应的 RWKV-7 0.4B 模型权重(G1 系列)— Runner 自带或从 BlinkDL 下载

### 2. 在 Runner 加载

1. 打开 RWKV Runner
2. 加载 RWKV-7 0.4B 模型(G1 系列)
3. Runner 应能自动识别 `translate_state.pth` 中的 `blocks.{i}.att.time_state` 键并加载为初始 state

### 3. 验收测试用例

输入格式对应训练数据约定(`User: {中文}\n\nAssistant:`):

| 输入 | 期望行为 |
|---|---|
| `User: 谢谢你的帮助。\n\nAssistant:` | 输出英文翻译尝试(含 thank/help 语义) |
| `User: 早上好。\n\nAssistant:` | 输出英文(含 morning/good 语义) |
| `User: The weather is nice.\n\nAssistant:`(英文) | 续写英文,不翻译(条件性) |

**通过标准(方向)**:
- 中文输入 → 英文翻译尝试(语义相关)
- state 加载前后行为有明显差异(证明 state 生效)

**已知限制(训练质量,非导出问题)**:
- 当前 ep04 state 的翻译质量不达翻译水准,表现为"英文单词硬凑+关键词命中但语义不完整"。
  这是训练侧问题,详见 [P1-已知问题.md](P1-已知问题.md)。

### 4. 对照:无 state 基线

同一批中文输入,不加载 state 文件时,Runner 应输出中文续写(不翻译)。
state 加载前后的行为差异 = 训练效果。

## Windows 真机验收记录(2026-07-10)

| 条件 | 输出 | 判读 |
|---|---|---|
| 无 state 对照 | 通顺中文续写 | ✅ 基模型/Runner/CUDA 健康 |
| 官方 roleplay state | 角色扮演行为正常 | ✅ Runner v7 state 路径健康 |
| 我们 swap 版(初版) | `onhooden that "" and onhooded...` 纯碎渣 | ❌ 方向错误 |
| 我们 noswap 版(修正后) | 无拼写错误英文,关键词命中,非翻译水准 | ✅ 方向正确 |

## 常见问题排查

**Q: 输出是乱码或重复碎片(如 "onhooden that...")**
→ 方向错误。x070 模型导出不应 swapaxes。确认用修正后的 CLI(默认 x070=True)重新导出。

**Q: Runner 报错 "key not found" 或 state 未生效**
→ 检查 pth 的 key 是否为 `blocks.{i}.att.time_state`:
```bash
uv run python -c "import torch; d=torch.load('translate_state.pth', map_location='cpu', weights_only=True); print(list(d.keys())[:3])"
```

**Q: 输出是英文但语义不通(单词硬凑)**
→ 方向正确,但训练质量不足。这是训练侧问题,非导出问题。见 [P1-已知问题.md](P1-已知问题.md)。
