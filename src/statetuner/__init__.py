"""RWKV-7 State Tuner for Mac.

冻结模型全部权重,只训练每层的初始状态矩阵 S₀。
核心引擎:Apple 维护的 mlx-lm rwkv7 前向 + 自研训练循环。

子模块:
  core   — patch ops 路径 + 可训练 state + 兼容 generate
  inference — 独立生成/采样/A-B API
  chat   — 模型常驻交互与动态 state 会话控制
  data   — JSON/JSONL 数据集 → tokenize + loss mask
  inspection — 环境、数据与 state 预检
  service — CLI/未来 sidecar 共用的应用编排
  events — 结构化训练事件(为 sidecar IPC 铺路)
  train  — 训练循环(lr/std 监控/早停/checkpoint/恢复)
  export — .pth 导出器(RWKV Runner 可挂载)
  cli    — 训练/推理/导出与检查入口
"""
__version__ = "0.1.0"
