# P0-07 数据预检自动阈值实测（2026-07-18）

## 结论

训练配置页自动全量预检阈值锁定为 **10,000 条**。超过阈值不做抽样统计，也不在 ctx/template 变化时自动重算；用户必须显式运行一次最终配置的完整检查，结果完成前训练按钮保持不可用。

## 实测口径

- 设备：Apple M4，macOS 26。
- 数据：本仓 NekoQA-10K 全量 10,066 条。
- 模型/tokenizer：本地转换后的 RWKV-7 G1D 0.4B World tokenizer。
- 命令：`dataset-preview --training-data-route --template qa --ctx-len 512 --page-size 3`。
- 范围：冷进程启动、加载 tokenizer、读取全量数据、按最终 QA 模板渲染并逐条 tokenize、聚合统计及写首屏结果。
- 墙钟：2.94 秒。

10K 阈值取略低于实测数据规模的保守整数边界；macOS 14 未在当前机器上提供可运行环境，因此不伪造跨系统数据。实现只依赖 Foundation/Process/CryptoKit 与既有 Python 路径，并由部署目标编译和回归测试覆盖兼容性；发布矩阵仍应在 macOS 14 真机复核交互响应。
