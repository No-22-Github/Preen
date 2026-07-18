# NekoQA 200 `1.1.0-1` 内容审查记录

- 审查日期：2026-07-18
- 源文件：`train_data/NekoQA_10k/NekoQA-10K.json`
- 源文件 SHA-256：`b4d260ad117c29c9fd64abcb513ad24d62e2fac383640e17ea230d12ae03b849`
- 发行文件：`macos/Preen/Resources/Datasets/NekoQA200/nekoqa_200.json`
- 发行文件 SHA-256：`435f9a3ac9d5b1151fb917955fe94180a82da8602a23a0d3a84dd105dcc5939f`
- 样本数：200

## 审查方法

以现有 smoke 前 200 条作为起点，但不直接发行前 200 条。逐条阅读候选的
`instruction` 与 `output`，只按源索引筛选，不改写文本；随后从完整源文件补充
通过审查的条目。最终索引固定写入 `manifest.json`。

人工排除重点：恋人/婚姻关系、所有权与绝对服从、永不离开或排他性陪伴、鼓励
情感依赖、露骨或性化互动、危险行为、医疗/法律/财务建议、明显乱码或脱离角色、
异常长文本与会损害默认示例信任的条目。源索引 0 的“永远等着你”即因依赖性
表述被排除，现有 `nekoqa_smoke_200.json` 因而只保留为历史 smoke 基线。

自动检查覆盖：

- 恰好 200 条，字段集合为非空 `instruction` / `output`；
- instruction 完全重复为 0；按规范化文本 `SequenceMatcher >= 0.86` 的近似重复为 0；
- output 按 `SequenceMatcher >= 0.92` 的近似重复为 0；
- 命中隐私、露骨内容、危险指导、专业建议、婚恋/所有权/绝对依赖词表为 0；
- 文本总长范围 59–308 字，中位数 100 字；
- 场景覆盖包含问候、闲聊、情绪回应、行为描写和角色称谓。

训练有效性由发布校验测试与 `statetuner data-info` 锁定：QA 模板、默认
`ctx_len=512` 下必须得到 200 条有效样本、`target_fully_truncated=0`。

## 归属复核

2026-07-18 复核上游 Hugging Face 数据集卡：仓库 owner 为 `liumindmind`，
许可证元数据为 `apache-2.0`，卡片引用作者为 `MindsRiverPonder`。上游未声明可
照抄的版权年份，因此发行 NOTICE 不沿用仓库旧文件中未经卡片支持的
“Copyright 2024 liumindmind”，只记录可验证的 owner、citation author、来源与许可。
