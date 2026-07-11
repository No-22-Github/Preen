# NekoQA-10K 数据集 — 归属与许可说明

## 数据集信息

| 项 | 值 |
|---|---|
| **名称** | NekoQA-10K |
| **作者** | `liumindmind` |
| **来源** | [huggingface.co/datasets/liumindmind/NekoQA-10K](https://huggingface.co/datasets/liumindmind/NekoQA-10K) |
| **许可证** | Apache License 2.0（见同目录 `LICENSE`） |
| **格式** | JSON 数组，每条 `{instruction, output}`，共约 10066 条 |
| **用途** | 猫娘风格角色扮演 QA，用于 state tuning 的风格迁移任务 |

## 下载

完整数据集（`NekoQA-10K.json`，约 9.5MB）**不提交到本仓库**（体积考虑）。
本地训练/冒烟需要时，从 Hugging Face 下载：

```bash
# 方式一：huggingface-cli
huggingface-cli download liumindmind/NekoQA-10K \
  --repo-type dataset \
  --local-dir train_data/NekoQA_10k \
  --include "NekoQA-10K.json"

# 方式二：git lfs（需安装 git-lfs）
git clone https://huggingface.co/datasets/liumindmind/NekoQA-10K train_data/NekoQA_10k_hf
cp train_data/NekoQA_10k_hf/NekoQA-10K.json train_data/NekoQA_10k/
```

下载后放置到 `train_data/NekoQA_10k/NekoQA-10K.json`，`scripts/nekoqa_smoke.sh`
等脚本会自动识别。

## 本仓库内的衍生文件

| 文件 | 说明 | 许可 |
|---|---|---|
| `nekoqa_smoke_200.json` | 取 `NekoQA-10K.json` 前 200 条的**未修改**子集，用于冒烟测试自包含 | Apache-2.0（同源） |

> `nekoqa_smoke_200.json` 是原始数据集的子集（仅截断，无内容修改），
> 按 Apache-2.0 §4(b) 要求，仍适用原许可证。

## 许可证

本目录下的数据文件（含衍生子集）适用 **Apache License 2.0**，全文见 `LICENSE`。

```
Copyright 2024 liumindmind

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## 引用

使用本数据集时，请引用原始 Hugging Face 数据集：

```
@misc{nekoqa10k,
  author    = {liumindmind},
  title     = {NekoQA-10K},
  year      = {2024},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/liumindmind/NekoQA-10K}
}
```
