"""训练/推理样本格式模板。

问题根源(本单要修的三件事之一):
  训练样本无终止符 → 模型学不会"停";格式字面量("\n" 拼接的前缀构造)散落在
  data.py / cli.py / tests / experiments 多处,改一处漏一处,训练-推理编码路径
  容易不同构 → mask 错位。

本模块是全仓**唯一**允许持有 "\n"-拼接前缀构造的位置(验收 d)。
其余代码一律从 TaskTemplate 实例派生 prefix/target 字符串,禁止手写格式字面量。

约定:
  - prefix_template / target_template 是 str.format 模板,占位符由调用方传入。
  - encode_sample 分别独立 encode(prefix) 和 encode(target),再拼 + stop_token。
    禁止整段联合 encode(见 data.encode_sample)。
  - stop_token: token id,追加到 full_ids 末尾。默认 0 = World tokenizer 的
    eos(\x00),与 core.generate 的 `next_token == 0: break` 分支对齐。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskTemplate:
    """单个任务的 prefix/target 格式 + 终止符。

    prefix_template: 条件段(中文 prompt / 指令)。str.format 模板。
    target_template: 目标段(英文翻译 / 回答)。str.format 模板。
                     注意可含前导空白(如 NEKO_QA 的 " {a}")。
    stop_token: 追加到编码末尾的终止 token id。0 = World tokenizer eos。

    用法:
        t = P0_BARE
        prefix_text = t.format_prefix(cn="你好")      # "你好\\n"
        target_text = t.format_target(en="Hello")     # "Hello"
    """

    prefix_template: str
    target_template: str
    stop_token: int = 0

    def format_prefix(self, **kwargs) -> str:
        """渲染 prefix 字符串。占位符由 kwargs 提供(如 cn=..., en=...)。"""
        return self.prefix_template.format(**kwargs)

    def format_target(self, **kwargs) -> str:
        """渲染 target 字符串。占位符由 kwargs 提供。"""
        return self.target_template.format(**kwargs)


# ── 内置模板 ────────────────────────────────────────────────

P0_BARE = TaskTemplate(
    prefix_template="{cn}\n",
    target_template="{en}",
    stop_token=0,
)
"""P0 翻译任务裸格式: prefix="{中文}\\n", target="{英文}"。

训练/推理都用它:推理 prompt = format_prefix(cn=...) == "{中文}\\n",
与 ep04.npz 这个 P0 state 训练时一致。
"""

NEKO_QA = TaskTemplate(
    prefix_template="User: {q}\n\nAssistant:",
    target_template=" {a}",
    stop_token=0,
)
"""NekoQA 问答格式(预置,当前不接入任何管线)。

prefix 以 "Assistant:" 结尾(无尾随空白),target 以一个前导空格开始,
拼接后是 "Assistant: {回答}"。stop_token 同 P0。
"""
