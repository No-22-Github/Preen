"""训练/推理样本格式模板。

本模块是全仓**唯一**允许持有 "\n"-拼接前缀构造的位置(验收 d)。
其余代码一律从 TaskTemplate 实例派生 prefix/target 字符串,禁止手写格式字面量。

约定:
  - prefix_template / target_template 是 str.format 模板,占位符由调用方传入。
  - encode_template_sample 分别独立 encode(prefix) 和 encode(target),再拼 + stop_token。
    禁止整段联合 encode(见 data.encode_template_sample)。
  - stop_token: token id,追加到编码末尾。默认 0 = World tokenizer 的
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
        t = NEKO_QA
        prefix_text = t.format_prefix(q="你好")        # "User: 你好\\n\\nAssistant:"
        target_text = t.format_target(a="喵~")         # " 喵~"
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

NEKO_QA = TaskTemplate(
    prefix_template="User: {q}\n\nAssistant:",
    target_template=" {a}",
    stop_token=0,
)
"""NekoQA 问答格式(角色扮演 / QA 任务)。

prefix 以 "Assistant:" 结尾(无尾随空白),target 以一个前导空格开始,
拼接后是 "Assistant: {回答}"。stop_token = 0(World tokenizer eos)。
"""
