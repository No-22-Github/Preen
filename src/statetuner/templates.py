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
    # 推理时若模型开始生成下一轮角色标记，应在该序列前停止。
    inference_stop_sequences: tuple[str, ...] = ()

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
    inference_stop_sequences=("\nUser:",),
)
"""NekoQA 问答格式(角色扮演 / QA 任务)。

prefix 以 "Assistant:" 结尾(无尾随空白),target 以一个前导空格开始,
拼接后是 "Assistant: {回答}"。stop_token = 0(World tokenizer eos)。
"""


G1G = TaskTemplate(
    prefix_template="<|rwkv_tokenizer_end_of_text|>User: {q}\n\nAssistant: <think>\n</think>",
    target_template=" {a}",
    stop_token=0,
    inference_stop_sequences=("\nUser:", "\nSystem:"),
)
"""RWKV7-G1 原生对话格式(不带思考)。

对齐 g1g 官方 chat_template 的 enable_thinking=False 渲染结果(已实测 token 序列
一致,见 tokenizer_config.json)。与 NEKO_QA 的关键差异:
  - 开头 <|rwkv_tokenizer_end_of_text|>(= token 0 / bos):RWKV 每轮对话都以它
    起始,缺它 state 初始化偏离训练分布。
  - 结尾 Assistant: <think>\\n</think>:空 think 标签告诉模型"思考段为空,直接答题"。
    缺它模型续写时不知该思考还是直答,易跑飞(实测:raw 格式跑满 120 token 自报
    "ChatGPT";本格式 39 token 正常 eos 自然回答)。

stop_token = 0(World tokenizer eos)。inference 还要在生成出下一轮角色标记
(\\nUser: / \\nSystem:)前停下。模型输出开头常带一个 \\n(</think> 后的自然换行),
显示侧由 chat 层 lstrip 处理。
"""
