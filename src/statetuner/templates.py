"""训练/推理样本格式模板。

本模块是全仓**唯一**允许持有 "\n"-拼接前缀构造的位置(验收 d)。
其余代码一律从 TaskTemplate 实例派生 prefix/target 字符串,禁止手写格式字面量。

分类学(Phase 3 Spec §1.1):
  - **训练/推理模板**(TaskTemplate 实例):决定样本结构与 prompt 包装。
    内置: qa / instruction / raw(raw 是推理原样,无 TaskTemplate)。
  - **think 档位**(推理侧,仅 reasoning 模型):off / fast / on,
    只影响 prompt 尾部渲染。见 inference.render_prompt。

历史更名(2026-07,Phase 3,Spec §1.2):
  - 旧数据集名命名模板 → QA(硬重命名,无 alias;通用格式不该用数据集命名)
  - 旧 reasoning 整包模板 → 删除,拆解为 qa + reasoning 方言(bos 前缀)+ think 档位,
    渲染逻辑落在 inference.render_prompt(单一事实源)。
  数据集名 NekoQA 照常存在(train_data、demo、历史 docs 不动)。

约定:
  - prefix_template / target_template 是 str.format 模板,占位符由调用方传入。
  - encode_template_sample 分别独立 encode(prefix) 和 encode(target),再拼 + stop_token。
    禁止整段联合 encode(见 data.encode_template_sample)。
  - stop_token: token id,追加到编码末尾。默认 0 = World tokenizer 的
    eos(\x00),与 core.generate 的 `next_token == 0: break` 分支对齐。
  - continuation_prefix_template: 多轮续传时,上一轮回答文本之后、本轮 User: 之前
    喂入的胶水模板。None = 该模板语义单任务,禁用多轮(instruction)。
    多轮本身是 InferenceEngine §2 的事,这里只提供数据。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TaskTemplate:
    """单个任务的 prefix/target 格式 + 终止符。

    prefix_template: 条件段(用户问题 / 指令)。str.format 模板。
    target_template: 目标段(回答)。str.format 模板。
                     注意可含前导空白(如 QA 的 " {a}")。
    stop_token: 追加到编码末尾的终止 token id。0 = World tokenizer eos。
    inference_stop_sequences: 推理时若模型开始生成下一轮角色标记,
                              应在该序列前停止。
    continuation_prefix_template: 多轮续传胶水模板,见模块 docstring。None=禁多轮。
    drop_input_when_empty: 当 kwargs 中 `input` 为空时,format_prefix 走降级
                           (instruction 专用:去掉 Input 段及前后空行)。

    用法:
        t = QA
        prefix_text = t.format_prefix(q="你好")        # "User: 你好\\n\\nAssistant:"
        target_text = t.format_target(a="喵~")         # " 喵~"
    """

    prefix_template: str
    target_template: str
    stop_token: int = 0
    inference_stop_sequences: tuple[str, ...] = ()
    continuation_prefix_template: Optional[str] = None
    drop_input_when_empty: bool = False

    def format_prefix(self, **kwargs) -> str:
        """渲染 prefix 字符串。占位符由 kwargs 提供(如 q=...)。

        drop_input_when_empty=True 时(instruction 模板),若 kwargs 中 `input`
        为空字符串,降级为不含 Input 段的格式(去掉 Input: 段及其前后空行)。
        """
        if self.drop_input_when_empty and not str(kwargs.get("input", "")).strip():
            # instruction 降级:Instruction: {instruction}\n\nResponse:
            # 只对 INSTRUCTION 形如 "Instruction: {instruction}\n\nInput: {input}\n\nResponse:" 有效。
            # 把 Input 段连同它前后的 \n\n 一起删掉。
            kwargs = {**kwargs, "input": ""}
            rendered = self.prefix_template.format(**kwargs)
            # 删除 "\n\nInput: \n\n" 这种空 Input 残段(空 input 占位渲染出的固定形态)。
            # 注意:这里只删字面 "\n\nInput: \n\n"(空 input 时 str.format 输出固定)。
            return rendered.replace("\n\nInput: \n\n", "\n\n")
        return self.prefix_template.format(**kwargs)

    def format_target(self, **kwargs) -> str:
        """渲染 target 字符串。占位符由 kwargs 提供。"""
        return self.target_template.format(**kwargs)


# ── 内置模板 ────────────────────────────────────────────────

QA = TaskTemplate(
    prefix_template="User: {q}\n\nAssistant:",
    target_template=" {a}",
    stop_token=0,
    inference_stop_sequences=("\nUser:",),
    continuation_prefix_template="\n\nUser: {q}\n\nAssistant:",
)
"""QA 问答格式(角色扮演 / QA 任务)。

prefix 以 "Assistant:" 结尾(无尾随空白),target 以一个前导空格开始,
拼接后是 "Assistant: {回答}"。stop_token = 0(World tokenizer eos)。

continuation_prefix_template = "\\n\\nUser: {q}\\n\\nAssistant:" ——
多轮续传时(Phase 3 §2 InferenceEngine 多轮改造),上一轮回答之后、本轮 User:
之前喂入的胶水。首轮用 prefix_template,后续轮用 continuation_prefix_template。
本期(InferenceEngine §2 之前)该字段只作为数据契约存在,尚无消费方。
"""


INSTRUCTION = TaskTemplate(
    prefix_template="Instruction: {instruction}\n\nInput: {input}\n\nResponse:",
    target_template=" {a}",
    stop_token=0,
    inference_stop_sequences=("\nInstruction:",),
    continuation_prefix_template=None,
    drop_input_when_empty=True,
)
"""指令问答格式(对齐官方 Instruction/Input/Response 模式)。

语义上是单任务,continuation_prefix_template=None(多轮禁用)。

drop_input_when_empty=True: input 为空时 format_prefix 自动降级为
"Instruction: {instruction}\\n\\nResponse:"(去掉 Input 段及其前后空行),
避免出现 "\\n\\n\\n" 残留空行(验收 d)。导入器与推理共用同一降级规则。
"""
