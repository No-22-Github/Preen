"""think=on 输出的拆分逻辑(UI 中立层,T1)。

此前拆分逻辑有两处不一致的事实源:
  - chat._display_text 用 rfind("</think>")(取最后一个闭合,容错中间出现标签)
  - console.split_thinking 用 find("</think>")(取第一个闭合)
SwiftUI/SidecarClient 要再写一遍就是第三个。本模块是单一事实源,
chat/console/serve 都从这里派生。

拆分规则(对齐 reasoning 模型品类惯例):
  - 找到 </think> → 前段(清洗)= thinking, 后段 = answer
  - 找不到(max_tokens 截断等)→ 已生成内容当 thinking(它确实是未写完的思考),
    answer=""。展示层据此 dim 显示思考 + 标注被截断;history 层据此存空 answer。
  - </think> 紧开头(空思考)→ thinking="", answer=剩余

find vs rfind:用 **find**(取第一个闭合标签)。reasoning 模型的 think 段
不会在内部合法出现 </think>(那是闭合标签),第一个就是真正的边界。
"""
from __future__ import annotations

THINK_CLOSE = "</think>"


def split_thinking(raw: str) -> tuple[str, str]:
    """把 think=on 的原始输出拆成 (thinking, answer)。

    thinking 已清洗(strip + 去掉开标签闭合残余的 '>');answer 原样,
    由调用方按需再做模板相关清洗(chat._display_text 的 lstrip('\\n'))。

    仅对 think=on 有意义;off/fast 调用方不会进来。
    """
    idx = raw.find(THINK_CLOSE)
    if idx < 0:
        # 未闭合:已生成内容是未写完的思考,answer 为空。
        return _clean_thinking(raw), ""
    return _clean_thinking(raw[:idx]), raw[idx + len(THINK_CLOSE):]


def classify_phase(accumulated: str) -> str:
    """根据已累积的文本判断当前 phase(T1: text_chunk.phase 字段)。

    返回 "think"(还在思考段,未越过 </think>)或 "answer"(已闭合)。
    用于流式事件标记 phase,UI 据此 dim/正常渲染增量。
    """
    return "answer" if THINK_CLOSE in accumulated else "think"


def _clean_thinking(text: str) -> str:
    """清洗 think 段文本:strip + 去掉开头的残余 ``>``。

    think=on 的 prompt 以 ``<think`` 结尾,模型续写时把它补全成 ``<think>``,
    所以 raw 输出常以 ``>\\n`` 开头(标签闭合的机械副产品)。这个 ``>`` 不是
    思考内容,显示出来突兀,清掉。只删开头 ``>`` + 紧跟换行的固定形态,
    不影响思考正文里合法的 ``>``。
    """
    cleaned = text.strip()
    if cleaned.startswith(">\n"):
        cleaned = cleaned[2:].lstrip("\n")
    elif cleaned == ">":
        cleaned = ""
    return cleaned
