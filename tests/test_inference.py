"""推理 golden 测试(快,~20s,CI 友好)。

用 P0 验收通过的 state(ep04)做贪心解码,对比 golden 快照。
核心断言:注入 state 后中文输入产出英文翻译(翻译行为发生)。
条件性:英文/乱码输入不触发反向翻译。
基线:无 state 时原始模型不翻译。

这些测试固化 P0 的核心验收结论(命题③④⑤)。
"""
import json

import pytest

from conftest import GOLDEN_DIR
from statetuner.core import generate


def load_golden(name):
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


def first_line(s):
    return s.split("\n")[0].strip() if "\n" in s else s.strip()


def english_ratio(text):
    """文本中 ASCII 字母占比,判断是否英文输出。"""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) < 128) / len(letters)


# ── 翻译 golden 测试 ────────────────────────────────────────

@pytest.mark.parametrize("golden_file", ["translate_train.json", "translate_test.json"])
def test_translate_produces_english(app, state_file, golden_file):
    """注入训练 state,中文输入应产出英文(翻译行为发生)。"""
    model, tok = app
    golden = load_golden(golden_file)
    for cn in golden:
        out = first_line(generate(model, tok, f"{cn}\n", state=state_file, max_tokens=70))
        assert english_ratio(out) > 0.5, (
            f"[{golden_file}] {cn}\n输出非英文: {out!r}"
        )


def test_translate_test_set_majority_english(app, state_file):
    """测试集(未训练)翻译:多数条目应是英文翻译尝试(命题③核心)。"""
    model, tok = app
    golden = load_golden("translate_test.json")
    en_count = sum(
        1
        for cn in golden
        if english_ratio(first_line(generate(model, tok, f"{cn}\n", state=state_file, max_tokens=70))) > 0.5
    )
    assert en_count >= 4, f"测试集翻译: 仅 {en_count}/{len(golden)} 英文, 期望 ≥4"


# ── 条件性对照 ─────────────────────────────────────────────

def test_conditional_english_no_chinese(app, state_file):
    """英文输入不应反向翻译成中文(state 方向是 中→英)。"""
    model, tok = app
    golden = load_golden("conditional.json")["english"]
    for en_in in golden:
        out = first_line(generate(model, tok, f"{en_in}\n", state=state_file, max_tokens=40))
        cn_chars = [c for c in out if c.isalpha() and ord(c) > 127]
        assert len(cn_chars) < 5, f"英文输入 {en_in!r} 意外产出中文: {out!r}"


def test_conditional_junk_no_chinese(app, state_file):
    """乱码/数字输入:不产出中文(方向不应学反)。"""
    model, tok = app
    golden = load_golden("conditional.json")["junk"]
    for junk_in in golden:
        out = first_line(generate(model, tok, f"{junk_in}\n", state=state_file, max_tokens=40))
        cn_chars = [c for c in out if c.isalpha() and ord(c) > 127]
        assert len(cn_chars) < 5, f"乱码输入 {junk_in!r} 意外产出中文: {out!r}"


# ── 基线对照 ───────────────────────────────────────────────

def test_baseline_no_state_no_translate(app):
    """无 state 注入时,原始模型对中文不翻译成英文。

    证明翻译能力来自训练的 state,而非基座模型本身。
    """
    model, tok = app
    golden = load_golden("baseline_no_state.json")
    for cn in golden:
        out = first_line(generate(model, tok, f"{cn}\n", state=None, max_tokens=50))
        assert english_ratio(out) < 0.5, (
            f"[无state] {cn}\n意外输出英文: {out!r}\n翻译能力应来自 state"
        )
