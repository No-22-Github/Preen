"""推理 golden 测试(快,~20s,CI 友好)。

用 P0 验收通过的 state(ep04)做贪心解码,对比 golden 快照。
核心断言:注入 state 后中文输入产出英文翻译(翻译行为发生)。
条件性:英文/乱码输入不触发反向翻译。
基线:无 state 时原始模型不翻译。

这些测试固化 P0 的核心验收结论(命题③④⑤)。

prompt 一律从 templates.P0_BARE 派生(与训练 encode_sample 同源),
禁止在此手写 f"{...}\\n" 格式字面量(验收 d)。
"""
import json

import pytest

from conftest import GOLDEN_DIR
from statetuner.core import generate
from statetuner.templates import P0_BARE


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
        out = first_line(generate(model, tok, P0_BARE.format_prefix(cn=cn), state=state_file, max_tokens=70))
        assert english_ratio(out) > 0.5, (
            f"[{golden_file}] {cn}\n输出非英文: {out!r}"
        )


@pytest.mark.parametrize("golden_file", ["translate_train.json", "translate_test.json"])
def test_golden_verbatim(app, state_file, golden_file):
    """golden 逐字零回退(验收 a):输出必须与 golden 快照的 "out" 字段逐字相等。

    prompt 编码路径改为从 P0_BARE 派生,但 P0_BARE.format_prefix(cn=cn) 的字符串
    内容与旧 f"{cn}\\n" 完全相同,且推理 state(ep04.npz)未变 → 输出必然逐字一致。
    若此断言失败,说明编码路径改动引入了非预期差异(mask/token 边界错位)。
    """
    model, tok = app
    golden = load_golden(golden_file)
    mismatches = []
    for cn, entry in golden.items():
        out = first_line(generate(model, tok, P0_BARE.format_prefix(cn=cn), state=state_file, max_tokens=70))
        if out != entry["out"]:
            mismatches.append((cn, entry["out"], out))
    assert not mismatches, (
        f"[{golden_file}] golden 逐字回退, {len(mismatches)}/{len(golden)} 条不一致:\n"
        + "\n".join(
            f"  CN: {cn}\n    golden: {g!r}\n    now:    {n!r}"
            for cn, g, n in mismatches
        )
    )


def test_translate_test_set_majority_english(app, state_file):
    """测试集(未训练)翻译:多数条目应是英文翻译尝试(命题③核心)。"""
    model, tok = app
    golden = load_golden("translate_test.json")
    en_count = sum(
        1
        for cn in golden
        if english_ratio(first_line(generate(model, tok, P0_BARE.format_prefix(cn=cn), state=state_file, max_tokens=70))) > 0.5
    )
    assert en_count >= 4, f"测试集翻译: 仅 {en_count}/{len(golden)} 英文, 期望 ≥4"


# ── 条件性对照 ─────────────────────────────────────────────

def test_conditional_english_no_chinese(app, state_file):
    """英文输入不应反向翻译成中文(state 方向是 中→英)。"""
    model, tok = app
    golden = load_golden("conditional.json")["english"]
    for en_in in golden:
        out = first_line(generate(model, tok, P0_BARE.format_prefix(cn=en_in), state=state_file, max_tokens=40))
        cn_chars = [c for c in out if c.isalpha() and ord(c) > 127]
        assert len(cn_chars) < 5, f"英文输入 {en_in!r} 意外产出中文: {out!r}"


def test_conditional_junk_no_chinese(app, state_file):
    """乱码/数字输入:不产出中文(方向不应学反)。"""
    model, tok = app
    golden = load_golden("conditional.json")["junk"]
    for junk_in in golden:
        out = first_line(generate(model, tok, P0_BARE.format_prefix(cn=junk_in), state=state_file, max_tokens=40))
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
        out = first_line(generate(model, tok, P0_BARE.format_prefix(cn=cn), state=None, max_tokens=50))
        assert english_ratio(out) < 0.5, (
            f"[无state] {cn}\n意外输出英文: {out!r}\n翻译能力应来自 state"
        )
