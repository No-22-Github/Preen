"""推理 golden 测试(快,~20s,CI 友好)。

用 NekoQA 基准 state(nekoqa_04b_s42,0.4B + smoke_200 × 3epoch + seed 42)做贪心解码,
对比 golden 快照。核心守护:
  - golden 逐字零回退:注入 state 后输出与快照逐字相等(编码路径未被改动)。
  - state 生效:有 state vs 无 state 输出不同(行为来自 state,非基座)。
  - eos 终止:注入 state 后生成能自发终止(不顶到 max_tokens 上限)。

prompt 一律从 templates.NEKO_QA 派生(与训练 load_qa_dataset 同源),
禁止在此手写 f"{...}\\n" 格式字面量(验收 d)。
"""
import json

from conftest import GOLDEN_DIR
from statetuner.core import generate
from statetuner.templates import NEKO_QA


def load_golden(name):
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


def generate_with_state(model, tok, prompt, state_file, max_tokens=70):
    """从 NEKO_QA 模板派生 prompt,注入 state 生成。"""
    wrapped = NEKO_QA.format_prefix(q=prompt)
    return generate(model, tok, wrapped, state=state_file, max_tokens=max_tokens)


# ── golden 逐字零回退(核心)──────────────────────────────

def test_nekoqa_golden_verbatim(app, state_file):
    """注入 state 后输出必须与 golden 快照的 "out" 字段逐字一致。

    state 文件(nekoqa_04b_s42.npz)未变 → 输出必然逐字一致。
    若此断言失败,说明 generate / state 加载 / 编码路径改动引入了非预期差异。
    """
    model, tok = app
    golden = load_golden("nekoqa_generate.json")
    mismatches = []
    for prompt, entry in golden.items():
        out = generate_with_state(model, tok, prompt, state_file)
        if out != entry["out"]:
            mismatches.append((prompt, entry["out"], out))
    assert not mismatches, (
        f"golden 逐字回退, {len(mismatches)}/{len(golden)} 条不一致:\n"
        + "\n".join(
            f"  Q: {q}\n    golden: {g!r}\n    now:    {n!r}"
            for q, g, n in mismatches
        )
    )


# ── state 生效证明 ─────────────────────────────────────────

def test_nekoqa_state_changes_output(app, state_file):
    """有 state vs 无 state 输出应不同,证明行为来自 state 而非基座。

    不硬编码"猫娘"特征(避免脆性),只断言"有 state 时输出与无 state 时不同"。
    用 baseline golden 的 prompt 集逐条对比。
    """
    model, tok = app
    golden = load_golden("nekoqa_baseline.json")
    same_count = 0
    for prompt in golden:
        out_state = generate_with_state(model, tok, prompt, state_file, max_tokens=40)
        out_no_state = generate_with_state(model, tok, prompt, None, max_tokens=40)
        if out_state == out_no_state:
            same_count += 1
    # 允许个别 prompt 碰巧相同(基座在短输出上可能重复),但不应全部相同
    assert same_count < len(golden), (
        f"有/无 state 输出全部相同({same_count}/{len(golden)}),state 未生效?"
    )


# ── eos 自发终止 ───────────────────────────────────────────

def test_nekoqa_eos_termination(app, state_file):
    """注入 state 后生成应能自发终止(遇 eos 停下),不顶到 max_tokens 上限。

    generate 遇 token 0(eos)会 break。训练时追加了 stop_token,
    state 学会了"停"。若输出长度 == max_tokens,说明没自发终止
    (可能是退化重复,但 0.4B 基座有固有重复缺陷,放宽:多数条目终止即可)。
    """
    model, tok = app
    golden = load_golden("nekoqa_generate.json")
    max_tokens = 70
    terminated = 0
    for prompt, entry in golden.items():
        out = generate_with_state(model, tok, prompt, state_file, max_tokens=max_tokens)
        # 输出长度 < max_tokens 视为自发终止(eos 提前停)
        # (generate 的 len 是 token 数,这里用字符长度近似:终止的通常 < max_tokens 字符)
        if len(out) < max_tokens:
            terminated += 1
    # 多数条目应自发终止(允许 0.4B 重复缺陷导致个别不终止)
    assert terminated >= len(golden) - 1, (
        f"仅 {terminated}/{len(golden)} 条自发终止,期望 ≥{len(golden) - 1}"
    )
