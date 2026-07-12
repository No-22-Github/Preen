"""eval 用例测试：service 层编排 + 模板同源回归（FakeEngine，不加载真实模型）。

覆盖：
  ① 回归：prompt 渲染与 stop sequences 使用同一 template（历史 bug 防线）。
  ② service 层单测：数据加载失败、limit 生效、结果结构。
"""
import json

import pytest

from statetuner.inference import GenerationConfig, GenerationResult, with_template_stops
from statetuner.service import (
    DEFAULT_EVAL_QUESTIONS,
    EvaluationRequest,
    run_evaluation,
    validate_evaluation_request,
)
from statetuner.templates import QA as NEKO_QA  # tests 局部别名


class _RecordingEngine:
    """记录每次 generate 收到的 prompt 与 config，返回固定 GenerationResult。

    用于断言 prompt 渲染与 stop sequences 是否同源（历史 bug：render_prompt
    硬编码 nekoqa，与 --template g1g 的 stops 分裂）。
    """

    def __init__(self):
        self.calls: list[tuple[str, GenerationConfig]] = []

    def generate(self, prompt, *, state=None, config=None, on_text=None):
        self.calls.append((prompt, config))
        return GenerationResult(
            text="reply",
            display_token_ids=[1],
            stop_reason="eos",
            elapsed=0.0,
            used_state=state is not None,
            config=config,
        )


# ── ① 回归：prompt 渲染与 stops 同源 ────────────────────────

def test_eval_nekoqa_prompt_and_stops_share_template():
    """qa：render_prompt 的格式与 config.stop_sequences 都来自 QA 模板（旧 NEKO_QA）。"""
    engine = _RecordingEngine()
    cfg = with_template_stops(GenerationConfig(max_tokens=5), "qa")
    run_evaluation(
        EvaluationRequest(
            engine=engine, state="s.npz", template="qa", config=cfg, limit=1
        )
    )
    prompt, used_cfg = engine.calls[0]
    assert prompt == NEKO_QA.format_prefix(q=DEFAULT_EVAL_QUESTIONS[0][0])
    assert used_cfg.stop_sequences == NEKO_QA.inference_stop_sequences


def test_eval_g1g_prompt_and_stops_share_template():
    """qa + reasoning + think=fast：等价旧 G1G 渲染（防 render_prompt 硬编码的 bug）。

    Spec §1.2：旧 G1G 模板已拆解为 qa + reasoning 方言 + think 档位。
    stops 来自 QA.inference_stop_sequences（reasoning 不改 stop 边界）。
    """
    engine = _RecordingEngine()
    cfg = with_template_stops(GenerationConfig(max_tokens=5), "qa")
    run_evaluation(
        EvaluationRequest(
            engine=engine, state="s.npz", template="qa", config=cfg, limit=1,
            reasoning=True, think="fast",
        )
    )
    prompt, used_cfg = engine.calls[0]
    # 等价旧 G1G.format_prefix：bos 前缀 + qa prefix + 空 think 标签
    assert prompt.startswith("<|rwkv_tokenizer_end_of_text|>")
    assert prompt.endswith("Assistant: <think>\n</think>")
    assert used_cfg.stop_sequences == NEKO_QA.inference_stop_sequences


# ── ② service 层单测 ────────────────────────────────────────

def test_run_evaluation_data_load_failure_raises(tmp_path):
    """评估数据格式错误时，load_qa_pairs 抛 ValueError（不经 CLI 转译）。"""
    bad = tmp_path / "bad.json"
    bad.write_text('[{"instruction": ""}]', encoding="utf-8")
    engine = _RecordingEngine()
    cfg = with_template_stops(GenerationConfig(max_tokens=5), "qa")
    request = EvaluationRequest(
        engine=engine, state="s.npz", template="qa", config=cfg, data=bad, limit=3
    )
    with pytest.raises(ValueError, match="非空字符串"):
        run_evaluation(request)
    assert engine.calls == []  # 加载失败 → 不生成


def test_run_evaluation_limit_truncates(tmp_path):
    """limit 截断：数据有 5 条，limit=2 只生成 2 条。"""
    items = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(5)]
    data = tmp_path / "data.json"
    data.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    engine = _RecordingEngine()
    cfg = with_template_stops(GenerationConfig(max_tokens=5), "qa")
    result = run_evaluation(
        EvaluationRequest(
            engine=engine, state="s.npz", template="qa",
            config=cfg, data=data, limit=2,
        )
    )
    assert len(result.items) == 2
    assert [item.index for item in result.items] == [1, 2]
    assert [item.question for item in result.items] == ["Q0", "Q1"]


def test_run_evaluation_uses_default_questions_without_data():
    """无 --data 时使用内置示例（DEFAULT_EVAL_QUESTIONS）。"""
    engine = _RecordingEngine()
    cfg = with_template_stops(GenerationConfig(max_tokens=5), "qa")
    result = run_evaluation(
        EvaluationRequest(
            engine=engine, state="s.npz", template="qa",
            config=cfg, limit=10,
        )
    )
    assert len(result.items) == len(DEFAULT_EVAL_QUESTIONS)
    assert [item.question for item in result.items] == [q for q, _ in DEFAULT_EVAL_QUESTIONS]


def test_run_evaluation_result_structure():
    """结果结构：index 从 1 起，text 已 strip，generation 字段完整。"""
    engine = _RecordingEngine()
    cfg = with_template_stops(GenerationConfig(max_tokens=5), "qa")
    result = run_evaluation(
        EvaluationRequest(
            engine=engine, state="s.npz", template="qa", config=cfg, limit=2
        )
    )
    payload = result.to_dict()
    assert "results" in payload
    first = payload["results"][0]
    assert first["index"] == 1
    assert first["text"] == "reply"
    assert first["stop_reason"] == "eos"
    assert "display_token_ids" in first


def test_validate_evaluation_request_rejects_bad_template(tmp_path):
    engine = _RecordingEngine()
    cfg = GenerationConfig()
    with pytest.raises(ValueError, match="qa / instruction / raw"):
        validate_evaluation_request(
            EvaluationRequest(
                # 用已废弃的旧模板名触发拒绝（nekoqa 在新世界不再合法）
                engine=engine, state="s.npz", template="nekoqa", config=cfg
            )
        )


def test_validate_evaluation_request_rejects_missing_state():
    engine = _RecordingEngine()
    cfg = GenerationConfig()
    with pytest.raises(ValueError, match="state"):
        validate_evaluation_request(
            EvaluationRequest(
                engine=engine, state=None, template="qa", config=cfg
            )
        )
