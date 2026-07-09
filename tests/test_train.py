"""训练行为断言测试(慢,~4min,标 @pytest.mark.slow)。

不做 bit-golden(MLX GPU ULP 不确定),改为断言行为指标:
  - 梯度冒烟:24层 grad 全非零(patch 生效、state 进了计算图)
  - 过拟合:10条 loss < 0.5(管线正确)
  - 全量收敛:loss < 1.0 + state std 合理 + 能翻译

默认不跑(--slow 开启)。

另外含不依赖模型的快测试:数据编码边界、事件序列化、lr 调度。
"""
import math
import random
from pathlib import Path

import numpy as np
import pytest

from conftest import DATA_PATH, MODEL_PATH

pytestmark_for_slow = pytest.mark.slow


# ── 不依赖模型的快测试 ─────────────────────────────────────

def test_data_extract_cn_en():
    """extract_cn_en 兼容 User/Assistant 模板和裸格式。"""
    from statetuner.data import extract_cn_en

    # 模板格式
    cn, en = extract_cn_en({"text": "User: 你好\n\nAssistant: Hello"})
    assert cn == "你好"
    assert en == "Hello"
    # 裸格式
    cn, en = extract_cn_en({"text": "你好\nHello"})
    assert cn == "你好"
    assert en == "Hello"
    # cn/en 字段
    cn, en = extract_cn_en({"cn": "你好", "en": "Hello"})
    assert (cn, en) == ("你好", "Hello")


def test_train_test_split_reproducible():
    """train_test_split 相同 seed 产出相同划分。"""
    from statetuner.data import Sample, train_test_split

    samples = [
        Sample([1, 2], [2, 3], [0, 1], f"cn{i}", f"en{i}", 1) for i in range(20)
    ]
    tr1, te1 = train_test_split(samples, test_ratio=0.2, seed=42)
    tr2, te2 = train_test_split(samples, test_ratio=0.2, seed=42)
    assert [s.cn for s in tr1] == [s.cn for s in tr2]
    assert [s.cn for s in te1] == [s.cn for s in te2]
    assert len(te1) == 4
    assert len(tr1) == 16


def test_event_serialization():
    """事件 JSON 序列化,字段为原生类型。"""
    import json

    from statetuner import events

    em = events.EventEmitter(quiet=True)
    em.emit(events.start({"lr": 0.01, "epochs": 20}))
    em.emit(events.epoch_end(0, loss=1.5, state_std=0.2, lr=0.01))
    em.emit(events.std_warning(1, 1.2, 1.0))
    em.emit(events.early_stop(3, best=0.8, held_out_loss=0.9))

    assert len(em.events) == 4
    for ev in em.events:
        s = json.dumps(ev)  # 不抛异常 = 全原生类型
        assert "type" in ev
        assert "timestamp" in ev
    assert em.events[0]["type"] == "start"
    assert em.events[0]["config"]["lr"] == 0.01
    assert em.events[2]["type"] == "std_warning"
    assert em.events[3]["type"] == "early_stop"


def test_cosine_lr_schedule():
    """cosine lr: warmup 线性升 → cosine 衰减到 floor。"""
    from statetuner.train import TrainConfig, cosine_lr

    cfg = TrainConfig(lr=0.01, lr_floor=0.0001, warmup=10, epochs=10)
    total = cfg.total_steps(100)  # 1000

    # warmup: step 0 → ~0, step 9 → ~peak
    assert cosine_lr(0, total, cfg) < cosine_lr(5, total, cfg)
    assert cosine_lr(9, total, cfg) == pytest.approx(0.01, rel=0.1)
    # peak 后下降
    lr_mid = cosine_lr(100, total, cfg)
    lr_late = cosine_lr(500, total, cfg)
    assert lr_late < lr_mid < cfg.lr
    # 最终接近 floor
    assert cosine_lr(total - 1, total, cfg) < cfg.lr_floor + (cfg.lr - cfg.lr_floor) * 0.05


# ── 依赖模型的训练测试(slow)──────────────────────────────

@pytest.fixture(scope="module")
def model_tokenizer():
    """训练用模型(patch ops 路径)。"""
    if not (MODEL_PATH / "model.safetensors").exists():
        pytest.skip(f"模型不存在: {MODEL_PATH}")
    from statetuner.core import load_model

    model, tok = load_model(str(MODEL_PATH), patch=True)
    model.freeze()
    return model, tok


@pytest.mark.slow
def test_smoke_gradient(model_tokenizer):
    """第一级:梯度冒烟。24层 state grad 全非零、无 NaN。"""
    import mlx.core as mx
    import mlx.nn as nn

    from statetuner.core import compute_loss, forward_with_state, make_state_params
    from statetuner.data import load_dataset

    model, tok = model_tokenizer
    samples = load_dataset(str(DATA_PATH / "data_100.jsonl"), tok, max_len=64)
    s = samples[0]
    sp = make_state_params(model, dtype=mx.float32)
    inp = mx.array([s.input_ids])
    lab = mx.array([s.labels])
    msk = mx.array([[float(x) for x in s.mask]], dtype=mx.float32)

    loss_fn = compute_loss(model, (inp, lab, msk), sp)
    loss, grads = mx.value_and_grad(loss_fn)(*[sp[i] for i in range(len(sp))])
    mx.eval(loss, grads)

    for i, g in enumerate(grads):
        assert not bool(mx.any(mx.isnan(g))), f"layer {i} NaN"
        assert float(mx.abs(g).sum()) > 1e-12, f"layer {i} 零梯度 (patch 未生效?)"


@pytest.mark.slow
def test_overfit(model_tokenizer):
    """第二级:10条样本过拟合(lr=1.0, 200步),loss 应 < 0.5。"""
    import mlx.core as mx
    import mlx.optimizers as optim

    from statetuner.core import compute_loss, make_state_params
    from statetuner.data import load_dataset

    model, tok = model_tokenizer
    samples = load_dataset(str(DATA_PATH / "data_100.jsonl"), tok, max_len=64)[:10]
    sp = make_state_params(model, dtype=mx.float32)
    opt = optim.Adam(learning_rate=1.0, betas=[0.9, 0.99], eps=1e-8)
    random.seed(42)
    best = 1e9
    for step in range(200):
        s = samples[step % 10]
        inp = mx.array([s.input_ids])
        lab = mx.array([s.labels])
        msk = mx.array([[float(x) for x in s.mask]], dtype=mx.float32)
        if step < 10:
            lr = 1.0 * (step + 1) / 10
        else:
            prog = (step - 10) / max(1, 200 - 10)
            lr = 0.01 + (1.0 - 0.01) * 0.5 * (1 + math.cos(math.pi * prog))
        opt.learning_rate = lr
        loss_fn = compute_loss(model, (inp, lab, msk), sp)
        loss, grads = mx.value_and_grad(loss_fn)(*[sp[i] for i in range(len(sp))])
        grads = {k: mx.clip(g, -1.0, 1.0) for k, g in grads.items()}
        sp = opt.apply_gradients(grads, sp)
        mx.eval(sp, loss)
        best = min(best, float(loss))
    assert best < 0.5, f"过拟合失败: 最低 loss {best:.4f} >= 0.5"


@pytest.mark.slow
def test_full_train_translates(model_tokenizer):
    """全量训练(lr=0.01, 6 epoch)+ 翻译验证。

    三项断言:loss 收敛 / state std 合理 / 产出英文翻译。
    用产品化的 Trainer 跑(验证 train.py 本身)。
    """
    import mlx.core as mx

    from statetuner import events
    from statetuner.data import load_dataset
    from statetuner.train import Trainer, TrainConfig

    model, tok = model_tokenizer
    samples = load_dataset(str(DATA_PATH / "data_100.jsonl"), tok, max_len=128)

    cfg = TrainConfig(lr=0.01, lr_floor=0.0001, warmup=10, epochs=6, early_stop=False)
    em = events.EventEmitter(quiet=True)
    result = Trainer(model, cfg, em).train(samples)

    # 1. loss 收敛
    assert result.final_loss < 1.0, f"loss 未收敛: {result.final_loss:.4f}"
    # 2. state std 合理(不爆炸)
    assert 0.05 < result.final_state_std < 1.0, (
        f"state std 异常: {result.final_state_std:.4f} (合理 0.05~1.0)"
    )
    # 3. 产出英文翻译
    from statetuner.core import generate
    from statetuner.train import save_state_npz

    cn = samples[0].cn
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_npz = f.name
    try:
        save_state_npz(result.states, tmp_npz)
        out = generate(model, tok, f"{cn}\n", state=tmp_npz, max_tokens=50)
    finally:
        Path(tmp_npz).unlink(missing_ok=True)
    out = out.split("\n")[0].strip() if "\n" in out else out.strip()
    letters = [c for c in out if c.isalpha()]
    en_ratio = sum(1 for c in letters if ord(c) < 128) / max(1, len(letters))
    assert en_ratio > 0.5, f"未产出英文翻译 (en_ratio={en_ratio:.2f}): {out!r}"

    # 4. 训练事件序列完整
    types = [e["type"] for e in em.events]
    assert "start" in types
    assert "epoch_end" in types
    assert "final" in types
