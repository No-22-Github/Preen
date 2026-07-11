"""
训练行为断言测试 (慢, ~5min, 标 @pytest.mark.slow)。

不做 bit-golden (MLX GPU ULP 不确定), 改为断言行为指标:
  - 梯度冒烟: 24层 grad 全非零
  - 过拟合: 10条 loss < 0.5
  - 全量训练收敛: loss < 1.0 + state std 在合理范围 + 能翻译

默认不跑 (conftest 的 pytest_collection_modifyitems),
pytest --slow 显式开启。
"""
import math
import random
import sys
from pathlib import Path

import pytest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from conftest import MODEL_PATH, DATA_PATH
from state_tuner import patch_rwkv7_for_train, make_state_params, forward_with_state, cosine_lr
from data_v2 import prepare_samples_v2
from statetuner.templates import P0_BARE
from state_tuner import generate

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model_tokenizer():
    """训练用模型 (需 patch ops 路径)。独立于推理测试的 app fixture。"""
    if not (MODEL_PATH / "model.safetensors").exists():
        pytest.skip(f"模型不存在: {MODEL_PATH}")
    patch_rwkv7_for_train()
    from mlx_lm import load
    model, tok = load(str(MODEL_PATH), tokenizer_config={"trust_remote_code": True})
    model.freeze()
    return model, tok


def _masked_loss(model, inp, lab, msk, sd):
    """masked 交叉熵 loss (closure)。"""
    logits = forward_with_state(model, inp, sd, 1)
    lp = nn.log_softmax(logits, -1)
    g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
    return (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)


def _run_train(model, samples, lr_peak, lr_floor, epochs, seed=42):
    """跑一次训练,返回 (final_state_params, last_epoch_loss, state_std)。

    通用训练循环,供各测试复用。lr 走 cosine + warmup 10。
    """
    random.seed(seed)
    sp = make_state_params(model, dtype=mx.float32)
    opt = optim.Adam(learning_rate=lr_peak, betas=[0.9, 0.99], eps=1e-8)
    cfg = type("C", (), {"lr_peak": lr_peak, "lr_floor": lr_floor, "warmup": 10})()
    total = epochs * len(samples)
    step = 0
    last_loss = 0.0
    for _ in range(epochs):
        order = list(range(len(samples)))
        random.shuffle(order)
        losses = []
        for si in order:
            s = samples[si]
            inp = mx.array([s[0]]); lab = mx.array([s[1]])
            msk = mx.array([[float(x) for x in s[2]]], dtype=mx.float32)
            opt.learning_rate = cosine_lr(step, total, cfg)
            loss, grads = mx.value_and_grad(lambda sd: _masked_loss(model, inp, lab, msk, sd))(sp)
            grads = {k: mx.clip(g, -1.0, 1.0) for k, g in grads.items()}
            sp = opt.apply_gradients(grads, sp)
            mx.eval(sp, loss)
            losses.append(float(loss))
            step += 1
        last_loss = sum(losses) / len(losses)
    sstd = float(np.mean([np.array(sp[i]).std() for i in sp]))
    return sp, last_loss, sstd


def test_smoke_gradient(model_tokenizer):
    """第一级: 梯度冒烟。24层 state grad 全非零、无 NaN。"""
    model, tok = model_tokenizer
    sp = make_state_params(model, dtype=mx.float32)
    samples = prepare_samples_v2(str(DATA_PATH / "data_100.jsonl"), tok, max_len=64)
    s = samples[0]
    inp = mx.array([s[0]]); lab = mx.array([s[1]])
    msk = mx.array([[float(x) for x in s[2]]], dtype=mx.float32)
    loss, grads = mx.value_and_grad(lambda sd: _masked_loss(model, inp, lab, msk, sd))(sp)
    mx.eval(loss, grads)

    zero_layers, nan_layers = [], []
    for i, g in grads.items():
        if float(mx.abs(g).sum()) < 1e-12:
            zero_layers.append(i)
        if bool(mx.any(mx.isnan(g))):
            nan_layers.append(i)
    assert not nan_layers, f"NaN 层: {nan_layers}"
    assert not zero_layers, f"零梯度层 (patch未生效?): {zero_layers}"


def test_overfit(model_tokenizer):
    """第二级: 10条样本过拟合 (lr=1.0, 200步), loss 应 < 0.5。"""
    model, tok = model_tokenizer
    samples = prepare_samples_v2(str(DATA_PATH / "data_100.jsonl"), tok, max_len=64)[:10]
    sp = make_state_params(model, dtype=mx.float32)
    opt = optim.Adam(learning_rate=1.0, betas=[0.9, 0.99], eps=1e-8)
    random.seed(42)
    best = 1e9
    for step in range(200):
        s = samples[step % 10]
        inp = mx.array([s[0]]); lab = mx.array([s[1]])
        msk = mx.array([[float(x) for x in s[2]]], dtype=mx.float32)
        # warmup 10 → cosine 1.0→0.01
        if step < 10:
            lr = 1.0 * (step + 1) / 10
        else:
            prog = (step - 10) / max(1, 200 - 10)
            lr = 0.01 + (1.0 - 0.01) * 0.5 * (1 + math.cos(math.pi * prog))
        opt.learning_rate = lr
        loss, grads = mx.value_and_grad(lambda sd: _masked_loss(model, inp, lab, msk, sd))(sp)
        grads = {k: mx.clip(g, -1.0, 1.0) for k, g in grads.items()}
        sp = opt.apply_gradients(grads, sp)
        mx.eval(sp, loss)
        best = min(best, float(loss))
    assert best < 0.5, f"过拟合失败: 最低 loss {best:.4f} >= 0.5"


def test_full_train_and_translate(model_tokenizer):
    """全量训练 (lr=0.01, 6 epoch) + 训后翻译验证。

    三项行为断言:
      1. loss 收敛 < 1.0
      2. state std 在合理范围 (0.05~1.0, 不爆炸)
      3. 训出的 state 能让中文翻译成英文
    """
    model, tok = model_tokenizer
    samples = prepare_samples_v2(str(DATA_PATH / "data_100.jsonl"), tok, max_len=128)
    sp, last_loss, sstd = _run_train(
        model, samples, lr_peak=0.01, lr_floor=0.0001, epochs=6)

    # 1. loss 收敛
    assert last_loss < 1.0, f"loss 未收敛: {last_loss:.4f} >= 1.0"
    # 2. state std 合理
    assert 0.05 < sstd < 1.0, f"state std 异常: {sstd:.4f} (合理 0.05~1.0)"
    # 3. 训出的 state 能翻译
    cn = samples[0][4]  # 第5元是中文原文
    out = generate(model, tok, P0_BARE.format_prefix(cn=cn), state_npz=None, max_tokens=50)  # 先确认构造可用
    # 注入训出的 state 重新生成
    import numpy as _np
    tmp = Path(__file__).parent / "_tmp_trained.npz"
    _np.savez(tmp, **{f"layer_{k}": _np.array(sp[k]) for k in sp})
    out = generate(model, tok, P0_BARE.format_prefix(cn=cn), state_npz=str(tmp), max_tokens=50)
    tmp.unlink(missing_ok=True)
    out = out.split("\n")[0].strip() if "\n" in out else out.strip()
    letters = [c for c in out if c.isalpha()]
    en_ratio = sum(1 for c in letters if ord(c) < 128) / max(1, len(letters))
    assert en_ratio > 0.5, f"训出的 state 未产生英文翻译 (en_ratio={en_ratio:.2f}): {out!r}"
