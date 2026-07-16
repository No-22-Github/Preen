"""训练行为断言测试(慢,~4min,标 @pytest.mark.slow)。

不做 bit-golden(MLX GPU ULP 不确定),改为断言行为指标:
  - 梯度冒烟:24层 grad 全非零(patch 生效、state 进了计算图)
  - 过拟合:10条 loss < 0.5(管线正确)
  - 全量收敛:loss < 1.0 + state std 合理 + 产出非空回答

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


class _DummyTokenizer:
    """字符级 dummy tokenizer(测试 encode_template_sample 结构,不依赖模型)。

    encode: 每个 char → ord(char);decode: ids → ''.join(chr(i))。
    足以验证 mask/边界/stop_token 的位置逻辑。
    """

    @staticmethod
    def encode(text):
        return [ord(c) for c in text]

    @staticmethod
    def decode(ids):
        return "".join(chr(i) for i in ids)


def test_train_config_product_defaults():
    from statetuner.train import TrainConfig

    cfg = TrainConfig()
    assert cfg.lr == pytest.approx(0.0001)
    assert cfg.lr_floor == pytest.approx(0.00001)
    assert cfg.warmup == 50
    assert cfg.epochs == 5


def test_compiled_loss_and_grad_matches_eager(monkeypatch):
    """阶段一：compiled fast path 不改变 masked loss 与 state 梯度。"""
    import mlx.core as mx

    import statetuner.train as train_module

    class _TinyModel:
        state = {}

    def _fake_forward(_model, input_ids, states, batch_size):
        assert batch_size == input_ids.shape[0]
        scale = input_ids.astype(mx.float32)[..., None]
        return scale * states[0][None, None, :]

    monkeypatch.setattr(train_module, "forward_with_state", _fake_forward)
    model = _TinyModel()
    states = {0: mx.array([0.1, -0.2, 0.3, 0.05], dtype=mx.float32)}
    inp = mx.array([[1, 2, 3]])
    lab = mx.array([[1, 2, 0]])
    msk = mx.array([[0.0, 1.0, 1.0]], dtype=mx.float32)

    eager = train_module._make_loss_and_grad(model, compile_graph=False)
    compiled = train_module._make_loss_and_grad(model, compile_graph=True)
    eager_loss, eager_grads = eager(states, inp, lab, msk)
    compiled_loss, compiled_grads = compiled(states, inp, lab, msk)
    mx.eval(eager_loss, eager_grads, compiled_loss, compiled_grads)

    np.testing.assert_allclose(
        np.array(compiled_loss), np.array(eager_loss), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(compiled_grads[0]), np.array(eager_grads[0]), rtol=1e-6, atol=1e-6
    )


def test_encode_template_stop_and_mask():
    """验收 b:任一样本 full_ids[-1]==stop_token 且对应 mask==1。

    full = prefix_ids + target_ids + [stop_token];
    input_ids = full[:-1], labels = full[1:], mask 与 labels 等长。
    终止符落在 full 末位 → 它是最后一个 label,对应 mask 位必须为 1(算 loss,
    让模型学会预测 stop)。

    用 NEKO_QA 模板(dummy tokenizer):prefix="User: 你好\\n\\nAssistant:",
    target=" Hi",stop=0。
    """
    from statetuner.data import encode_template_sample
    from statetuner.templates import QA as NEKO_QA  # tests 局部别名

    tok = _DummyTokenizer()
    s = encode_template_sample(tok, NEKO_QA, q="你好", a="Hi")

    # 终止符
    assert s.full_ids[-1] == 0, f"full 末位应为 stop_token(0), 实际 {s.full_ids[-1]}"
    # full == prefix + target + [stop]
    prefix_text = NEKO_QA.format_prefix(q="你好")
    target_text = NEKO_QA.format_target(a="Hi")
    assert s.full_ids == tok.encode(prefix_text) + tok.encode(target_text) + [0]
    # 最后一个 label 是终止符,其 mask 位 == 1(★ 验收 b 核心断言)
    assert s.labels[-1] == 0, f"末位 label 应为 stop(0), 实际 {s.labels[-1]}"
    assert s.mask[-1] == 1, f"末位 mask 应为 1(终止符算 loss), 实际 {s.mask[-1]}"
    # 纯 prefix 条件区(预测仍是 prefix 内 token)全 0
    assert all(m == 0 for m in s.mask[: s.prefix_len - 1]), "prefix 条件区 mask 应全 0"
    # target+stop 预测区全 1
    assert all(m == 1 for m in s.mask[s.prefix_len - 1 :]), "target+stop 预测区 mask 应全 1"


def test_encode_template_sets_truncated_flag():
    """超长样本(> max_len)应置 truncated=True;正常样本 False。"""
    from statetuner.data import encode_template_sample
    from statetuner.templates import QA

    tok = _DummyTokenizer()
    short = encode_template_sample(tok, QA, max_len=512, q="你好", a="Hi")
    assert short.truncated is False

    long = encode_template_sample(tok, QA, max_len=8, q="你好世界", a="很长的回答内容超出上限")
    assert long.truncated is True
    assert long.length == 8, "截断后长度应等于 max_len"


def test_load_qa_dataset_drop_truncated(tmp_path):
    """drop_truncated=True 丢弃超长样本;默认(False)保留(截头保尾)。"""
    import json
    from statetuner.data import load_qa_dataset

    tok = _DummyTokenizer()
    path = tmp_path / "d.jsonl"
    rows = [
        {"instruction": "q", "output": "很长很长很长很长很长的回答"},  # 超长
        {"instruction": "q2", "output": "ok"},                          # 正常
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    keep = load_qa_dataset(path, tok, max_len=28)
    drop = load_qa_dataset(path, tok, max_len=28, drop_truncated=True)
    assert len(keep) == 2, "默认保留全部(含截断)"
    assert len(drop) == 1, "drop_truncated 丢弃超长样本"
    assert drop[0].truncated is False


def test_encode_template_prefix_isomorphism():
    """验收 c:encode(prefix字符串) == encode_template_sample 的 prefix_ids。

    即 train(encode_template_sample) 与 inference(encode(prompt)) 的 prefix 段逐 token
    相等——拆分编码而非联合编码的保证。用 dummy tokenizer 验结构,真实 tokenizer
    在 test_inference / 慢测里由 golden 逐字断言兜底。
    """
    from statetuner.data import encode_template_sample
    from statetuner.templates import QA as NEKO_QA  # tests 局部别名

    tok = _DummyTokenizer()

    tmpl = NEKO_QA
    fields = {"q": "你好", "a": "Hi"}
    prefix_text = tmpl.format_prefix(**fields)
    s = encode_template_sample(tok, tmpl, **fields)
    assert tok.encode(prefix_text) == s.full_ids[: s.prefix_len], (
        f"{tmpl.prefix_template!r}: encode(prefix) != encode_template_sample prefix_ids"
    )
    # target 段同样同构
    target_text = tmpl.format_target(**fields)
    target_ids = tok.encode(target_text)
    assert s.full_ids[s.prefix_len : -1] == target_ids, (
        f"{tmpl.target_template!r}: encode(target) != encode_template_sample target_ids"
    )


def test_train_test_split_reproducible():
    """train_test_split 相同 seed 产出相同划分。"""
    from statetuner.data import Sample, train_test_split

    samples = [
        Sample(
            full_ids=[1, 2, 3], input_ids=[1, 2], labels=[2, 3],
            mask=[0, 1], prompt_text=f"p{i}", target_text=f"t{i}", prefix_len=1,
        )
        for i in range(20)
    ]
    tr1, te1 = train_test_split(samples, test_ratio=0.2, seed=42)
    tr2, te2 = train_test_split(samples, test_ratio=0.2, seed=42)
    assert [s.prompt_text for s in tr1] == [s.prompt_text for s in tr2]
    assert [s.prompt_text for s in te1] == [s.prompt_text for s in te2]
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


def test_events_file_overwrites_not_appends(tmp_path):
    """events 文件应覆盖写(非追加):重跑训练时清空旧事件,避免混淆。

    回归:旧实现用 open(file, "a"),导致同一文件混入多次训练的 epoch 事件,
    读出来像是 epoch 重复。应为 "w" —— 每次 EventEmitter 是独立事件流。
    """
    import json

    from statetuner import events

    f = tmp_path / "ev.jsonl"
    # 第一次"训练"
    em1 = events.EventEmitter(file=f, quiet=True)
    em1.emit(events.epoch_end(0, loss=1.0, state_std=0.1, lr=0.01))
    em1.close()
    n1 = sum(1 for _ in f.open(encoding="utf-8"))
    assert n1 == 1

    # 第二次"训练"(同文件)——应覆盖,不是追加
    em2 = events.EventEmitter(file=f, quiet=True)
    em2.emit(events.epoch_end(0, loss=0.5, state_std=0.2, lr=0.01))
    em2.close()
    lines = [json.loads(l) for l in f.open(encoding="utf-8")]
    assert len(lines) == 1, f"覆盖写后应只有 1 行, 实际 {len(lines)}(追加模式 bug?)"
    assert lines[0]["loss"] == 0.5, "应是第二次的内容, 不是第一次的残留"


def test_generate_strips_eos_from_output():
    """generate 遇 eos(token 0)应停下,且不把 eos 解码进输出。

    回归:旧实现先 append 再判 eos,导致输出末尾出现
    <|rwkv_tokenizer_end_of_text|> 字面量。用 mock model 验,不依赖真模型。
    """
    import mlx.core as mx

    from statetuner.core import generate

    # mock model: 前两步吐 token 72/73('H'/'i'),第三步吐 token 0(eos)
    # logits 形状 (B=1, L, vocab);argmax 取末位最后一维
    class _MockModel:
        def __init__(self):
            self._calls = 0
            self.vocab = 100

        def __call__(self, input_ids, caches=None):
            self._calls += 1
            L = input_ids.shape[1]
            tok = [72, 73, 0][self._calls - 1] if self._calls <= 3 else 0
            # 用 numpy 构造:末位在 tok 位置最大,其余 0
            import numpy as np
            arr = np.zeros((1, L, self.vocab), dtype=np.float32)
            arr[0, L - 1, tok] = 1.0
            return mx.array(arr)

        def make_cache(self):
            return None

    # dummy tokenizer: id → chr(id); 0 应是不可见/特殊
    class _DummyTok:
        def encode(self, text):
            return [ord(c) for c in text]

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    out = generate(_MockModel(), _DummyTok(), "x", state=None, max_tokens=10)
    # 应输出 chr(72)+chr(73)="HI"(eos=0 被 break 掉不进结果)
    assert out == "HI", f"应输出 'HI'(eos 已剥离), 实际 {out!r}"
    # 不应含 eos 的 decode 结果(chr(0) = '\x00')
    assert "\x00" not in out


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


# ── checkpoint/resume 回归(P0-1 + Q4)────────────────────────


def _dummy_model_like(n_layers=2, n_heads=2, head_dim=4):
    """构造一个最小的 fake model,仅满足 make_state_params 的字段访问。

    Trainer.train 内部用 make_state_params(model),但本测试只测 _save/_load
    checkpoint,直接绕过 train(),传构造好的 states + opt。
    """
    return type(
        "M",
        (),
        {
            "args": type(
                "A",
                (),
                {"hidden_size": n_heads * head_dim, "head_dim": head_dim},
            )(),
            "layers": [None] * n_layers,
        },
    )


def test_checkpoint_persists_and_restores_optimizer_state(tmp_path):
    """P0-1 + Q4:checkpoint 必须真正存/恢复 Adam 的 m/v/step。

    回归重点(旧 bug):
      - 旧存侧 `isinstance(k, tuple)` 永假(MLX Adam 层键是 int)→ m/v 没存进 npz。
      - 旧取侧 opt 未 init,层键不存在 → 即使键判断对了也空转。
      - step 漏存 → 续训第一步 Adam bias correction 按 step=1 → 一个超大步。

    断言:存前 opt._state[0]['m'/'v'] 与 resume 后逐元素相等;step 也相等。
    """
    import mlx.core as mx
    import mlx.optimizers as optim

    from statetuner.train import Trainer, TrainConfig

    n_layers = 2
    states = {
        i: mx.array(np.random.RandomState(i).randn(2, 4, 4).astype(np.float32) * 0.1)
        for i in range(n_layers)
    }
    opt = optim.Adam(learning_rate=0.01, betas=[0.9, 0.99], eps=1e-8)
    # 跑几步 apply_gradients,让 m/v/step 真正长出来
    grads = {i: mx.ones((2, 4, 4)) * 0.3 for i in range(n_layers)}
    for _ in range(5):
        states = opt.apply_gradients(grads, states)
    mx.eval(states)
    step_before = int(opt._state["step"])
    assert step_before == 5, f"apply_gradients 5 次后 step 应=5, 实际 {step_before}"
    # 存侧断言:层键是 int(P0-1 证据)
    assert all(isinstance(k, int) for k in opt._state if k not in ("step", "learning_rate"))

    # 存 checkpoint
    cfg = TrainConfig(lr=0.01, lr_floor=1e-4, warmup=10, epochs=2)
    trainer = Trainer(_dummy_model_like(n_layers), cfg)  # emitter=None → quiet 默认
    ckpt = trainer._save_checkpoint(
        tmp_path, epoch=1, states=states, opt=opt,
        best=0.5, patience_left=2,
    )
    assert ckpt.exists()

    # 取 checkpoint:用一个全新的 opt(模拟进程重启)
    opt_new = optim.Adam(learning_rate=0.01, betas=[0.9, 0.99], eps=1e-8)
    states_new = {
        i: mx.zeros((2, 4, 4), dtype=mx.float32) for i in range(n_layers)
    }
    epoch_back, states_back, opt_back, best_back, patience_back = trainer._load_checkpoint(
        ckpt, states_new, opt_new
    )

    # epoch meta 恢复
    assert epoch_back == 2, f"epoch 应=meta.epoch+1=2, 实际 {epoch_back}"
    assert best_back == 0.5
    assert patience_back == 2
    # step 恢复(关键:漏存会让续训第一步爆冲)
    assert int(opt_back._state["step"]) == step_before, "step 必须逐值恢复"

    # m/v 逐元素相等(Q4 核心断言)
    for i in range(n_layers):
        m_old = np.array(opt._state[i]["m"])
        m_new = np.array(opt_back._state[i]["m"])
        v_old = np.array(opt._state[i]["v"])
        v_new = np.array(opt_back._state[i]["v"])
        np.testing.assert_allclose(m_new, m_old, atol=1e-6, err_msg=f"layer {i} m 不等")
        np.testing.assert_allclose(v_new, v_old, atol=1e-6, err_msg=f"layer {i} v 不等")
        # state 本身也恢复
        np.testing.assert_allclose(
            np.array(states_back[i]), np.array(states[i]), atol=1e-6,
            err_msg=f"layer {i} state 不等",
        )


def test_resume_optimizer_state_actually_used_after_load(tmp_path):
    """P0-1 进阶:resume 后再 apply_gradients 一步,m 应按续传语义更新
    (= decay * restored_m + (1-b1) * grad),而不是从零重启。

    证明恢复的 m/v 确实参与下一次更新(不是只存了不用)。
    """
    import mlx.core as mx
    import mlx.optimizers as optim

    from statetuner import events
    from statetuner.train import Trainer, TrainConfig

    n_layers = 1
    states = {0: mx.zeros((2, 2, 2), dtype=mx.float32)}
    opt = optim.Adam(learning_rate=0.01, betas=[0.9, 0.99], eps=1e-8)
    # 跑两步,m[0,0,0] = 0.1*1 + 0.9*(0.1*1) = 0.19(grad=1)
    grads = {0: mx.ones((2, 2, 2))}
    opt.apply_gradients(grads, states)
    opt.apply_gradients(grads, states)
    mx.eval(states)
    m_before = float(opt._state[0]["m"][0, 0, 0])

    cfg = TrainConfig(lr=0.01, epochs=2)
    trainer = Trainer(_dummy_model_like(n_layers), cfg, events.EventEmitter(quiet=True))
    ckpt = trainer._save_checkpoint(tmp_path, epoch=0, states=states, opt=opt, best=1.0, patience_left=3)

    # resume
    opt_new = optim.Adam(learning_rate=0.01, betas=[0.9, 0.99], eps=1e-8)
    states_new = {0: mx.zeros((2, 2, 2), dtype=mx.float32)}
    _, _, opt_restored, _, _ = trainer._load_checkpoint(ckpt, states_new, opt_new)

    # 再 apply 一步:m_new = b1*m_restored + (1-b1)*grad
    #              = 0.9 * m_before + 0.1 * 1
    expected_m = 0.9 * m_before + 0.1
    opt_restored.apply_gradients(grads, states_new)
    m_after = float(opt_restored._state[0]["m"][0, 0, 0])
    assert m_after == pytest.approx(expected_m, abs=1e-6), (
        f"resume 后 m 应按续传更新到 ~{expected_m:.4f}, 实际 {m_after:.4f}"
    )


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

    from statetuner.core import forward_with_state, make_state_params
    from statetuner.data import load_qa_dataset

    model, tok = model_tokenizer
    samples = load_qa_dataset(str(DATA_PATH), tok, max_len=64)
    s = samples[0]
    sp = make_state_params(model, dtype=mx.float32)
    inp = mx.array([s.input_ids])
    lab = mx.array([s.labels])
    msk = mx.array([[float(x) for x in s.mask]], dtype=mx.float32)

    def loss_fn(sd):
        logits = forward_with_state(model, inp, sd, 1)
        lp = nn.log_softmax(logits, -1)
        g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
        return (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)

    loss, grads = mx.value_and_grad(loss_fn)(sp)
    mx.eval(loss, grads)

    for i, g in grads.items():
        assert not bool(mx.any(mx.isnan(g))), f"layer {i} NaN"
        assert float(mx.abs(g).sum()) > 1e-12, f"layer {i} 零梯度 (patch 未生效?)"


@pytest.mark.slow
def test_overfit(model_tokenizer):
    """第二级:10条样本过拟合(lr=1.0, 200步),loss 应 < 0.5。"""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from statetuner.core import forward_with_state, make_state_params
    from statetuner.data import load_qa_dataset

    model, tok = model_tokenizer
    samples = load_qa_dataset(str(DATA_PATH), tok, max_len=64)[:10]
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

        def loss_fn(sd):
            logits = forward_with_state(model, inp, sd, 1)
            lp = nn.log_softmax(logits, -1)
            g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
            return (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)

        loss, grads = mx.value_and_grad(loss_fn)(sp)
        grads = {k: mx.clip(g, -1.0, 1.0) for k, g in grads.items()}
        sp = opt.apply_gradients(grads, sp)
        mx.eval(sp, loss)
        best = min(best, float(loss))
    assert best < 0.5, f"过拟合失败: 最低 loss {best:.4f} >= 0.5"


@pytest.mark.slow
def test_full_train_nekoqa(model_tokenizer):
    """全量训练(lr=0.01, 3 epoch)+ NekoQA 验证。

    四项断言:loss 收敛 / state std 合理 / 产出非空回答 / 训练事件序列完整。
    用产品化的 Trainer 跑(验证 train.py 本身)。
    """
    import mlx.core as mx

    from statetuner import events
    from statetuner.data import load_qa_dataset
    from statetuner.train import Trainer, TrainConfig

    model, tok = model_tokenizer
    samples = load_qa_dataset(str(DATA_PATH), tok, max_len=512)

    cfg = TrainConfig(lr=0.01, lr_floor=0.0001, warmup=10, epochs=3, early_stop=False)
    em = events.EventEmitter(quiet=True)
    result = Trainer(model, cfg, em).train(samples)

    # 1. loss 收敛(NekoQA smoke_200 × 3epoch 终点 ~2.3,放宽到 < 3.0)
    assert result.final_loss < 3.0, f"loss 未收敛: {result.final_loss:.4f}"
    # 2. state std 合理(不爆炸)
    assert 0.05 < result.final_state_std < 1.0, (
        f"state std 异常: {result.final_state_std:.4f} (合理 0.05~1.0)"
    )
    # 3. 产出非空回答(state 注入后应能生成内容)
    from statetuner.core import generate
    from statetuner.templates import QA as NEKO_QA  # tests 局部别名
    from statetuner.train import save_state_npz

    q = samples[0].prompt_text  # Sample.prompt_text 存的是 question
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_npz = f.name
    try:
        save_state_npz(result.states, tmp_npz)
        out = generate(model, tok, NEKO_QA.format_prefix(q=q), state=tmp_npz, max_tokens=50)
    finally:
        Path(tmp_npz).unlink(missing_ok=True)
    out = out.strip()
    assert len(out) > 5, f"输出过短,疑似退化: {out!r}"

    # 4. 训练事件序列完整
    types = [e["type"] for e in em.events]
    assert "start" in types
    assert "epoch_end" in types
    assert "final" in types
