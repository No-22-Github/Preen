"""
单 lr 验证: lr=0.01 (比 P0 指南的 1.0 小 100 倍)。

假设: state 爆炸 (std 7~12) 是 lr=1.0 所致。
正常推理 state std 约 0.1, lr=0.01 应让 state 温和生长到合理范围。

训完检查:
  1. state std 是否在 ~0.1~1 量级 (合理范围)
  2. held-out 输出是否语义相关
  3. 条件性: 英文输入是否不触发翻译
"""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load

from state_tuner import patch_rwkv7_for_train, make_state_params, forward_with_state, cosine_lr
from data_v2 import prepare_samples_v2

MODEL = str(Path(__file__).parent.parent.parent / "models" / "converted" / "rwkv7-g1d-0.4b")
DATA = str(Path(__file__).parent.parent.parent / "train_data" / "translate" / "data_100.jsonl")
CKPT_DIR = Path(__file__).parent / "checkpoints_v3"


def pad_one(s):
    return mx.array([s[0]]), mx.array([s[1]]), mx.array([[float(x) for x in s[2]]], dtype=mx.float32)


def main():
    print("=" * 60)
    print("lr=0.01 验证训练 (排查 state 爆炸)")
    print("=" * 60)
    random.seed(42)
    Path(CKPT_DIR).mkdir(exist_ok=True)
    patch_rwkv7_for_train()
    model, tok = load(MODEL, tokenizer_config={"trust_remote_code": True})
    model.freeze()
    samples = prepare_samples_v2(DATA, tok, max_len=128)

    sp = make_state_params(model, dtype=mx.float32)
    opt = optim.Adam(learning_rate=0.01, betas=[0.9, 0.99], eps=1e-8)

    EPOCHS = 20
    total = EPOCHS * len(samples)
    print(f"lr=0.01→0.0001 cosine, {EPOCHS} epochs, 每2epoch存checkpoint")
    print(f"{'ep':>3} {'loss':>8} {'state_std':>10}")
    print("-" * 30)

    step = 0
    for ep in range(EPOCHS):
        order = list(range(len(samples)))
        random.shuffle(order)
        losses = []
        for si in order:
            s = samples[si]
            inp, lab, msk = pad_one(s)

            def loss_fn(sd):
                logits = forward_with_state(model, inp, sd, 1)
                lp = nn.log_softmax(logits, -1)
                g = mx.take_along_axis(lp, lab[..., None], -1).squeeze(-1)
                return (-g * msk).sum() / mx.maximum(msk.sum(), 1.0)

            lg = mx.value_and_grad(loss_fn)
            lr = cosine_lr(step, total, type("C", (), {"lr_peak": 0.01, "lr_floor": 0.0001, "warmup": 10})())
            opt.learning_rate = lr
            loss, grads = lg(sp)
            grads = {k: mx.clip(g, -1.0, 1.0) for k, g in grads.items()}
            sp = opt.apply_gradients(grads, sp)
            mx.eval(sp, loss)
            losses.append(float(loss))
            step += 1

        avg = sum(losses) / len(losses)
        sstd = float(np.mean([np.array(sp[i]).std() for i in sp]))
        if ep % 2 == 0 or ep == EPOCHS - 1:
            print(f"{ep:>3} {avg:>8.3f} {sstd:>10.4f}")
        if (ep + 1) % 4 == 0 or ep == EPOCHS - 1:
            np.savez(CKPT_DIR / f"ep{ep+1:02d}.npz", **{f"layer_{k}": np.array(sp[k]) for k in sp})


if __name__ == "__main__":
    main()
