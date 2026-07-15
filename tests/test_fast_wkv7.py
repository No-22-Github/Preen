"""test_fast_wkv7.py —— Metal checkpoint kernel 正确性回归。

对比对象:
  - fast_wkv7.make_wkv7_checkpoint(整段 Metal kernel,带 VJP)
  - mlx_lm._wkv7_step_ops Python 循环(= core.patch_rwkv7_for_train 注入的 ops 路径)

不加载模型,纯算子级快测(~5s)。覆盖:
  1. forward 数值一致(相对误差 < 1e-3,实测 ~8e-4,fp32 累加顺序差异)
  2. 6 梯度(dr/dw/dk/dv/da/db)一致(相对误差 < 2e-3,实测 < 1.3e-3)
  3. S₀ 透传梯度(d_h_in)一致 —— Preen 唯一可训参数,梯度必须穿回 kernel

口径:用 np.testing.assert_allclose(rtol=...),与 tests/ 现有数值测试同风格。
"""
import numpy as np
import pytest

import mlx.core as mx
from mlx_lm.models.rwkv7 import _wkv7_step_ops

from statetuner.fast_wkv7 import make_wkv7_checkpoint

# 0.4B 模型维度:hidden=1024, head_dim=64 → H=16
B, T, H, D = 1, 512, 16, 64
# forward 用"全局相对误差"判据(max_abs_diff / max_abs_output),实测 ~8e-4。
# 不用 np.testing.assert_allclose 的逐元素 rtol:输出含接近零的元素时,
# 逐元素 rtol 会被零附近的小绝对差放大成假阳性(实测 max_rel_diff=15× 但
# 全局相对误差仅 8e-4,纯属 fp32 累加顺序差异,非数值错误)。
FWD_REL_TOL = 1e-3
RTOL_GRAD = 2e-3  # 梯度逐元素相对误差(梯度普遍非零,逐元素 rtol 合适;实测 < 1.3e-3)


def _assert_fwd_close(actual: mx.array, desired: mx.array, msg: str, tol: float = FWD_REL_TOL):
    """forward 数值一致判据:全局相对误差 = max|a-d| / max|d| < tol。"""
    mx.eval(actual, desired)
    a, d = np.asarray(actual), np.asarray(desired)
    abs_diff = float(np.max(np.abs(a - d)))
    norm = float(np.max(np.abs(d)))
    rel = abs_diff / norm if norm > 0 else abs_diff
    assert rel < tol, (
        f"{msg}: 全局相对误差 {rel:.3e} >= {tol:.0e} "
        f"(abs_diff={abs_diff:.3e}, output_norm={norm:.3e})"
    )


def _make_inputs(seed: int = 42, sigma: float = 0.3):
    """固定 seed 造输入;σ=0.3 是 AGENTS.md 训练口径的典型量级。"""
    mx.random.seed(seed)
    r = mx.random.normal((B, T, H, D)) * sigma
    w = mx.ones((B, T, H, D)) * 0.95
    k = mx.random.normal((B, T, H, D)) * sigma
    v = mx.random.normal((B, T, H, D)) * sigma
    a = mx.random.normal((B, T, H, D)) * sigma * 0.3
    b = mx.random.normal((B, T, H, D)) * sigma * 0.3
    return r, w, k, v, a, b


def _ops_forward(r, w, k, v, a, b, h_in):
    """复刻 core._wkv7_train 闭包的 ops 循环(forward only)。"""
    state = h_in
    ys = []
    for t in range(T):
        y, state = _wkv7_step_ops(r[:, t], w[:, t], k[:, t], v[:, t], a[:, t], b[:, t], state)
        ys.append(y.squeeze(-1))
    return mx.stack(ys, axis=1)


@pytest.fixture(scope="module")
def inputs():
    return _make_inputs()


@pytest.fixture(scope="module")
def h_in():
    # S₀ 非零:验证梯度能穿回非平凡初值(零初值下 d_h_in 仍非零,但非零初值更严)
    mx.random.seed(7)
    return (mx.random.normal((B, H, D, D)) * 0.05).astype(mx.float32)


@pytest.fixture(scope="module")
def wkv7_kernel():
    return make_wkv7_checkpoint(B, T, H, D)


class TestForward:
    """forward 数值一致性(kernel vs ops 循环)。"""

    def test_forward_matches_ops(self, inputs, h_in, wkv7_kernel):
        r, w, k, v, a, b = inputs
        out_ops = _ops_forward(r, w, k, v, a, b, h_in)
        out_met, _ = wkv7_kernel(r, w, k, v, a, b, h_in)
        _assert_fwd_close(out_met, out_ops, "forward: Metal kernel vs ops 循环不一致")


class TestGradients:
    """6 输入梯度 + S₀ 透传梯度一致性。"""

    def test_six_input_grads(self, inputs, h_in, wkv7_kernel):
        r, w, k, v, a, b = inputs

        def loss_ops(r, w, k, v, a, b, h_in):
            return mx.mean(_ops_forward(r, w, k, v, a, b, h_in))

        def loss_met(r, w, k, v, a, b, h_in):
            o, _ = wkv7_kernel(r, w, k, v, a, b, h_in)
            return mx.mean(o)

        _, g_ops = mx.value_and_grad(loss_ops, argnums=list(range(7)))(r, w, k, v, a, b, h_in)
        _, g_met = mx.value_and_grad(loss_met, argnums=list(range(7)))(r, w, k, v, a, b, h_in)
        mx.eval(*g_ops, *g_met)

        names = ["dr", "dw", "dk", "dv", "da", "db"]
        for i, n in enumerate(names):
            np.testing.assert_allclose(
                np.asarray(g_met[i]), np.asarray(g_ops[i]),
                rtol=RTOL_GRAD, atol=1e-6,
                err_msg=f"梯度 {n}: Metal kernel vs ops 循环不一致",
            )

    def test_state_grad_passthrough(self, inputs, h_in, wkv7_kernel):
        """d_h_in(S₀ 透传梯度)一致性 —— Preen 唯一可训参数,核心路径。"""
        r, w, k, v, a, b = inputs

        def loss_ops(r, w, k, v, a, b, h_in):
            return mx.mean(_ops_forward(r, w, k, v, a, b, h_in))

        def loss_met(r, w, k, v, a, b, h_in):
            o, _ = wkv7_kernel(r, w, k, v, a, b, h_in)
            return mx.mean(o)

        _, g_ops = mx.value_and_grad(loss_ops, argnums=list(range(7)))(r, w, k, v, a, b, h_in)
        _, g_met = mx.value_and_grad(loss_met, argnums=list(range(7)))(r, w, k, v, a, b, h_in)
        mx.eval(g_ops[6], g_met[6])

        d_h_in_ops = np.asarray(g_ops[6])
        d_h_in_met = np.asarray(g_met[6])
        np.testing.assert_allclose(
            d_h_in_met, d_h_in_ops,
            rtol=RTOL_GRAD, atol=1e-7,
            err_msg="d_h_in (S₀ 透传梯度): kernel vs ops 不一致 —— 可训练 S₀ 路径断裂",
        )
        # 防回归:梯度不应全零(零 state 下仍应有非零梯度流)
        assert np.abs(d_h_in_met).max() > 0, "d_h_in 全零 —— 梯度未穿回 S₀"


class TestEdgeCases:
    """边界:零 state、不同 T。"""

    def test_zero_state_forward(self, inputs, wkv7_kernel):
        """零 S₀(推理默认)forward 一致。"""
        r, w, k, v, a, b = inputs
        h0 = mx.zeros((B, H, D, D))
        out_ops = _ops_forward(r, w, k, v, a, b, h0)
        out_met, _ = wkv7_kernel(r, w, k, v, a, b, h0)
        _assert_fwd_close(out_met, out_ops, "零 state forward 不一致")

    def test_short_sequence_T128(self):
        """短序列(T=128,4 chunks)正确性 —— 防边界 off-by-one。"""
        T_short = 128
        mx.random.seed(99)
        r = mx.random.normal((B, T_short, H, D)) * 0.3
        w = mx.ones((B, T_short, H, D)) * 0.95
        k = mx.random.normal((B, T_short, H, D)) * 0.3
        v = mx.random.normal((B, T_short, H, D)) * 0.3
        a = mx.random.normal((B, T_short, H, D)) * 0.1
        b = mx.random.normal((B, T_short, H, D)) * 0.1
        h0 = mx.zeros((B, H, D, D))

        def ops_fwd(r, w, k, v, a, b, h_in):
            state = h_in
            ys = []
            for t in range(T_short):
                y, state = _wkv7_step_ops(r[:, t], w[:, t], k[:, t], v[:, t], a[:, t], b[:, t], state)
                ys.append(y.squeeze(-1))
            return mx.stack(ys, axis=1)

        kern = make_wkv7_checkpoint(B, T_short, H, D)
        out_ops = ops_fwd(r, w, k, v, a, b, h0)
        out_met, _ = kern(r, w, k, v, a, b, h0)
        _assert_fwd_close(out_met, out_ops, "T=128 forward 不一致")
