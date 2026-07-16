"""test_fast_wkv7.py —— Metal checkpoint kernel 正确性回归。

对比对象:
  - fast_wkv7.make_wkv7_checkpoint(整段 Metal kernel,带 VJP)
  - mlx_lm._wkv7_step_ops Python 循环(= core.patch_rwkv7_for_train 注入的 ops 路径)

不加载模型,纯算子级快测(~5s)。覆盖:
  1. forward 数值一致(相对误差 < 1e-3,实测 ~8e-4,fp32 累加顺序差异)
  2. 6 梯度(dr/dw/dk/dv/da/db)一致(相对误差 < 2e-3,实测 < 1.3e-3)
  3. S₀ 透传梯度(d_h_in)一致 —— Preen 唯一可训参数,梯度必须穿回 kernel
  4. T=64 跨 chunk、非零 h_in 与 h_out cotangent 下,七梯度对 CPU einsum
     reference 的最坏相对误差 < 1e-5
  5. 七梯度 compiled backward 重复执行 bitwise 确定
  6. bf16 eager/compiled State backward 确定，且与 fp32 物化路径一致

口径:旧 ops 对照保留 np.testing.assert_allclose；跨实现 golden 用各梯度
max-abs 差除以 reference max-abs，再取七梯度最坏值。
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

# Metal backward 的严格 golden 口径。T=64 刚好跨越两个 CHUNK=32，
# 又足够小，可以在快速测试中用 CPU einsum 作独立 reference。
GOLDEN_B, GOLDEN_T, GOLDEN_H, GOLDEN_D = 2, 64, 4, 64
GRADIENT_NAMES = ("dr", "dw", "dk", "dv", "da", "db", "dh_in")
GOLDEN_REL_ERR = 1e-5
RACE_B, RACE_T, RACE_H, RACE_D = 1, 512, 24, 64


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


def _einsum_reference_with_state(r, w, k, v, a, b, h_in):
    """CPU einsum reference，显式接受 h_in 并返回 h_out。

    GPU einsum 在该 reduction 上会有约 1e-3 的 throughput-oriented
    累加偏差，无法作为 1e-5 golden 的仲裁。CPU stream 使这项
    测试真正衡量 Metal kernel，而不是 reference 自身的 GPU reduction。
    """
    with mx.stream(mx.cpu):
        h = h_in
        outputs = []
        for t in range(r.shape[1]):
            r_t, w_t, k_t = r[:, t], w[:, t], k[:, t]
            v_t, a_t, b_t = v[:, t], a[:, t], b[:, t]
            sa = mx.einsum("bhsd,bhd->bhs", h, a_t)
            h = (
                h * w_t[:, :, None, :]
                + mx.einsum("bhs,bhd->bhsd", v_t, k_t)
                + mx.einsum("bhs,bhd->bhsd", sa, b_t)
            )
            outputs.append(mx.einsum("bhsd,bhd->bhs", h, r_t))
        return mx.stack(outputs, axis=1), h


def _make_golden_inputs():
    """RWKV7 真实形态的 fp32 输入，避免近零 decay 病态放大。"""
    mx.random.seed(0)
    shape = (GOLDEN_B, GOLDEN_T, GOLDEN_H, GOLDEN_D)
    r, k, v = tuple(mx.random.normal(shape) * 0.5 for _ in range(3))
    a = mx.random.normal(shape)
    a = a / mx.sqrt(mx.sum(a * a, axis=-1, keepdims=True) + 1e-12)
    b = -a * mx.random.uniform(0.9, 1.1, shape=shape)
    w = mx.exp(-mx.exp(mx.random.normal(shape) * 0.5 - 2.5))
    h_in = mx.random.normal((GOLDEN_B, GOLDEN_H, GOLDEN_D, GOLDEN_D)) * 0.1
    p_out = mx.random.normal(shape)
    p_h = mx.random.normal((GOLDEN_B, GOLDEN_H, GOLDEN_D, GOLDEN_D))
    mx.eval(r, w, k, v, a, b, h_in, p_out, p_h)
    return (r, w, k, v, a, b, h_in), p_out, p_h


def _projected_loss(forward, p_out, p_h):
    """让 out 和 h_out 都产生非零、全元素混合的 cotangent。"""
    def loss(r, w, k, v, a, b, h_in):
        out, h_out = forward(r, w, k, v, a, b, h_in)
        return (out * p_out).sum() + (h_out * p_h).sum()

    return loss


def _golden_gradient_functions():
    inputs, p_out, p_h = _make_golden_inputs()
    metal = make_wkv7_checkpoint(
        GOLDEN_B, GOLDEN_T, GOLDEN_H, GOLDEN_D, chunk=32
    )
    metal_grad = mx.compile(
        mx.grad(_projected_loss(metal, p_out, p_h), argnums=list(range(7)))
    )
    reference_grad = mx.grad(
        _projected_loss(_einsum_reference_with_state, p_out, p_h),
        argnums=list(range(7)),
    )
    return inputs, metal_grad, reference_grad


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


class TestBackwardGolden:
    """T=64 跨 chunk 时的七梯度正确性与确定性。"""

    def test_gradients_match_cpu_einsum_reference(self):
        """七梯度最大相对误差必须小于 fp32 golden 线 1e-5。"""
        inputs, metal_grad, reference_grad = _golden_gradient_functions()
        metal_gradients = metal_grad(*inputs)
        reference_gradients = reference_grad(*inputs)
        mx.eval(*metal_gradients, *reference_gradients)

        relative_errors = {}
        for name, actual, expected in zip(
            GRADIENT_NAMES, metal_gradients, reference_gradients
        ):
            scale = float(mx.abs(expected).max()) + 1e-30
            max_error = float(mx.abs(actual - expected).max())
            relative_errors[name] = max_error / scale

        worst_name = max(relative_errors, key=relative_errors.get)
        worst = relative_errors[worst_name]
        details = ", ".join(
            f"{name}={relative_errors[name]:.2e}" for name in GRADIENT_NAMES
        )
        assert worst < GOLDEN_REL_ERR, (
            f"GOLDEN METRIC FAILED: {worst_name}={worst:.2e}; {details}"
        )

    def test_seven_gradients_are_bitwise_deterministic(self):
        """T=512/H=24 的 5 次独立 dispatch 中，七梯度逐位相等。

        T=64/H=4 是为 CPU golden 准备的小 fixture，不足以稳定触发
        缺 barrier 的竞态。确定性回归保留原始大配置、bf16 输入、
        2 次 eager + 3 次 compiled；每次都从 host snapshot 重建
        primal 并立即 eval，确保不是同一 lazy 结果或单次 dispatch 复用。
        """
        mx.random.seed(77)
        shape = (RACE_B, RACE_T, RACE_H, RACE_D)
        race_inputs = (
            (mx.random.normal(shape) * 0.3).astype(mx.bfloat16),
            (
                mx.sigmoid(mx.random.normal(shape) * 1.2) * 0.25 + 0.74
            ).astype(mx.bfloat16),
            (mx.random.normal(shape) * 0.3).astype(mx.bfloat16),
            (mx.random.normal(shape) * 0.3).astype(mx.bfloat16),
            (mx.random.normal(shape) * 0.09).astype(mx.bfloat16),
            (mx.random.normal(shape) * 0.09).astype(mx.bfloat16),
        )
        race_h_in = (
            mx.random.normal((RACE_B, RACE_H, RACE_D, RACE_D)) * 0.05
        ).astype(mx.float32)
        mx.eval(*race_inputs, race_h_in)

        # fp32 host snapshot 保存 bf16 的精确可表示值。每次 run 都会
        # 生成 storage 独立的 MLX 输入，排除 lazy graph 复用假阳性。
        input_values = tuple(
            np.array(value.astype(mx.float32), copy=True) for value in race_inputs
        )
        h_in_value = np.array(race_h_in, copy=True)
        kernel = make_wkv7_checkpoint(RACE_B, RACE_T, RACE_H, RACE_D)

        def loss_fn(r, w, k, v, a, b, state):
            output, h_out = kernel(r, w, k, v, a, b, state)
            time_weights = mx.linspace(0.5, 1.5, RACE_T)[None, :, None, None]
            state_weights = mx.linspace(-0.3, 0.7, RACE_D)[None, None, None, :]
            return mx.mean(output * time_weights) + mx.mean(
                h_out * state_weights
            )

        eager = mx.grad(loss_fn, argnums=list(range(7)))
        compiled = mx.compile(mx.grad(loss_fn, argnums=list(range(7))))

        def run(grad_fn):
            current_inputs = tuple(
                mx.array(value).astype(mx.bfloat16) for value in input_values
            )
            current_h_in = mx.array(h_in_value)
            gradients = grad_fn(*current_inputs, current_h_in)
            mx.eval(*gradients)
            return tuple(
                np.array(gradient.astype(mx.float32), copy=True)
                for gradient in gradients
            )

        runs = [
            run(eager),
            run(eager),
            run(compiled),
            run(compiled),
            run(compiled),
        ]
        reference = runs[0]
        for run_index, gradients in enumerate(runs[1:], start=2):
            for name, expected, actual in zip(
                GRADIENT_NAMES, reference, gradients
            ):
                np.testing.assert_array_equal(
                    actual,
                    expected,
                    err_msg=(
                        f"梯度 {name} 在第 {run_index} 次独立 backward "
                        "dispatch 发生变化"
                    ),
                )


class TestGradients:
    """6 输入梯度 + S₀ 透传梯度一致性。"""

    def test_state_grad_is_deterministic(self, inputs, h_in, wkv7_kernel):
        """同一输入重复 backward 必须得到完全一致的 S₀ 梯度。

        backward 每个时间步都会重用 threadgroup shared arrays。若上一个时间步
        仍在读取而下一个时间步已经覆盖它们，单次与 ops 参考值可能仍在容差内，
        但重复运行会随线程调度产生不同梯度。这里同时锁定 eager 与训练实际使用
        的 compiled graph，避免这类数据竞争再次潜入训练路径。
        """
        # 生产训练路径直接把模型 bf16 激活交给 Metal kernel；这里不能只测
        # 默认 fp32 随机数组，否则覆盖不到 fe0c493 引入的直读路径。
        r, w, k, v, a, b = (x.astype(mx.bfloat16) for x in inputs)
        h_in_np = np.array(h_in, copy=True)

        def loss_met(state):
            out, _ = wkv7_kernel(r, w, k, v, a, b, state)
            return mx.mean(out)

        eager_grad = mx.value_and_grad(loss_met)
        compiled_grad = mx.compile(mx.value_and_grad(loss_met))

        def run(grad_fn):
            # 每次创建值相同但 storage 独立的输入，确保 Metal backward 真正重跑，
            # 而不是复用已物化的 lazy graph 输出。
            state = mx.array(h_in_np)
            _, grad = grad_fn(state)
            mx.eval(grad)
            return np.array(grad, copy=True)

        eager_runs = [run(eager_grad) for _ in range(3)]
        compiled_runs = [run(compiled_grad) for _ in range(3)]
        reference = eager_runs[0]
        for mode, runs in (("eager", eager_runs), ("compiled", compiled_runs)):
            for index, grad in enumerate(runs, start=1):
                assert np.array_equal(grad, reference), (
                    f"{mode} 第 {index} 次 backward 的 S₀ 梯度不确定 —— "
                    "疑似存在 threadgroup shared-memory 数据竞争"
                )

    def test_bf16_direct_matches_fp32_materialization(self, inputs, h_in, wkv7_kernel):
        """bf16 直读保持旧 fp32 物化路径的 loss 与 S₀ 梯度语义。"""
        bf16_inputs = tuple(x.astype(mx.bfloat16) for x in inputs)

        def loss_direct(state):
            out, _ = wkv7_kernel(*bf16_inputs, state)
            return mx.mean(out)

        def loss_materialized(state):
            fp32_inputs = tuple(x.astype(mx.float32) for x in bf16_inputs)
            out, _ = wkv7_kernel(*fp32_inputs, state)
            return mx.mean(out)

        direct_loss, direct_grad = mx.value_and_grad(loss_direct)(h_in)
        old_loss, old_grad = mx.value_and_grad(loss_materialized)(h_in)
        mx.eval(direct_loss, direct_grad, old_loss, old_grad)

        np.testing.assert_array_equal(np.asarray(direct_loss), np.asarray(old_loss))
        np.testing.assert_array_equal(np.asarray(direct_grad), np.asarray(old_grad))

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
