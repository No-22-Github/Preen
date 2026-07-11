"""实验 A: MLX(kernel) vs rwkv pip(RWKV_x070) 双实现 state 方向裁决。

对每句中文前缀，两边各以"零初始 state"逐 token forward，取全部 24 层末态
att_kv state(shape (H,N,N)=(16,64,64))，比较:
  err_direct     = |mlx_S - rwkv_S|          (原样)
  err_transposed = |mlx_S.T - rwkv_S|        (转置)

裁决: 哪个方向的 max/mean 误差小一个数量级以上，即为正确方向。
预期 bf16(mlx) vs fp32(rwkv) 在"匹配"方向误差 1e-2~1e-3 量级。

关键对齐: 两边 tokenizer 必须产出同一 token 序列(RWKV-7 World 词表)。
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# ── rwkv pip 侧: 必须先设环境变量再 import ──────────────────────────────
os.environ["RWKV_V7_ON"] = "1"  # x070 专用类(见 C2 报告)
import numpy as np
import torch

N_LAYER = 24
N_HEAD = 16
HEAD_DIM = 64

RWKV_NATIVE = str(REPO / "models" / "rwkv7-g1d-0.4b-20260210-ctx8192")
MLX_MODEL = str(REPO / "models" / "converted" / "rwkv7-g1d-0.4b")

# 3 句(取自 train_data/translate/test_10.jsonl 前 3 条 User 段)
GOLDEN_3 = [
    "今天下午三点开会，别忘了带上项目文档。",
    "这家餐厅的招牌菜是红烧肉，味道非常地道。",
    "由于连续降雨，部分地区出现了轻微内涝，请居民注意出行安全。",
]


def run_rwkv_pip():
    """rwkv pip RWKV_x070: 零 state forward, 取每层 state[i*3+1]。"""
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE

    print(f"[rwkv] 类={RWKV.__name__}, 加载 {RWKV_NATIVE} (cpu fp32)", flush=True)
    model = RWKV(RWKV_NATIVE, "cpu fp32")
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    tokenizer = pipeline.tokenizer  # TRIE_TOKENIZER

    per_sent = []  # [{layer: np.ndarray(16,64,64)}, ...]
    for s in GOLDEN_3:
        ids = tokenizer.encode(s)
        assert len(ids) > 0, f"编码为空: {s}"
        state = None  # 零初始 state
        # 逐 token forward(state 在模型内累积)
        out, state = model.forward(ids, state)
        assert len(state) == N_LAYER * 3, f"state len={len(state)} 预期{N_LAYER*3}"
        layers = {}
        for i in range(N_LAYER):
            layers[i] = state[i * 3 + 1].detach().cpu().numpy().astype(np.float32)
        per_sent.append({"ids": ids, "layers": layers})
        print(f"[rwkv] {s[:18]}... -> {len(ids)} tokens, state[1] shape {layers[0].shape}", flush=True)
    return per_sent, tokenizer


def run_mlx():
    """MLX kernel 路径(不 patch): 零 state forward, 取每层 cache[1]。"""
    import mlx.core as mx
    from statetuner.core import load_model

    print(f"[mlx] 加载 {MLX_MODEL} (kernel 路径, patch=False)", flush=True)
    model, tokenizer = load_model(MLX_MODEL, patch=False)

    per_sent = []
    for s in GOLDEN_3:
        ids = tokenizer.encode(s)
        assert len(ids) > 0, f"MLX 编码为空: {s}"
        caches = model.make_cache()  # 零初始 state
        input_ids = mx.array([ids])
        logits = model(input_ids, caches)
        mx.eval(logits)
        layers = {}
        for i in range(N_LAYER):
            # cache[1] = att_kv state, shape (1,H,D,D) bf16。
            # mx bf16 -> numpy: 先 mx.astype(float32) 再 np.array(避免 buffer 协议冲突)
            mx_state = caches[i][1].astype(mx.float32)
            s_arr = np.array(mx_state)[0]  # 去 batch 维 -> (H,D,D)
            layers[i] = s_arr
        per_sent.append({"ids": ids, "layers": layers})
        print(f"[mlx] {s[:18]}... -> {len(ids)} tokens, state shape {layers[0].shape}", flush=True)
    return per_sent, tokenizer


def compare(rwkv_per, mlx_per):
    """逐句逐层算 direct / transposed 误差。返回全量结果 + 汇总。"""
    assert len(rwkv_per) == len(mlx_per) == len(GOLDEN_3)
    rows = []  # (sent, layer, dir_max, dir_mean, tr_max, tr_mean)
    agg = {"direct": [], "transposed": []}
    for si in range(len(GOLDEN_3)):
        r_ids = rwkv_per[si]["ids"]
        m_ids = mlx_per[si]["ids"]
        # 先报 token 对齐情况(供报告说明)
        for li in range(N_LAYER):
            rs = rwkv_per[si]["layers"][li]   # (16,64,64)
            ms = mlx_per[si]["layers"][li]    # (16,64,64)
            assert rs.shape == ms.shape == (N_HEAD, HEAD_DIM, HEAD_DIM), \
                f"shape 不符 sent{si} layer{li}: {rs.shape} vs {ms.shape}"
            d = np.abs(ms - rs)
            t = np.abs(ms.swapaxes(-2, -1) - rs)
            d_max, d_mean = float(d.max()), float(d.mean())
            t_max, t_mean = float(t.max()), float(t.mean())
            rows.append((si, li, d_max, d_mean, t_max, t_mean))
            agg["direct"].append((d_max, d_mean))
            agg["transposed"].append((t_max, t_mean))
    return rows, agg


def main():
    print("=" * 70)
    print("实验 A: MLX(kernel) vs rwkv pip(RWKV_x070) state 方向裁决")
    print("=" * 70)

    rwkv_per, rwkv_tok = run_rwkv_pip()
    mlx_per, mlx_tok = run_mlx()

    # ── tokenizer 对齐校验(铁律: 两边必须同 token 序列) ──────────────────
    print("\n--- tokenizer 对齐校验 ---")
    all_aligned = True
    for si, s in enumerate(GOLDEN_3):
        r_ids = rwkv_per[si]["ids"]
        m_ids = mlx_per[si]["ids"]
        ok = r_ids == m_ids
        all_aligned &= ok
        print(f"  sent{si}: rwkv={len(r_ids)}tok mlx={len(m_ids)}tok 对齐={ok}")
        if not ok:
            print(f"    rwkv ids[:8]: {r_ids[:8]}")
            print(f"    mlx  ids[:8]: {m_ids[:8]}")
    print(f"  => 全部对齐: {all_aligned}")

    rows, agg = compare(rwkv_per, mlx_per)

    # ── 全量表(24 层 × 3 句) ───────────────────────────────────────────
    print("\n--- 全量误差表 (sent, layer, direct_max, direct_mean, trans_max, trans_mean) ---")
    print(f"{'s':>2} {'ly':>3} {'dir_max':>11} {'dir_mean':>11} {'tr_max':>11} {'tr_mean':>11}")
    for (si, li, dm, dme, tm, tme) in rows:
        print(f"{si:>2} {li:>3} {dm:>11.4e} {dme:>11.4e} {tm:>11.4e} {tme:>11.4e}")

    # ── 汇总 ──────────────────────────────────────────────────────────
    d_maxes = [x[0] for x in agg["direct"]]
    t_maxes = [x[0] for x in agg["transposed"]]
    d_mean_of_mean = np.mean([x[1] for x in agg["direct"]])
    t_mean_of_mean = np.mean([x[1] for x in agg["transposed"]])
    print("\n--- 汇总(72 个 layer×sent) ---")
    print(f"  direct:     max范围[{min(d_maxes):.3e},{max(d_maxes):.3e}]  mean均值={d_mean_of_mean:.3e}")
    print(f"  transposed: max范围[{min(t_maxes):.3e},{max(t_maxes):.3e}]  mean均值={t_mean_of_mean:.3e}")

    ratio = t_mean_of_mean / d_mean_of_mean if d_mean_of_mean > 0 else float("inf")
    verdict = "原样匹配(direct)" if d_mean_of_mean < t_mean_of_mean else "转置匹配(transposed)"
    print(f"\n  direct_mean / transposed_mean = {d_mean_of_mean:.3e} / {t_mean_of_mean:.3e}")
    print(f"  => 方向裁决: {verdict}")
    print(f"  (两者比值 trans/direct = {ratio:.2f}; 显著差异则裁决可信)")

    # ── 存 json 供报告引用 ─────────────────────────────────────────────
    out = REPO / "experiments" / "p0_translate" / "exp_a_direction_result.json"
    out.write_text(json.dumps({
        "golden_3": GOLDEN_3,
        "tokenizer_aligned": all_aligned,
        "rows": [{"s": r[0], "ly": r[1], "dir_max": r[2], "dir_mean": r[3],
                  "tr_max": r[4], "tr_mean": r[5]} for r in rows],
        "summary": {
            "direct_mean_of_mean": float(d_mean_of_mean),
            "transposed_mean_of_mean": float(t_mean_of_mean),
            "direct_max_range": [float(min(d_maxes)), float(max(d_maxes))],
            "transposed_max_range": [float(min(t_maxes)), float(max(t_maxes))],
            "verdict": verdict,
        },
    }, ensure_ascii=False, indent=2))
    print(f"\n[结果已存] {out}")


if __name__ == "__main__":
    main()
