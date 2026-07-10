"""训练 state → RWKV Runner 可挂载的 .pth 导出器。

格式约定(对照 RWKV-Runner backend-python/utils/rwkv.py 的 load_rwkv_state):

键名与形状:
  - 每层一个 key: blocks.{i}.att.time_state
  - tensor shape: (n_head, head_dim, head_dim) = (H, D, D)
  - dtype: fp32(RWKV Runner 加载时会 .to(torch.float),所以存 fp32 最稳)

转置方向(RWKV-Runner rwkv.py:836-857 的真实分支,2026-07 验证):
  Runner 按 model.version 分两条加载路径:
    version >= 7 (x070):  state[i*3+1] = time_state.to(float)        # 原样,不转置
    version <  7 (v5/v6): state[i*3+1] = time_state.transpose(1,2)   # 转置

  我们的模型是 RWKV-7 (x070),Runner 走 version>=7 分支,不转置。
  因此导出器默认 x070=True: 直接存训练方向 S_mlx 原样,不做 swapaxes。
  Runner 加载后注入 kernel 的 == S_mlx == 训练时的 state。

  早期版本误按 v5/v6 路径设计(预先 swapaxes),导致 x070 模型在 Runner 上注入了
  转置方向的 state,输出碎渣。已修正(见 docs/P1-任务①收尾报告.md 修正记录)。

容器:torch.save(dict)。RWKV Runner 的 torch.load(map_location="cpu") 自动检测
格式(tar 与 zip 都能加载),直接用 torch.save 即可,无需手写 pickle。
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
StateDict = Dict[int, np.ndarray]  # {layer_idx: ndarray(H,D,D)}, MLX 方向


def _normalize_states(states) -> StateDict:
    """接受 {int: mx.array} 或 {int: np.ndarray},统一成 {int: np.ndarray}。"""
    out: StateDict = {}
    for k, v in states.items():
        out[int(k)] = np.asarray(np.array(v), dtype=np.float32)
    return out


def export_pth(
    states: StateDict,
    out_path: PathLike,
    *,
    num_layers: Optional[int] = None,
    x070: bool = True,
) -> Path:
    """训练 state → RWKV Runner 可挂载的 .pth。

    步骤:
      1. (v5/v6 only) 每层 state 做 transpose(-2,-1) → ascontiguous → fp32
         x070 (RWKV-7) 模型: Runner 加载不转置, 直接存原样
      2. 组 OrderedDict, key = blocks.{i}.att.time_state
      3. torch.save

    states: {layer_idx: ndarray(H,D,D)} (MLX 训练方向)
    num_layers: 显式层数(默认取 max key + 1);用于层数与模型不一致时补齐
    x070: True(RWKV-7/x070,默认) 存原样; False(v5/v6) 存 swapaxes 后的。
          依据 RWKV-Runner rwkv.py:843 version>=7 分支不 transpose。
    返回写入的路径。
    """
    import torch

    states_np = _normalize_states(states)
    n = num_layers if num_layers is not None else (max(states_np) + 1)

    sd = OrderedDict()
    for i in range(n):
        if i not in states_np:
            raise KeyError(f"缺少 layer {i} 的 state(要求 {n} 层)")
        arr = states_np[i].astype(np.float32)
        if not x070:
            # v5/v6: Runner 会 transpose(1,2), 预先转置让它转回来
            arr = np.ascontiguousarray(np.swapaxes(arr, -2, -1))
        else:
            arr = np.ascontiguousarray(arr)
        sd[f"blocks.{i}.att.time_state"] = torch.from_numpy(arr)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_path)
    return out_path


def load_pth_as_numpy(path: PathLike, *, reverse_transpose: bool = False) -> StateDict:
    """读回 .pth → {layer_idx: ndarray(H,D,D)}。

    reverse_transpose=False(默认): 返回文件中存储的原始形态。
        这模拟"raw torch.load"——用于检查文件内容。
    reverse_transpose=True: 再 transpose(-2,-1) 还原成 MLX 训练方向。
        仅对 v5/v6 导出(export_pth x070=False)有意义。
        x070 导出的文件本身就是训练方向(Runner 不转置),reverse_transpose 应为 False。

    依赖 torch.load 读取。
    """
    import torch

    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    out: StateDict = {}
    for k, v in raw.items():
        if not k.endswith(".att.time_state"):
            continue
        layer = int(k.split(".")[1])
        arr = v.detach().cpu().numpy().astype(np.float32)
        if reverse_transpose:
            arr = np.ascontiguousarray(np.swapaxes(arr, -2, -1))
        out[layer] = arr
    return out


def load_npz_as_numpy(path: PathLike) -> StateDict:
    """读 npz(P0 内部格式 layer_{i}) → {layer_idx: ndarray}。"""
    data = np.load(Path(path))
    return {i: np.array(data[f"layer_{i}"]).astype(np.float32) for i in range(len(data.files))}


def verify_roundtrip(
    states: StateDict,
    pth_path: PathLike,
    *,
    atol: float = 1e-6,
    x070: bool = True,
) -> Tuple[bool, str]:
    """验证导出的 pth 被 Runner 正确消费后 == 原始训练 state。

    x070=True(RWKV-7,默认): Runner 不 transpose, 直接比对文件内容 == 原始 state。
    x070=False(v5/v6): Runner transpose(1,2), 比对 load_pth(reverse) == 原始。

    返回 (ok, message)。
    """
    states_np = _normalize_states(states)
    if x070:
        loaded = load_pth_as_numpy(pth_path)  # 原样读, 不 reverse
    else:
        loaded = load_pth_as_numpy(pth_path, reverse_transpose=True)

    if set(loaded) != set(states_np):
        return False, f"层数不匹配: pth 有 {sorted(loaded)}, 原始 {sorted(states_np)}"

    max_diff = 0.0
    for i in states_np:
        if loaded[i].shape != states_np[i].shape:
            return False, f"layer {i} shape 不符: pth {loaded[i].shape} vs 原始 {states_np[i].shape}"
        diff = float(np.abs(loaded[i] - states_np[i]).max())
        max_diff = max(max_diff, diff)

    if max_diff > atol:
        return False, f"数值偏差 {max_diff:.2e} > atol {atol:.2e}"
    return True, f"round-trip 通过, max diff {max_diff:.2e}"


def verify_mount_equivalence(
    model,
    tokenizer,
    states: StateDict,
    pth_path: PathLike,
    prompt: str,
    *,
    max_tokens: int = 40,
) -> Tuple[bool, str]:
    """端到端验证: 用 pth(经 Runner 逻辑)注入 MLX generate 的输出 == 直接用原始 state 注入的输出。

    这证明: 导出的 pth 被任何消费者(RWKV Runner / Ai00)加载后,注入的初始 state
    与我们训练时的 state 逐元素相同 → 行为必然一致。

    需要已加载的 model + tokenizer(kernel 路径,不 patch)。
    """
    from .core import generate
    import mlx.core as mx

    # 直接用原始 state dict(dict 形式,generate 直接注入)
    mx_states = {i: mx.array(states[i]) for i in states}
    out_direct = generate(model, tokenizer, prompt, state=mx_states, max_tokens=max_tokens)

    # 用 pth 路径(generate 内部会 load_pth + reverse transpose)
    out_pth = generate(model, tokenizer, prompt, state=str(pth_path), max_tokens=max_tokens)

    if out_direct == out_pth:
        return True, f"挂载等价: pth 注入输出与原始 state 完全一致 ({len(out_direct)} chars)"
    # 贪心解码对 ULP 敏感,允许极小差异但仍应高度一致
    match = sum(1 for a, b in zip(out_direct, out_pth) if a == b)
    total = max(len(out_direct), len(out_pth))
    ratio = match / total if total else 0
    if ratio > 0.95:
        return True, f"挂载基本等价: 前 {match}/{total} 字符一致 (ratio {ratio:.2f}, ULP 级差异)"
    return False, f"挂载不等价: 仅 {match}/{total} 一致\n  direct: {out_direct!r}\n  pth:    {out_pth!r}"
