"""RWKV-7 WKV7 Metal checkpoint kernel —— 训练 fast path。

理论要点(见 docs/P0-理论指南.md):
  §二: 训练必须走可微路径。core.patch_rwkv7_for_train 走 _wkv7_step_ops
       Python 循环(L 次 dispatch),慢但梯度能穿透 512 步回到 S₀。
       本模块用整段 Metal kernel 替换:forward 一次 dispatch、backward 一次
       dispatch(checkpoint 重构),通过 mx.custom_function 注册 VJP。

state 语义(与 core.py 同向):
  h_in / h_out 是 (B, H, D, D),递归为
    state = state * w + v ⊗ k + sab ;  y = state @ r
  与 _wkv7_step_ops 完全一致(实验 A/B 验证:6 梯度相对误差 < 1.3e-3)。
  h_in 即注入的可训练 S₀(build_state_cache 广播到 batch),梯度能穿回 S₀。

本模块三件事:
  1. make_wkv7_checkpoint(B, T, H, D) → 工厂,按 (H,T) JIT 缓存 Metal kernel。
  2. 返回的 wkv7_train(r,w,k,v,a,b,h_in) —— 接收外部 h_in(可训练 S₀),
     forward + backward 全 Metal,梯度对 r/w/k/v/a/b/h_in 全通。
  3. wkv7_train_zero(...) —— 零 state 便捷入口,兼容参考实现原用法。

来源:移植自 rwkv-metal(Apache-2.0, Alexei Goncharov / ImpulseLeap)。
       原 make_wkv7_checkpoint 把 h_in 固化成零;本模块改为暴露给调用方,
       以支持 Preen 的可训练 S₀。kernel 本体(forward/backward Metal 源码)、
       checkpoint 反向数值稳定设计、bf16 dtype cast 均原样保留。

约束: T % CHUNK(=32) == 0。一次训练 run 内 T 固定 → kernel 只 JIT 一次。
"""
from __future__ import annotations

import mlx.core as mx

HEAD_SIZE = 64
CHUNK = 32

_fwd_cache: dict = {}
_bwd_cache: dict = {}


def _get_ckpt_fwd(H: int, T: int):
    key = (H, T)
    if key in _fwd_cache:
        return _fwd_cache[key]
    N = T // CHUNK
    hdr = f"""
    constant uint HEAD_SIZE_C = {HEAD_SIZE};
    constant uint T_C         = {T};
    constant uint CHUNK_C     = {CHUNK};
    constant uint N_CHUNKS_C  = {N};
    constant uint H_C         = {H};
    """
    src = r"""
    uint dv  = thread_position_in_grid.y;
    uint bhi = thread_position_in_grid.x;
    uint bi  = bhi / H_C, hi = bhi % H_C;
    float h_row[HEAD_SIZE_C];
    uint hb = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[hb+dk];

    for (uint c=0; c<N_CHUNKS_C; c++) {
        for (uint t=0; t<CHUNK_C; t++) {
            uint base = ((bi*T_C + c*CHUNK_C + t)*H_C + hi)*HEAD_SIZE_C;
            float sa = 0;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a[base+dk];
            sa_out[base+dv] = sa;
            float vv = v[base+dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                h_row[dk] = w[base+dk]*h_row[dk] + vv*k[base+dk] + sa*b[base+dk];
            float y = 0;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r[base+dk];
            out[base+dv] = y;
        }
        uint ckb = ((bi*H_C+hi)*N_CHUNKS_C + c)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_checkpoints[ckb+dk] = h_row[dk];
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[hb+dk] = h_row[dk];
    """
    kern = mx.fast.metal_kernel(
        name=f"wkv7_ckpt_fwd_H{H}_T{T}",
        input_names=["r", "w", "k", "v", "a", "b", "h_in"],
        output_names=["out", "h_out", "sa_out", "h_checkpoints"],
        header=hdr, source=src,
    )
    _fwd_cache[key] = kern
    return kern


def _get_ckpt_bwd(H: int, T: int):
    key = (H, T)
    if key in _bwd_cache:
        return _bwd_cache[key]
    N = T // CHUNK
    hdr = f"""
    constant uint HEAD_SIZE_C = {HEAD_SIZE};
    constant uint T_C         = {T};
    constant uint CHUNK_C     = {CHUNK};
    constant uint N_CHUNKS_C  = {N};
    constant uint H_C         = {H};
    """
    src = r"""
    uint dv  = thread_position_in_threadgroup.x;
    uint bhi = threadgroup_position_in_grid.x;
    uint bi  = bhi / H_C, hi = bhi % H_C;

    threadgroup float accum[HEAD_SIZE_C][HEAD_SIZE_C];
    threadgroup float k_sh[HEAD_SIZE_C], v_sh[HEAD_SIZE_C], r_sh[HEAD_SIZE_C];
    threadgroup float w_sh[HEAD_SIZE_C], a_sh[HEAD_SIZE_C], b_sh[HEAD_SIZE_C];
    threadgroup float dy_sh[HEAD_SIZE_C], sa_sh[HEAD_SIZE_C], dsa_sh[HEAD_SIZE_C];

    float C_row[HEAD_SIZE_C], h_row[HEAD_SIZE_C];
    uint hb = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] = d_h_out[hb+dk];

    for (int c=(int)N_CHUNKS_C-1; c>=0; c--) {
        uint ckb = ((bi*H_C+hi)*N_CHUNKS_C+(uint)c)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_ckpts[ckb+dk];

        for (int t=(int)CHUNK_C-1; t>=0; t--) {
            uint base = ((bi*T_C+(uint)c*CHUNK_C+(uint)t)*H_C+hi)*HEAD_SIZE_C;

            k_sh[dv]=k[base+dv]; v_sh[dv]=v[base+dv]; r_sh[dv]=r[base+dv];
            w_sh[dv]=w[base+dv]; a_sh[dv]=a[base+dv]; b_sh[dv]=b[base+dv];
            dy_sh[dv]=d_out[base+dv]; sa_sh[dv]=sa_fwd[base+dv];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dy_dv = dy_sh[dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] += dy_dv*r_sh[dk];

            float dsa_dv=0, dv_val=0;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
                dsa_dv += C_row[dk]*b_sh[dk];
                dv_val  += C_row[dk]*k_sh[dk];
            }
            dv_out[base+dv] = dv_val;
            dsa_sh[dv] = dsa_dv;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = dy_dv*h_row[dk];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dr_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dr_val+=accum[s][dv];
            dr_out[base+dv] = dr_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float sa_dv=sa_sh[dv], v_dv=v_sh[dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
                float hp=(h_row[dk]-v_dv*k_sh[dk]-sa_dv*b_sh[dk])/w_sh[dk];
                accum[dv][dk]=C_row[dk]*hp; h_row[dk]=hp;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dw_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dw_val+=accum[s][dv];
            dw_out[base+dv] = dw_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk]=C_row[dk]*v_dv;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dk_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dk_val+=accum[s][dv];
            dk_out[base+dv] = dk_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk]=dsa_sh[dv]*h_row[dk];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float da_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) da_val+=accum[s][dv];
            da_out[base+dv] = da_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk]=sa_sh[dv]*C_row[dk];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float db_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) db_val+=accum[s][dv];
            db_out[base+dv] = db_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                C_row[dk] = C_row[dk]*w_sh[dk] + dsa_dv*a_sh[dk];
        }
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) dh_in_out[hb+dk] = C_row[dk];
    """
    kern = mx.fast.metal_kernel(
        name=f"wkv7_ckpt_bwd_H{H}_T{T}",
        input_names=["r", "w", "k", "v", "a", "b", "h_ckpts", "sa_fwd", "d_out", "d_h_out"],
        output_names=["dr_out", "dw_out", "dk_out", "dv_out", "da_out", "db_out", "dh_in_out"],
        header=hdr, source=src, atomic_outputs=False,
    )
    _bwd_cache[key] = kern
    return kern


def make_wkv7_checkpoint(B: int, T: int, H: int, D: int = HEAD_SIZE):
    """创建 wkv7_train 函数,使用 checkpoint Metal kernel。

    B/T/H/D 在此固定 → kernel 按 (H,T) JIT 缓存,只编译一次。
    返回的 wkv7_train 接收外部 h_in(可训练 S₀),梯度对 h_in 全通。

    与 core._wkv7_train 闭包约定一致:h_in 即 (B,H,D,D) 广播后的可训练 state。
    """
    assert T % CHUNK == 0, f"T={T} 必须整除 CHUNK={CHUNK}(checkpoint kernel 约束)"
    assert D == HEAD_SIZE, f"D={D} != HEAD_SIZE={HEAD_SIZE}(kernel 硬编码)"
    N = T // CHUNK

    @mx.custom_function
    def _fwd(r, w, k, v, a, b, h_in):
        res = _get_ckpt_fwd(H, T)(
            inputs=[x.astype(mx.float32) for x in [r, w, k, v, a, b, h_in]],
            grid=(B * H, D, 1), threadgroup=(1, 1, 1),
            output_shapes=[(B, T, H, D), (B, H, D, D), (B, T, H, D), (B, H, N, D, D)],
            output_dtypes=[mx.float32] * 4,
        )
        return res[0], res[1], res[2], res[3]

    @_fwd.vjp
    def _vjp(primals, cotangents, outputs):
        r, w, k, v, a, b, h_in = primals
        d_out, d_h_out, _, _ = cotangents
        _, _, sa_fwd, h_ckpts = outputs
        res = _get_ckpt_bwd(H, T)(
            inputs=[x.astype(mx.float32) for x in [r, w, k, v, a, b, h_ckpts, sa_fwd, d_out, d_h_out]],
            grid=(B * H * D, 1, 1), threadgroup=(D, 1, 1),
            output_shapes=[(B, T, H, D)] * 6 + [(B, H, D, D)],
            output_dtypes=[mx.float32] * 7,
        )
        grads = [res[0], res[1], res[2], res[3], res[4], res[5], res[6]]
        return [g.astype(p.dtype) for g, p in zip(grads, primals)]

    def wkv7_train(r, w, k, v, a, b, h_in):
        """整段 WKV7 前向(可微)。h_in = 可训练 S₀,广播到 (B,H,D,D)。

        返回 (out, h_out):out=(B,T,H,D) 为每步 y;h_out=(B,H,D,D) 为递归末态。
        训练下游通常只取 out(h_out 供 _wkv7 契约返回 new_state)。
        """
        out, h_out, _, _ = _fwd(r, w, k, v, a, b, h_in)
        return out, h_out

    return wkv7_train


def wkv7_train_zero(r, w, k, v, a, b, *, num_heads: int, head_dim: int):
    """零 state 便捷入口(无需预建工厂)。推理/无 S₀ 训练用。

    每次按 (B,T,H,D) 查/建 kernel。head_dim 必须为 64(kernel 硬编码)。
    """
    B, T, H, D = r.shape
    assert H == num_heads and D == head_dim, f"形状不匹配: got (H={H},D={D}) expect ({num_heads},{head_dim})"
    wkv7_train = make_wkv7_checkpoint(B, T, H, D)
    h0 = mx.zeros((B, H, D, D), dtype=r.dtype)
    out, _ = wkv7_train(r, w, k, v, a, b, h0)
    return out
