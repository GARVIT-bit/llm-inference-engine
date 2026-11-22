"""Fused causal attention in Triton (FlashAttention-style forward pass).

Naive attention materialises the full [T, T] score matrix per head: softmax(q k^T) v.
That is O(T^2) memory traffic, and at T = 4096 the scores alone are 32 MB per head in
fp16. This kernel never writes the scores to GPU memory. Each program handles one
BLOCK_M-row tile of queries for one (batch, head):

    for each BLOCK_N tile of keys/values (stopping at the diagonal when causal):
        s     = q_tile @ k_tile^T * scale                    (on-chip, fp32)
        m_new = max(m, rowmax(s))                            running row max
        p     = exp(s - m_new)
        l     = l * exp(m - m_new) + rowsum(p)               running softmax denominator
        acc   = acc * exp(m - m_new) + p @ v_tile            rescale old partial output
        m     = m_new
    out = acc / l

This is the "online softmax": the result equals softmax over the whole row, computed
one tile at a time. `tiled_attention_reference` below runs the same loop in NumPy so
the algorithm is testable on any machine (Triton needs Linux + an NVIDIA GPU).
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except Exception:            # ImportError on Windows / CPU-only machines
    HAS_TRITON = False

LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------- NumPy mirror

def naive_attention(q, k, v, causal: bool = True):
    """softmax(q k^T / sqrt(D) + mask) v for [..., T, D] arrays (the definition)."""
    T, S = q.shape[-2], k.shape[-2]
    s = q @ np.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    if causal:
        s = np.where(np.tri(T, S, k=S - T, dtype=bool), s, -np.inf)
    s = np.exp(s - s.max(-1, keepdims=True))
    return (s / s.sum(-1, keepdims=True)) @ v


def tiled_attention_reference(q, k, v, causal: bool = True, block_m: int = 64, block_n: int = 64):
    """Exactly the kernel's algorithm (tiles, exp2, running max/sum), one (b, h) at a time."""
    *lead, T, D = q.shape
    q2, k2, v2 = (a.reshape(-1, T, D) for a in (q, k, v))
    out = np.empty_like(q2)
    scale = LOG2E / math.sqrt(D)          # fold log2(e) in so the kernel can use exp2
    for bh in range(q2.shape[0]):
        for m0 in range(0, T, block_m):
            rows = np.arange(m0, min(m0 + block_m, T))
            qt = q2[bh, rows]
            m_i = np.full(len(rows), -np.inf)
            l_i = np.zeros(len(rows))
            acc = np.zeros((len(rows), D))
            hi = min(m0 + block_m, T) if causal else T
            for n0 in range(0, hi, block_n):
                cols = np.arange(n0, min(n0 + block_n, T))
                s = (qt @ k2[bh, cols].T) * scale
                if causal:
                    s = np.where(rows[:, None] >= cols[None, :], s, -np.inf)
                m_new = np.maximum(m_i, s.max(1))
                p = np.exp2(s - m_new[:, None])
                alpha = np.exp2(m_i - m_new)
                l_i = l_i * alpha + p.sum(1)
                acc = acc * alpha[:, None] + p @ v2[bh, cols]
                m_i = m_new
            out[bh, rows] = acc / l_i[:, None]
    return out.reshape(q.shape)


# ---------------------------------------------------------------------- Triton kernel

if HAS_TRITON:
    @triton.jit
    def _attn_fwd(Q, K, V, O, qk_scale,
                  sqb, sqh, sqt, sqd, skb, skh, skt, skd,
                  svb, svh, svt, svd, sob, soh, sot, sod,
                  H, T,
                  D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  CAUSAL: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)

        q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqt + offs_d[None, :] * sqd,
                    mask=offs_m[:, None] < T, other=0.0)
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        hi = T
        if CAUSAL:
            hi = tl.minimum((pid_m + 1) * BLOCK_M, T)
        for n0 in range(0, hi, BLOCK_N):
            cols = n0 + offs_n
            kt = tl.load(K + b * skb + h * skh + cols[None, :] * skt + offs_d[:, None] * skd,
                         mask=cols[None, :] < T, other=0.0)                     # [D, BLOCK_N]
            s = tl.dot(q, kt) * qk_scale                                        # [BLOCK_M, BLOCK_N] fp32
            valid = cols[None, :] < T
            if CAUSAL:
                valid = valid & (offs_m[:, None] >= cols[None, :])
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp2(s - m_new[:, None])
            alpha = tl.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            vt = tl.load(V + b * svb + h * svh + cols[:, None] * svt + offs_d[None, :] * svd,
                         mask=cols[:, None] < T, other=0.0)                     # [BLOCK_N, D]
            acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
            m_i = m_new
        # rows past T (padding in the last tile) are NaN here but never stored
        acc = acc / l_i[:, None]
        tl.store(O + b * sob + h * soh + offs_m[:, None] * sot + offs_d[None, :] * sod,
                 acc.to(O.dtype.element_ty), mask=offs_m[:, None] < T)


def triton_attention(q, k, v, causal: bool = True, block_m: int = 64, block_n: int = 32,
                     num_warps: int = 8, num_stages: int = 1):
    """q, k, v: [B, H, T, D] CUDA tensors (fp16/bf16/fp32), same T. Returns [B, H, T, D].

    Defaults come from bench/attention_tune.py on a T4 (D=64): the original 64x64 tiles
    with 4 warps spilled 1,448 registers and ran 12x slower than this config, which has
    no spills. Tuned for D=64; D=128 is tested for correctness only.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton attention needs the triton package and a CUDA GPU")
    B, H, T, D = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, v must have the same [B, H, T, D] shape")
    if D not in (16, 32, 64, 128):
        raise ValueError("head dim must be 16, 32, 64 or 128")
    o = torch.empty_like(q)
    grid = (triton.cdiv(T, block_m), B * H)
    global LAST_COMPILED
    LAST_COMPILED = _attn_fwd[grid](q, k, v, o, LOG2E / math.sqrt(D),
                                    *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                                    H, T, D=D, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal,
                                    num_warps=num_warps, num_stages=num_stages)
    return o


LAST_COMPILED = None   # compiled kernel of the last launch (for bench/attention_tune.py diagnostics)


def compiled_info(ck) -> dict:
    """Registers, spills and whether the PTX uses tensor-core MMA instructions."""
    info = {}
    for name in ("n_regs", "n_spills"):
        try:
            info[name] = getattr(ck, name)
        except Exception:
            info[name] = None
    try:
        ptx = ck.asm["ptx"]
        info["ptx_mma"] = ptx.count("mma.sync")
        info["ptx_fma_f32"] = ptx.count("fma.rn.f32")
        info["ptx_ld_local"] = ptx.count("ld.local")      # register spills read back
    except Exception:
        pass
    return info
