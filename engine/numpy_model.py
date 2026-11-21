"""GPT-2 forward pass in pure NumPy (the reference implementation).

Shapes: T = sequence length, C = n_embd, H = n_head, D = C // H.
"""
from __future__ import annotations

import math

import numpy as np

# NOTE: constants are plain Python floats on purpose. Under NumPy 2 (NEP 50), a
# np.float64 scalar such as np.sqrt(2/np.pi) silently promotes float32 arrays
# to float64, which doubled memory traffic and made the forward pass ~10x slower.
_GELU_C = math.sqrt(2.0 / math.pi)


def gelu(x: np.ndarray) -> np.ndarray:
    # GPT-2 uses the tanh approximation ("gelu_new" in HF).
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # Subtract the max for numerical stability: exp never overflows.
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w + b  # w is [in, out] (GPT-2 Conv1D layout)


def ffn(x: np.ndarray, c_fc: dict, c_proj: dict) -> np.ndarray:
    return linear(gelu(linear(x, **c_fc)), **c_proj)  # [T, C] -> [T, 4C] -> [T, C]


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # q: [H, Tq, D], k/v: [H, Tk, D], mask: [Tq, Tk] additive (0 or -inf-ish)
    scores = q @ k.transpose(0, 2, 1) * (1.0 / math.sqrt(q.shape[-1])) + mask
    return softmax(scores, axis=-1) @ v


def causal_mask(t: int, dtype=np.float32) -> np.ndarray:
    # Large negative instead of -inf so softmax never sees inf - inf = nan.
    return (1 - np.tri(t, dtype=dtype)) * np.finfo(dtype).min


def mha(x: np.ndarray, c_attn: dict, c_proj: dict, n_head: int) -> np.ndarray:
    T, C = x.shape
    D = C // n_head
    qkv = linear(x, **c_attn)                              # [T, 3C]
    q, k, v = np.split(qkv, 3, axis=-1)                    # 3 x [T, C]
    # [T, C] -> [H, T, D]
    q, k, v = (a.reshape(T, n_head, D).transpose(1, 0, 2) for a in (q, k, v))
    out = attention(q, k, v, causal_mask(T, x.dtype))      # [H, T, D]
    out = out.transpose(1, 0, 2).reshape(T, C)             # merge heads
    return linear(out, **c_proj)


def block(x: np.ndarray, ln_1: dict, attn: dict, ln_2: dict, mlp: dict,
          n_head: int, eps: float) -> np.ndarray:
    # Pre-norm residual block.
    x = x + mha(layer_norm(x, **ln_1, eps=eps), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2, eps=eps), **mlp)
    return x


def gpt2(token_ids, params: dict, hparams: dict) -> np.ndarray:
    """token_ids: list/array of length T. Returns logits [T, n_vocab]."""
    ids = np.asarray(token_ids)
    T = len(ids)
    if T > hparams["n_ctx"]:
        raise ValueError(f"sequence length {T} exceeds n_ctx {hparams['n_ctx']}")
    x = params["wte"][ids] + params["wpe"][np.arange(T)]
    for blk in params["blocks"]:
        x = block(x, **blk, n_head=hparams["n_head"], eps=hparams["eps"])
    x = layer_norm(x, **params["ln_f"], eps=hparams["eps"])
    return x @ params["wte"].T  # tied embeddings: output head = wte^T


def generate(token_ids, params: dict, hparams: dict, n_new: int) -> list[int]:
    """Greedy decoding, no cache: recompute the full sequence each step."""
    ids = list(token_ids)
    for _ in range(n_new):
        logits = gpt2(ids, params, hparams)
        ids.append(int(np.argmax(logits[-1])))
    return ids[len(token_ids):]
