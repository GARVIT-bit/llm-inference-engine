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


def causal_mask(tq: int, tk: int | None = None, dtype=np.float32) -> np.ndarray:
    """Additive mask [tq, tk] for tq new queries at the END of a tk-long sequence.

    Query i sits at absolute position (tk - tq + i) and may see keys 0..that position.
    Large negative instead of -inf so softmax never sees inf - inf = nan.
    """
    tk = tq if tk is None else tk
    return (1 - np.tri(tq, tk, k=tk - tq, dtype=dtype)) * np.finfo(dtype).min


class KVCache:
    """Contiguous per-sequence KV cache, preallocated to n_ctx.

    k, v: [n_layer, n_head, n_ctx, head_dim]. `length` = tokens already stored.
    (Phase 4 replaces this with fixed-size blocks + block tables.)
    """

    def __init__(self, hparams: dict, dtype=np.float32):
        H = hparams["n_head"]
        shape = (hparams["n_layer"], H, hparams["n_ctx"], hparams["n_embd"] // H)
        self.k = np.zeros(shape, dtype)
        self.v = np.zeros(shape, dtype)
        self.length = 0
        self.n_ctx = hparams["n_ctx"]

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


def mha(x: np.ndarray, c_attn: dict, c_proj: dict, n_head: int,
        cache: KVCache | None = None, layer: int = 0) -> np.ndarray:
    T, C = x.shape
    D = C // n_head
    qkv = linear(x, **c_attn)                              # [T, 3C]
    q, k, v = np.split(qkv, 3, axis=-1)                    # 3 x [T, C]
    # [T, C] -> [H, T, D]
    q, k, v = (a.reshape(T, n_head, D).transpose(1, 0, 2) for a in (q, k, v))
    if cache is not None:
        # Store the new K/V after the ones already cached, attend over all of them.
        s = cache.length
        cache.k[layer, :, s:s + T] = k
        cache.v[layer, :, s:s + T] = v
        k = cache.k[layer, :, :s + T]
        v = cache.v[layer, :, :s + T]
    out = attention(q, k, v, causal_mask(T, k.shape[1], x.dtype))  # [H, T, D]
    out = out.transpose(1, 0, 2).reshape(T, C)             # merge heads
    return linear(out, **c_proj)


def block(x: np.ndarray, ln_1: dict, attn: dict, ln_2: dict, mlp: dict,
          n_head: int, eps: float, cache: KVCache | None = None, layer: int = 0) -> np.ndarray:
    # Pre-norm residual block.
    x = x + mha(layer_norm(x, **ln_1, eps=eps), **attn, n_head=n_head, cache=cache, layer=layer)
    x = x + ffn(layer_norm(x, **ln_2, eps=eps), **mlp)
    return x


def gpt2(token_ids, params: dict, hparams: dict, cache: KVCache | None = None,
         last_only: bool = False) -> np.ndarray:
    """Logits for `token_ids`.

    Without a cache: token_ids is the whole sequence, returns [T, n_vocab].
    With a cache: token_ids are the NEW tokens only (whole prompt on prefill, one
    token on decode); their positions start at cache.length; K/V are appended.
    last_only=True computes the output head for the final position only ([1, n_vocab]),
    which is all generation needs (the head is 50257x768, the biggest matmul).
    """
    ids = np.asarray(token_ids)
    T = len(ids)
    start = cache.length if cache is not None else 0
    if start + T > hparams["n_ctx"]:
        raise ValueError(f"sequence length {start + T} exceeds n_ctx {hparams['n_ctx']}")
    x = params["wte"][ids] + params["wpe"][np.arange(start, start + T)]
    for i, blk in enumerate(params["blocks"]):
        x = block(x, **blk, n_head=hparams["n_head"], eps=hparams["eps"], cache=cache, layer=i)
    if cache is not None:
        cache.length += T
    if last_only:
        x = x[-1:]
    x = layer_norm(x, **params["ln_f"], eps=hparams["eps"])
    return x @ params["wte"].T  # tied embeddings: output head = wte^T


def generate(token_ids, params: dict, hparams: dict, n_new: int,
             use_cache: bool = False) -> list[int]:
    """Greedy decoding.

    use_cache=False: re-run the full, growing sequence every step (O(T) work per token).
    use_cache=True : prefill the prompt once, then feed only the newest token.
    """
    ids = list(token_ids)
    if not use_cache:
        for _ in range(n_new):
            logits = gpt2(ids, params, hparams, last_only=True)
            ids.append(int(np.argmax(logits[-1])))
        return ids[len(token_ids):]

    cache = KVCache(hparams, dtype=params["wte"].dtype)
    logits = gpt2(ids, params, hparams, cache=cache, last_only=True)   # prefill
    for i in range(n_new):
        ids.append(int(np.argmax(logits[-1])))
        if i + 1 < n_new:
            logits = gpt2([ids[-1]], params, hparams, cache=cache, last_only=True)  # decode
    return ids[len(token_ids):]
