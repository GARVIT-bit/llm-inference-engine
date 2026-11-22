"""GPT-2 in PyTorch: the same math as engine/numpy_model.py, on CPU (fp32) or GPU (fp16).

Two entry points:
  forward_dense(ids)            no cache, one sequence -> logits [T, V] (the reference)
  forward_batch(seqs, cache)    many sequences at once over a paged KV cache: each one
                                brings its NEW tokens (a whole prompt on prefill, one
                                token on decode) and its block table.

Attention is written out by hand (matmuls + softmax); no torch SDPA, no nn.Module zoo.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from engine.kv_cache import BlockTable, PagedKVCache

_GELU_C = math.sqrt(2.0 / math.pi)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_dtype(device) -> torch.dtype:
    return torch.float16 if str(device).startswith("cuda") else torch.float32


def _to_torch(obj, device, dtype):
    if isinstance(obj, dict):
        return {k: _to_torch(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_torch(v, device, dtype) for v in obj]
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).to(device=device, dtype=dtype)
    raise TypeError(f"unexpected param type {type(obj)} (int8 params are NumPy-only)")


def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(_GELU_C * (x + 0.044715 * x ** 3)))


def layer_norm(x, g, b, eps: float):
    # Statistics in fp32 even when the model runs in fp16 (variance underflows otherwise).
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = ((xf - mean) ** 2).mean(-1, keepdim=True)
    return (g.float() * (xf - mean) / torch.sqrt(var + eps) + b.float()).to(x.dtype)


def linear(x, w, b):
    return torch.addmm(b, x, w)       # x @ w + b, w is [in, out] (Conv1D layout)


def lm_head(x, wte):
    """logits = x @ wte^T (tied embeddings). Computed as (wte @ x^T)^T: on CPU MKL this is
    ~2.7x faster for a few rows than multiplying by the transposed 50257 x 768 matrix."""
    return (wte @ x.T).T.contiguous()


def softmax(x, dim: int = -1):
    # Stable softmax, in fp32 for fp16 inputs.
    xf = x.float()
    e = torch.exp(xf - xf.amax(dim, keepdim=True))
    return (e / e.sum(dim, keepdim=True)).to(x.dtype)


class TorchGPT2:
    def __init__(self, params: dict, hparams: dict, device=None, dtype=None, attention: str = "torch"):
        self.device = torch.device(device or default_device())
        self.dtype = dtype or default_dtype(self.device)
        self.hp = hparams
        self.p = _to_torch(params, self.device, self.dtype)
        self.n_head = hparams["n_head"]
        self.eps = hparams["eps"]
        self.neg = torch.finfo(self.dtype).min
        if attention not in ("torch", "triton"):
            raise ValueError(attention)
        if attention == "triton":
            from kernels.attention_triton import HAS_TRITON
            if not HAS_TRITON or self.device.type != "cuda":
                raise RuntimeError("attention='triton' needs Triton and a CUDA device")
        self.attention = attention   # used by forward_dense (prefill-style, one sequence)

    def new_cache(self, num_blocks: int, block_size: int = 16) -> PagedKVCache:
        return PagedKVCache(self.hp, num_blocks, block_size, self.device, self.dtype)

    # ------------------------------------------------------------------ dense reference

    @torch.inference_mode()
    def forward_dense(self, ids) -> torch.Tensor:
        p, H = self.p, self.n_head
        ids = torch.as_tensor(ids, device=self.device)
        T = ids.shape[0]
        if T > self.hp["n_ctx"]:
            raise ValueError(f"sequence length {T} exceeds n_ctx {self.hp['n_ctx']}")
        C = p["wte"].shape[1]
        D = C // H
        x = p["wte"][ids] + p["wpe"][:T]
        mask = torch.triu(torch.full((T, T), self.neg, device=self.device, dtype=self.dtype), 1)
        for blk in p["blocks"]:
            h = layer_norm(x, **blk["ln_1"], eps=self.eps)
            q, k, v = linear(h, **blk["attn"]["c_attn"]).split(C, dim=-1)
            q, k, v = (t.view(T, H, D).transpose(0, 1) for t in (q, k, v))   # [H, T, D]
            if self.attention == "triton":
                from kernels.attention_triton import triton_attention
                att = triton_attention(q[None], k[None], v[None], causal=True)[0]
            else:
                att = softmax(q @ k.transpose(1, 2) * (1.0 / math.sqrt(D)) + mask) @ v
            x = x + linear(att.transpose(0, 1).reshape(T, C), **blk["attn"]["c_proj"])
            h = layer_norm(x, **blk["ln_2"], eps=self.eps)
            x = x + linear(gelu(linear(h, **blk["mlp"]["c_fc"])), **blk["mlp"]["c_proj"])
        x = layer_norm(x, **p["ln_f"], eps=self.eps)
        return lm_head(x, p["wte"])

    # ------------------------------------------------------------------ batched + paged

    @torch.inference_mode()
    def forward_batch(self, seqs: list[tuple[list[int], BlockTable]], cache: PagedKVCache,
                      all_logits: bool = False):
        """One forward pass for a batch of sequences sharing the paged cache.

        seqs: (new_token_ids, block_table) per sequence. The new tokens take positions
        table.length .. table.length + T - 1; the table must already have blocks for
        them (cache.reserve). K/V are written into the pool, attention reads each
        sequence's whole history through its block table, and table.length advances.

        Returns logits of each sequence's LAST new token, [B, V] (what sampling needs),
        or with all_logits=True a list of [T_b, V] tensors.

        All new tokens are flattened into one [N, C] matrix for the linear layers (one
        matmul for the whole batch). Attention pads to [B, Tmax] queries x [B, Lmax]
        keys and masks out padding and the future.
        """
        p, H, dev = self.p, self.n_head, self.device
        C = p["wte"].shape[1]
        D = C // H
        B = len(seqs)
        n_ctx = self.hp["n_ctx"]

        # ---- per-step index bookkeeping (plain Python, then one tensor each)
        flat_ids, positions, slots, dst, last = [], [], [], [], []
        Ts = [len(t) for t, _ in seqs]
        starts = [tab.length for _, tab in seqs]
        Tmax, Lmax = max(Ts), max(s + t for s, t in zip(starts, Ts))
        if Lmax > n_ctx:
            raise ValueError(f"sequence length {Lmax} exceeds n_ctx {n_ctx}")
        gather = torch.zeros((B, Lmax), dtype=torch.long)        # padded with slot 0 (masked)
        for b, ((toks, tab), s, T) in enumerate(zip(seqs, starts, Ts)):
            if T == 0:
                raise ValueError("every sequence needs at least one new token")
            flat_ids += toks
            positions += range(s, s + T)
            slots += cache.slots(tab, s, s + T)
            dst += range(b * Tmax, b * Tmax + T)
            last.append(len(flat_ids) - 1)
            gather[b, :s + T] = torch.tensor(cache.slots(tab, 0, s + T))
        to = dict(device=dev)
        flat_ids = torch.tensor(flat_ids, **to)
        positions = torch.tensor(positions, **to)
        slots = torch.tensor(slots, **to)
        dst = torch.tensor(dst, **to)
        gather = gather.view(-1).to(dev)
        # query i of sequence b sits at position starts[b] + i and may see keys j <= that
        qpos = torch.tensor(starts, **to)[:, None] + torch.arange(Tmax, **to)[None, :]   # [B, Tmax]
        allowed = torch.arange(Lmax, **to)[None, None, :] <= qpos[:, :, None]           # [B, Tmax, Lmax]
        mask = torch.zeros(allowed.shape, device=dev, dtype=self.dtype).masked_fill_(~allowed, self.neg)
        mask = mask[:, None]                                                            # [B, 1, Tmax, Lmax]
        N = flat_ids.shape[0]
        scale = 1.0 / math.sqrt(D)

        x = p["wte"][flat_ids] + p["wpe"][positions]                                    # [N, C]
        for layer, blk in enumerate(p["blocks"]):
            h = layer_norm(x, **blk["ln_1"], eps=self.eps)
            q, k, v = linear(h, **blk["attn"]["c_attn"]).split(C, dim=-1)
            kf = cache.k[layer].view(-1, H, D)                   # [num_blocks*bs, H, D]
            vf = cache.v[layer].view(-1, H, D)
            kf.index_copy_(0, slots, k.reshape(N, H, D))
            vf.index_copy_(0, slots, v.reshape(N, H, D))
            # read each sequence's history through its block table: [B, H, Lmax, D]
            # (a copy; a paged-attention kernel would read the blocks in place instead)
            K = kf.index_select(0, gather).view(B, Lmax, H, D).permute(0, 2, 1, 3).contiguous()
            V = vf.index_select(0, gather).view(B, Lmax, H, D).permute(0, 2, 1, 3).contiguous()
            Q = torch.zeros((B * Tmax, H, D), device=dev, dtype=self.dtype)
            Q[dst] = q.reshape(N, H, D)
            Q = Q.view(B, Tmax, H, D).permute(0, 2, 1, 3)        # [B, H, Tmax, D]
            att = softmax(Q @ K.transpose(-1, -2) * scale + mask) @ V   # [B, H, Tmax, D]
            att = att.permute(0, 2, 1, 3).reshape(B * Tmax, C)[dst]     # back to [N, C]
            x = x + linear(att, **blk["attn"]["c_proj"])
            h = layer_norm(x, **blk["ln_2"], eps=self.eps)
            x = x + linear(gelu(linear(h, **blk["mlp"]["c_fc"])), **blk["mlp"]["c_proj"])

        for (toks, tab) in seqs:
            tab.length += len(toks)
        if all_logits:
            x = layer_norm(x, **p["ln_f"], eps=self.eps)
            return list(lm_head(x, p["wte"]).split(Ts))
        x = layer_norm(x[torch.tensor(last, device=dev)], **p["ln_f"], eps=self.eps)
        return lm_head(x, p["wte"])                              # [B, V]
