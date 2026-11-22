"""Sliding-window perplexity.

Targets are tokens 1..n-1 (token 0 has no context). They are scored in chunks of
`stride`. For a chunk of targets [a, b), the model reads the window ids[lo : b-1] with
lo = max(0, b-1-ctx), so the window is at most `ctx` long, and logits at window
positions (a-1-lo) .. (b-2-lo) predict targets a .. b-1.

Every target is scored exactly once. (The loop in HF's "Perplexity of fixed-length
models" guide, with stride == ctx, silently skips the first token of every window
after the first; this version doesn't.) stride == ctx: least compute. stride < ctx:
each target gets more context (lower ppl, ~ctx/stride x more compute).
"""
from __future__ import annotations

import math

import numpy as np

from engine.numpy_model import gpt2


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    return x - m - np.log(np.sum(np.exp(x - m), axis=-1, keepdims=True))


def perplexity(ids, params: dict, hparams: dict, ctx: int = 1024, stride: int = 1024,
               progress=None) -> dict:
    if not 1 <= stride <= ctx <= hparams["n_ctx"]:
        raise ValueError("need 1 <= stride <= ctx <= n_ctx")
    ids = np.asarray(ids)
    n = len(ids)
    nll, count = 0.0, 0
    chunks = list(range(1, n, stride))
    for w, a in enumerate(chunks):
        b = min(a + stride, n)
        lo = max(0, b - 1 - ctx)
        logits = gpt2(ids[lo:b - 1], params, hparams)             # [b-1-lo, V]
        logp = log_softmax(logits[a - 1 - lo:].astype(np.float64))  # rows predicting a..b-1
        nll -= float(logp[np.arange(b - a), ids[a:b]].sum())
        count += b - a
        if progress:
            progress(w + 1, len(chunks), math.exp(nll / count))
    return {"ppl": math.exp(nll / count), "nll": nll, "tokens": count}
