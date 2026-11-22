"""Correctness ladder: cached == uncached (logits and greedy tokens)."""
import numpy as np
import pytest

from engine.numpy_model import KVCache, causal_mask, generate, gpt2
from engine.tokenizer import encode

PROMPT = "Alan Turing theorized that computers would one day become"


def test_causal_mask_with_offset():
    # 2 new queries at positions 3, 4 of a 5-long sequence
    m = causal_mask(2, 5)
    allowed = m == 0
    np.testing.assert_array_equal(allowed, [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    np.testing.assert_array_equal(causal_mask(4), causal_mask(4, 4))


def test_cache_shape(gpt2_small):
    hp, _ = gpt2_small
    c = KVCache(hp)
    assert c.k.shape == (12, 12, 1024, 64)
    assert c.nbytes == 2 * 12 * 12 * 1024 * 64 * 4  # 75.5 MB fp32 for a full 1024 context


def test_prefill_logits_equal_uncached(gpt2_small):
    hp, p = gpt2_small
    ids = encode(PROMPT)
    cache = KVCache(hp)
    np.testing.assert_allclose(gpt2(ids, p, hp, cache=cache), gpt2(ids, p, hp), atol=1e-4)
    assert cache.length == len(ids)


def test_decode_logits_equal_uncached_at_every_step(gpt2_small):
    """Feed one token at a time through the cache; each step's logits must equal
    the last row of a full uncached forward over the same prefix."""
    hp, p = gpt2_small
    ids = encode(PROMPT + " the most powerful machines on the planet.")
    cache = KVCache(hp)
    gpt2(ids[:3], p, hp, cache=cache)
    full = gpt2(ids, p, hp)
    for t in range(3, len(ids)):
        step = gpt2([ids[t]], p, hp, cache=cache)
        np.testing.assert_allclose(step[0], full[t], atol=1e-3, rtol=1e-5)
        assert int(step[0].argmax()) == int(full[t].argmax())
    assert cache.length == len(ids)


def test_chunked_prefill_equals_single_prefill(gpt2_small):
    hp, p = gpt2_small
    ids = encode(PROMPT)
    a, b = KVCache(hp), KVCache(hp)
    la = gpt2(ids, p, hp, cache=a)
    gpt2(ids[:4], p, hp, cache=b)
    lb = gpt2(ids[4:], p, hp, cache=b)
    np.testing.assert_allclose(la[4:], lb, atol=1e-3, rtol=1e-5)
    np.testing.assert_allclose(a.k[:, :, :len(ids)], b.k[:, :, :len(ids)], atol=1e-4)


def test_last_only_matches_full(gpt2_small):
    hp, p = gpt2_small
    ids = encode(PROMPT)
    # One row goes through BLAS gemv instead of gemm: same math, fp32 rounding ~1e-6 relative.
    a, b = gpt2(ids, p, hp, last_only=True)[0], gpt2(ids, p, hp)[-1]
    np.testing.assert_allclose(a, b, atol=1e-3, rtol=1e-5)
    assert a.argmax() == b.argmax()


@pytest.mark.parametrize("prompt,n", [(PROMPT, 30), ("Hello", 20), ("def add(a, b):\n   ", 25)])
def test_cached_tokens_equal_uncached_tokens(gpt2_small, prompt, n):
    hp, p = gpt2_small
    ids = encode(prompt)
    assert generate(ids, p, hp, n, use_cache=True) == generate(ids, p, hp, n, use_cache=False)


def test_cache_overflow_raises(gpt2_small):
    hp, p = gpt2_small
    cache = KVCache(hp)
    cache.length = hp["n_ctx"]
    with pytest.raises(ValueError):
        gpt2([0], p, hp, cache=cache)
