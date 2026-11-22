"""Correctness ladder rungs: torch vs NumPy, and paged/batched torch vs dense torch."""
import numpy as np
import pytest
import torch

from engine.kv_cache import BlockTable
from engine.numpy_model import generate as np_generate
from engine.numpy_model import gpt2 as np_gpt2
from engine.tokenizer import encode
from engine.torch_model import TorchGPT2
from engine.weights import load_gpt2

PROMPTS = [
    "Alan Turing theorized that computers would one day become",
    "Hello",
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
]


@pytest.fixture(scope="module")
def tm(gpt2_small):
    hp, p = gpt2_small
    return TorchGPT2(p, hp, device="cpu", dtype=torch.float32)


def test_torch_fp64_matches_numpy_fp64():
    hp, p64 = load_gpt2("gpt2", dtype=np.float64)
    m = TorchGPT2(p64, hp, device="cpu", dtype=torch.float64)
    for prompt in PROMPTS:
        ids = encode(prompt)
        diff = np.abs(m.forward_dense(ids).numpy() - np_gpt2(ids, p64, hp)).max()
        assert diff < 1e-4, (prompt, diff)


def test_torch_fp32_matches_numpy_fp32(gpt2_small, tm):
    # same tolerance policy as the HF test: relative to logit scale + identical argmax
    hp, p = gpt2_small
    for prompt in PROMPTS:
        ids = encode(prompt)
        a = tm.forward_dense(ids).numpy()
        b = np_gpt2(ids, p, hp)
        assert a.dtype == np.float32
        assert np.abs(a - b).max() / np.abs(b).max() < 1e-5
        np.testing.assert_array_equal(a.argmax(-1), b.argmax(-1))


def _close(a, b):
    return torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-3)


def test_paged_prefill_equals_dense_mixed_batch(tm):
    """Several prompts of different lengths in ONE forward_batch == each run dense."""
    cache = tm.new_cache(num_blocks=64)
    seqs = []
    for prompt in PROMPTS:
        ids = encode(prompt)
        tab = BlockTable()
        cache.reserve(tab, len(ids))
        seqs.append((ids, tab))
    out = tm.forward_batch(seqs, cache, all_logits=True)
    for (ids, tab), got in zip(seqs, out):
        assert tab.length == len(ids)
        _close(got, tm.forward_dense(ids))


def test_paged_decode_equals_dense_across_block_boundaries(tm):
    """Prefill, then decode token by token past several 16-token block boundaries,
    in a batch with sequences at different lengths. Every step's logits == dense."""
    cache = tm.new_cache(num_blocks=64, block_size=16)
    rng = np.random.default_rng(0)
    full = [rng.integers(0, 50257, size=n).tolist() for n in (40, 33, 52)]
    pre = [15, 16, 17]                       # prompt lengths straddling a block boundary
    tabs = [BlockTable() for _ in full]
    for f, n, t in zip(full, pre, tabs):
        cache.reserve(t, n)
    tm.forward_batch([(f[:n], t) for f, n, t in zip(full, pre, tabs)], cache)
    dense = [tm.forward_dense(f) for f in full]
    for step in range(20):
        batch, which = [], []
        for i, (f, t) in enumerate(zip(full, tabs)):
            if t.length < len(f):
                cache.reserve(t, t.length + 1)
                batch.append(([f[t.length]], t))
                which.append((i, t.length))
        logits = tm.forward_batch(batch, cache)
        for row, (i, pos) in zip(logits, which):
            _close(row, dense[i][pos])


def test_paged_uses_noncontiguous_blocks(tm):
    """A sequence whose blocks are scattered (and out of order) in the pool still matches."""
    cache = tm.new_cache(num_blocks=32, block_size=16)
    junk = [BlockTable() for _ in range(5)]
    for j in junk:
        cache.reserve(j, 16)                 # take blocks 0..4
    cache.free(junk[3]); cache.free(junk[1])  # holes at 3 and 1 -> next allocs are scattered
    ids = encode(PROMPTS[2])                  # 21 tokens -> 2 blocks
    tab = BlockTable()
    cache.reserve(tab, len(ids))
    assert tab.blocks == [1, 3]
    _close(tm.forward_batch([(ids, tab)], cache, all_logits=True)[0], tm.forward_dense(ids))


def test_forward_batch_requires_reserved_blocks(tm):
    cache = tm.new_cache(num_blocks=4)
    with pytest.raises(ValueError):
        tm.forward_batch([(encode(PROMPTS[0]), BlockTable())], cache)


def test_torch_greedy_equals_numpy_greedy(gpt2_small, tm):
    hp, p = gpt2_small
    ids = encode(PROMPTS[0])
    cache = tm.new_cache(num_blocks=8)
    tab = BlockTable()
    cache.reserve(tab, len(ids))
    logits = tm.forward_batch([(ids, tab)], cache)
    out = []
    for _ in range(20):
        out.append(int(logits[0].argmax()))
        cache.reserve(tab, tab.length + 1)
        logits = tm.forward_batch([([out[-1]], tab)], cache)
    assert out == np_generate(ids, p, hp, 20, use_cache=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU (run on Colab)")
def test_cuda_fp16_matches_fp32(gpt2_small, tm):
    hp, p = gpt2_small
    g = TorchGPT2(p, hp, device="cuda", dtype=torch.float16)
    for prompt in PROMPTS:
        ids = encode(prompt)
        a = g.forward_dense(ids).float().cpu()
        b = tm.forward_dense(ids)
        assert (a - b).abs().max() / b.abs().max() < 1e-2
        assert (a.argmax(-1) == b.argmax(-1)).float().mean() >= 0.9
