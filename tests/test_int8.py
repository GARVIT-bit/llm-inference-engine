"""int8 weight-only quantization: quantizer math, memory, and the int8-vs-fp32 ladder rung."""
import numpy as np
import pytest

from engine.data import wikitext2_test_text
from engine.eval import perplexity
from engine.numpy_model import HEAD_CHUNK, generate, gpt2, lm_head
from engine.tokenizer import encode
from engine.weights import QInt8, params_nbytes, quantize_int8

PROMPTS = [
    "Alan Turing theorized that computers would one day become",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
]


@pytest.fixture(scope="module")
def gpt2_int8(gpt2_small):
    hp, p = gpt2_small
    return hp, quantize_int8(p)


# ------------------------------------------------------------------ quantizer math

@pytest.mark.parametrize("out_axis", [0, 1])
def test_quantize_error_bounded_by_half_step(out_axis):
    rng = np.random.default_rng(0)
    w = rng.normal(size=(64, 48)).astype(np.float32) * rng.uniform(0.01, 3, size=(1, 48) if out_axis == 1 else (64, 1)).astype(np.float32)
    qt = QInt8.quantize(w, out_axis=out_axis)
    assert qt.q.dtype == np.int8 and qt.scale.dtype == np.float32
    assert qt.scale.shape == ((1, 48) if out_axis == 1 else (64, 1))
    assert qt.q.min() >= -127 and qt.q.max() <= 127
    # each channel's largest |w| maps to exactly +-127
    assert np.all(np.abs(qt.q).max(axis=1 - out_axis) == 127)
    err = np.abs(qt.dequantize() - w)
    assert np.all(err <= qt.scale / 2 + 1e-6)


def test_quantize_zero_channel():
    w = np.zeros((4, 3), np.float32)
    w[:, 1] = [1.0, -2.0, 0.5, 0.0]
    qt = QInt8.quantize(w, out_axis=1)
    assert np.all(np.isfinite(qt.dequantize()))
    np.testing.assert_array_equal(qt.dequantize()[:, 0], 0)
    np.testing.assert_allclose(qt.dequantize()[:, 1], w[:, 1], atol=2.0 / 127 / 2 + 1e-7)


def test_int8_linear_equals_dequantized_matmul():
    from engine.numpy_model import linear
    rng = np.random.default_rng(0)
    w = rng.normal(size=(32, 96)).astype(np.float32)
    b = rng.normal(size=96).astype(np.float32)
    qt = QInt8.quantize(w, out_axis=1)
    x = rng.normal(size=(5, 32)).astype(np.float32)
    # scale-after-matmul == matmul with the dequantized weight
    np.testing.assert_allclose(linear(x, qt, b), x @ qt.dequantize() + b, rtol=1e-5, atol=1e-5)
    # scratch reuse across differently shaped weights doesn't corrupt results
    qt2 = QInt8.quantize(rng.normal(size=(96, 32)).astype(np.float32), out_axis=1)
    y = linear(x, qt, b)
    linear(y, qt2, np.zeros(32, np.float32))
    np.testing.assert_allclose(linear(x, qt, b), y)


def test_chunked_int8_head_equals_full_dequant():
    rng = np.random.default_rng(0)
    V = 2 * HEAD_CHUNK + 123                         # not a multiple of the chunk size
    wte = QInt8.quantize(rng.normal(size=(V, 16)).astype(np.float32), out_axis=0)
    x = rng.normal(size=(3, 16)).astype(np.float32)
    np.testing.assert_allclose(lm_head(x, wte), x @ wte.dequantize().T, rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------------ real model

def test_int8_weight_memory(gpt2_small, gpt2_int8):
    _, p32 = gpt2_small
    _, p8 = gpt2_int8
    b32, b8 = params_nbytes(p32), params_nbytes(p8)
    assert b32 == 124_439_808 * 4
    # ~0.6% of params stay fp32 (wpe, biases, LN) + one fp32 scale per output channel
    assert 0.25 < b8 / b32 < 0.27, b8 / b32


def test_int8_logits_close_to_fp32(gpt2_small, gpt2_int8):
    hp, p32 = gpt2_small
    _, p8 = gpt2_int8
    agree, total = 0, 0
    for prompt in PROMPTS:
        ids = encode(prompt)
        a, b = gpt2(ids, p32, hp), gpt2(ids, p8, hp)
        assert b.dtype == np.float32 and np.all(np.isfinite(b))
        # per-position cosine similarity of logit vectors
        cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        assert cos.min() > 0.999, cos.min()
        agree += int((a.argmax(-1) == b.argmax(-1)).sum())
        total += len(ids)
    assert agree / total >= 0.9, f"top-1 agreement {agree}/{total}"


def test_int8_cached_equals_uncached(gpt2_int8):
    hp, p8 = gpt2_int8
    ids = encode(PROMPTS[0])
    assert generate(ids, p8, hp, 20, use_cache=True) == generate(ids, p8, hp, 20, use_cache=False)


def test_int8_generation_coherent(gpt2_int8):
    hp, p8 = gpt2_int8
    from engine.tokenizer import decode
    out = decode(generate(encode(PROMPTS[0]), p8, hp, 8, use_cache=True))
    assert "machines" in out or "computers" in out, out


def test_int8_perplexity_close_to_fp32_on_wikitext_slice(gpt2_small, gpt2_int8):
    """Correctness ladder: int8 vs fp32 perplexity (2 x 1024-token windows of WikiText-2 test).
    The full-test-set numbers come from bench/int8_bench.py."""
    hp, p32 = gpt2_small
    _, p8 = gpt2_int8
    ids = encode(wikitext2_test_text())[:2048]
    r32 = perplexity(ids, p32, hp)
    r8 = perplexity(ids, p8, hp)
    assert r32["tokens"] == r8["tokens"] == 2047
    assert 5 < r32["ppl"] < 100                      # sane range for GPT-2 small on WikiText
    assert abs(r8["ppl"] - r32["ppl"]) / r32["ppl"] < 0.03, (r32["ppl"], r8["ppl"])
