"""End-to-end tests of the NumPy GPT-2 with the real 124M weights."""
import numpy as np
import pytest

from engine.numpy_model import generate, gpt2
from engine.tokenizer import decode, encode


def test_weight_shapes(gpt2_small):
    hp, p = gpt2_small
    assert hp == {"n_vocab": 50257, "n_ctx": 1024, "n_embd": 768, "n_head": 12, "n_layer": 12, "eps": 1e-5}
    assert p["wte"].shape == (50257, 768)
    assert p["wpe"].shape == (1024, 768)
    assert len(p["blocks"]) == 12
    b = p["blocks"][0]
    # Conv1D layout: [in, out]
    assert b["attn"]["c_attn"]["w"].shape == (768, 3 * 768)
    assert b["attn"]["c_proj"]["w"].shape == (768, 768)
    assert b["mlp"]["c_fc"]["w"].shape == (768, 4 * 768)
    assert b["mlp"]["c_proj"]["w"].shape == (4 * 768, 768)
    assert p["wte"].dtype == np.float32


def test_parameter_count(gpt2_small):
    _, p = gpt2_small
    n = p["wte"].size + p["wpe"].size + sum(v.size for v in (p["ln_f"]["g"], p["ln_f"]["b"]))
    for blk in p["blocks"]:
        for sub in blk.values():
            for leaf in (sub.values() if "g" in sub else (x for d in sub.values() for x in d.values())):
                n += leaf.size
    assert n == 124_439_808  # GPT-2 small with tied output head


def test_tokenizer_roundtrip():
    text = "Alan Turing theorized that computers would one day become"
    ids = encode(text)
    assert decode(ids) == text
    assert encode("Hello world") == [15496, 995]


def test_logits_shape_and_finite(gpt2_small):
    hp, p = gpt2_small
    logits = gpt2(encode("Hello, my name is"), p, hp)
    assert logits.shape == (5, 50257)
    assert np.all(np.isfinite(logits))
    # guard against silent float64 promotion (NumPy 2 / NEP 50)
    assert logits.dtype == np.float32


def test_model_is_causal(gpt2_small):
    hp, p = gpt2_small
    a = gpt2(encode("The cat sat on the mat"), p, hp)
    b = gpt2(encode("The cat sat on the dog"), p, hp)
    np.testing.assert_allclose(a[:-1], b[:-1], atol=1e-4)


def test_rejects_too_long(gpt2_small):
    hp, p = gpt2_small
    with pytest.raises(ValueError):
        gpt2([0] * 1025, p, hp)


def test_turing_greedy_continuation(gpt2_small):
    hp, p = gpt2_small
    prompt = "Alan Turing theorized that computers would one day become"
    out = decode(generate(encode(prompt), p, hp, n_new=8))
    assert out == " the most powerful machines on the planet."
