"""Perplexity code checked against HF's built-in LM loss (test-only oracle)."""
import math

import numpy as np
import pytest

from engine.eval import log_softmax, perplexity
from engine.tokenizer import encode
from engine.weights import WEIGHTS_DIR

TEXT = ("Robert Boulter is an English film, television and theatre actor. He had a guest "
        "starring role on the television series The Bill in 2000. This was followed by a "
        "starring role in the play Herons written by Simon Stephens, which was performed in "
        "2001 at the Royal Court Theatre.")


def test_log_softmax_stable():
    x = np.array([[1000.0, 1000.0], [-1000.0, 0.0]])
    out = log_softmax(x)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(np.exp(out).sum(-1), 1.0)


def test_strided_windows_score_every_token_once(gpt2_small):
    hp, p = gpt2_small
    ids = encode(TEXT * 3)
    a = perplexity(ids, p, hp, ctx=32, stride=32)
    b = perplexity(ids, p, hp, ctx=32, stride=16)
    assert a["tokens"] == b["tokens"] == len(ids) - 1
    assert b["ppl"] < a["ppl"]   # overlapping windows give each token more context


def test_single_window_ppl_matches_hf_loss(gpt2_small):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    hp, p = gpt2_small
    ids = encode(TEXT)
    ours = perplexity(ids, p, hp)
    m = transformers.GPT2LMHeadModel.from_pretrained(
        str(WEIGHTS_DIR / "gpt2"), use_safetensors=True, attn_implementation="eager").eval()
    with torch.no_grad():
        t = torch.tensor([ids])
        loss = m(t, labels=t).loss.item()     # mean NLL over the len-1 predicted tokens
    assert ours["tokens"] == len(ids) - 1
    np.testing.assert_allclose(ours["ppl"], math.exp(loss), rtol=1e-4)
