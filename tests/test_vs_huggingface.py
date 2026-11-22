"""Correctness oracle: NumPy GPT-2 logits vs HuggingFace GPT2LMHeadModel.

`transformers` is used ONLY here (tests), never inside engine/.

Why two tests:
- fp64 (the 1e-4 spec): with both models in float64 the only differences left are
  in the math, so this proves the implementation is the same function as HF.
  Measured max abs diff is ~1e-12.
- fp32: GPT-2 logits reach |300|, where one fp32 ulp is ~3e-5, and HF's own fp32
  logits differ from its fp64 logits by up to ~1e-3. A raw 1e-4 bound in fp32
  would be testing rounding noise, so fp32 is checked relative to the logit scale
  and on argmax agreement.
"""
import numpy as np
import pytest

from engine.numpy_model import gpt2
from engine.tokenizer import encode
from engine.weights import WEIGHTS_DIR, load_gpt2

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

TOL_FP64 = 1e-4          # CLAUDE.md spec: max abs diff at every position
TOL_FP32_REL = 1e-5      # max abs diff / max |logit|  (measured: <= 3.4e-6)

PROMPTS = [
    "Alan Turing theorized that computers would one day become",
    "Hello",
    "The quick brown fox jumps over the lazy dog. " * 6,
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "In 1905, Albert Einstein published four papers that changed physics: the photoelectric "
    "effect, Brownian motion, special relativity, and mass-energy equivalence. Today, more than "
    "a century later, physicists still argue about what quantum mechanics really means, and",
]
IDS = [f"p{i}" for i in range(len(PROMPTS))]


def _hf(dtype):
    m = transformers.GPT2LMHeadModel.from_pretrained(
        str(WEIGHTS_DIR / "gpt2"), use_safetensors=True, attn_implementation="eager",
    )
    return m.to(dtype).eval()


@pytest.fixture(scope="module")
def hf32(gpt2_small):  # gpt2_small guarantees weights are downloaded
    return _hf(torch.float32)


@pytest.fixture(scope="module")
def hf64(gpt2_small):
    return _hf(torch.float64)


@pytest.fixture(scope="module")
def ours64():
    return load_gpt2("gpt2", dtype=np.float64)


def _hf_logits(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].numpy()


@pytest.mark.parametrize("prompt", PROMPTS, ids=IDS)
def test_logits_match_hf_fp64_every_position(prompt, ours64, hf64):
    hp, p = ours64
    ids = encode(prompt)
    ours = gpt2(ids, p, hp)
    theirs = _hf_logits(hf64, ids)
    assert ours.dtype == np.float64 and ours.shape == theirs.shape
    diff = np.abs(ours - theirs).max()
    assert diff < TOL_FP64, f"max abs diff {diff:.2e} (T={len(ids)})"


@pytest.mark.parametrize("prompt", PROMPTS, ids=IDS)
def test_logits_match_hf_fp32_every_position(prompt, gpt2_small, hf32):
    hp, p = gpt2_small
    ids = encode(prompt)
    ours = gpt2(ids, p, hp)
    theirs = _hf_logits(hf32, ids)
    assert ours.shape == theirs.shape
    rel = np.abs(ours - theirs).max() / np.abs(theirs).max()
    assert rel < TOL_FP32_REL, f"relative max diff {rel:.2e} (T={len(ids)})"
    np.testing.assert_array_equal(ours.argmax(-1), theirs.argmax(-1))
