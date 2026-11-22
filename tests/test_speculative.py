"""Correctness ladder rung: speculative greedy output == target greedy output."""
import pytest
import torch

from engine.speculative import SpecStats, greedy_generate, speculative_generate
from engine.tokenizer import decode, encode
from engine.torch_model import TorchGPT2
from engine.weights import load_gpt2

PROMPTS = [
    "Alan Turing theorized that computers would one day become",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "The history of the Roman Empire",
]


@pytest.fixture(scope="module")
def draft(gpt2_small):
    hp, p = gpt2_small
    return TorchGPT2(p, hp, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def target():
    hp, p = load_gpt2("gpt2-medium")
    assert hp["n_layer"] == 24 and hp["n_embd"] == 1024
    return TorchGPT2(p, hp, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def reference(target):
    return {pr: greedy_generate(target, encode(pr), 40) for pr in PROMPTS}


def test_target_greedy_is_coherent(reference):
    text = decode(reference[PROMPTS[0]])
    assert len(text.split()) > 10, text


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_speculative_equals_target_greedy(target, draft, reference, k):
    for pr in PROMPTS:
        st = SpecStats()
        out = speculative_generate(target, draft, encode(pr), 40, k, st)
        assert out == reference[pr], (pr, k)
        assert st.generated == 40 and 0 <= st.accepted <= st.drafted
        assert st.rounds == st.target_forwards - 1       # one verify per round (+ prefill)
        assert st.rounds <= 40


def test_speculation_saves_target_forwards(target, draft):
    st = SpecStats()
    speculative_generate(target, draft, encode(PROMPTS[1]), 40, 4, st)
    assert st.acceptance_rate > 0.3, st                  # small and medium agree often on code
    assert st.rounds < 40


def test_same_model_as_draft_accepts_everything(draft):
    st = SpecStats()
    ids = encode(PROMPTS[0])
    out = speculative_generate(draft, draft, ids, 30, 5, st)
    assert out == greedy_generate(draft, ids, 30)
    assert st.accepted == st.drafted
    assert st.rounds == 5                                # 6 tokens per round: 5 drafted + 1 bonus


@pytest.mark.parametrize("n_new, k", [(1, 4), (4, 4), (5, 4), (7, 3)])
def test_edge_lengths(target, draft, reference, n_new, k):
    ids = encode(PROMPTS[0])
    assert speculative_generate(target, draft, ids, n_new, k) == reference[PROMPTS[0]][:n_new]


def test_one_token_prompt(target, draft):
    ids = encode("The")
    assert len(ids) == 1
    assert speculative_generate(target, draft, ids, 12, 4) == greedy_generate(target, ids, 12)


def test_near_context_limit(target, draft):
    ids = (encode(PROMPTS[2]) * 200)[:1010]
    out = speculative_generate(target, draft, ids, 14, 8)
    assert out == greedy_generate(target, ids, 14)
    with pytest.raises(ValueError):
        speculative_generate(target, draft, ids, 15, 8)


def test_bad_args(target, draft):
    with pytest.raises(ValueError):
        speculative_generate(target, draft, encode("hi"), 5, 0)
    with pytest.raises(ValueError):
        speculative_generate(target, draft, [], 5, 2)
