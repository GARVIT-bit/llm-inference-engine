import pytest
import torch

from engine.sampler import sample


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_temperature_zero_is_argmax():
    logits = torch.randn(5, 100, generator=_gen())
    assert torch.equal(sample(logits, [0.0] * 5, [1.0] * 5), logits.argmax(-1))


def test_tiny_top_p_is_argmax():
    logits = torch.randn(5, 100, generator=_gen())
    out = sample(logits, [1.0] * 5, [1e-6] * 5, generator=_gen(1))
    assert torch.equal(out, logits.argmax(-1))


def test_top_p_never_samples_outside_nucleus():
    # probs 0.5, 0.3, 0.15, 0.05: top_p=0.7 keeps {0, 1}; top_p=0.8 keeps {0, 1} (0.5+0.3 reaches 0.8)
    logits = torch.tensor([0.5, 0.3, 0.15, 0.05]).log().repeat(4000, 1)
    out = sample(logits, [1.0] * 4000, [0.7] * 4000, generator=_gen())
    assert set(out.tolist()) == {0, 1}
    out = sample(logits, [1.0] * 4000, [0.81] * 4000, generator=_gen())
    assert set(out.tolist()) == {0, 1, 2}
    # renormalised inside the nucleus {0, 1}: P(0) = 0.5 / 0.8
    out = sample(logits, [1.0] * 4000, [0.7] * 4000, generator=_gen(3))
    assert abs((out == 0).float().mean().item() - 0.625) < 0.03


def test_temperature_one_matches_softmax():
    probs = torch.tensor([0.6, 0.25, 0.1, 0.05])
    n = 20000
    out = sample(probs.log().repeat(n, 1), [1.0] * n, [1.0] * n, generator=_gen())
    freq = torch.bincount(out, minlength=4).float() / n
    assert (freq - probs).abs().max() < 0.015, freq


def test_high_temperature_flattens():
    logits = torch.tensor([4.0, 0.0, 0.0, 0.0]).repeat(5000, 1)
    cold = sample(logits, [0.5] * 5000, [1.0] * 5000, generator=_gen())
    hot = sample(logits, [5.0] * 5000, [1.0] * 5000, generator=_gen())
    assert (hot != 0).float().mean() > (cold != 0).float().mean() + 0.3


def test_mixed_rows_and_seed_reproducible():
    logits = torch.randn(6, 50, generator=_gen())
    temps, tps = [0.0, 1.0, 0.0, 0.8, 1.2, 0.0], [1.0, 0.9, 0.5, 1.0, 0.95, 1.0]
    a = sample(logits, temps, tps, generator=_gen(7))
    b = sample(logits, temps, tps, generator=_gen(7))
    assert torch.equal(a, b)
    greedy = logits.argmax(-1)
    for i in (0, 2, 5):
        assert a[i] == greedy[i]


def test_bad_params_raise():
    logits = torch.zeros(2, 4)
    with pytest.raises(ValueError):
        sample(logits, [1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        sample(logits, [-1.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        sample(logits, [1.0, 1.0], [0.0, 1.0])
