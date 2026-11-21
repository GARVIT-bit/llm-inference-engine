"""Unit tests for the NumPy building blocks (no weights needed)."""
import math

import numpy as np

from engine.numpy_model import attention, causal_mask, gelu, layer_norm, mha, softmax


def test_gelu_known_values():
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    # exact erf-GELU for reference; tanh approximation is within ~1e-3
    exact = np.array([0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in x])
    np.testing.assert_allclose(gelu(x), exact, atol=1e-3)
    assert gelu(np.array(0.0)) == 0.0


def test_softmax_sums_to_one_and_is_stable():
    x = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 1000.0]])
    s = softmax(x)
    assert np.all(np.isfinite(s))
    np.testing.assert_allclose(s.sum(-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(s[0], softmax(np.array([0.0, 1.0, 2.0])), rtol=1e-6)


def test_layer_norm_normalizes():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 5.0, size=(4, 64))
    y = layer_norm(x, np.ones(64), np.zeros(64))
    np.testing.assert_allclose(y.mean(-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(y.var(-1), 1.0, atol=1e-3)
    # gain/bias applied
    y2 = layer_norm(x, 2 * np.ones(64), np.ones(64))
    np.testing.assert_allclose(y2, 2 * y + 1, atol=1e-6)


def test_causal_mask_shape_and_values():
    m = causal_mask(4)
    assert m.shape == (4, 4)
    assert np.all(m[np.tril_indices(4)] == 0)
    assert np.all(m[np.triu_indices(4, k=1)] < -1e30)


def test_attention_first_token_attends_only_to_itself():
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(2, 5, 8)).astype(np.float32) for _ in range(3))
    out = attention(q, k, v, causal_mask(5))
    np.testing.assert_allclose(out[:, 0], v[:, 0], rtol=1e-5, atol=1e-6)


def test_mha_is_causal():
    """Changing a later token must not change earlier outputs."""
    rng = np.random.default_rng(0)
    C, H = 16, 4
    c_attn = {"w": rng.normal(size=(C, 3 * C)).astype(np.float32), "b": np.zeros(3 * C, np.float32)}
    c_proj = {"w": rng.normal(size=(C, C)).astype(np.float32), "b": np.zeros(C, np.float32)}
    x = rng.normal(size=(6, C)).astype(np.float32)
    x2 = x.copy()
    x2[-1] += 10.0
    a = mha(x, c_attn, c_proj, H)
    b = mha(x2, c_attn, c_proj, H)
    np.testing.assert_allclose(a[:-1], b[:-1], rtol=1e-5, atol=1e-5)
    assert not np.allclose(a[-1], b[-1])
