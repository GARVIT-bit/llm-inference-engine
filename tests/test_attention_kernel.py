"""Correctness ladder rung: fused (tiled, online-softmax) attention vs plain attention.

The NumPy mirror of the kernel's loop runs everywhere; the Triton kernel itself runs
only with Triton + CUDA (Colab), and those tests skip otherwise."""
import numpy as np
import pytest

from kernels.attention_triton import HAS_TRITON, naive_attention, tiled_attention_reference

needs_triton = pytest.mark.skipif(not HAS_TRITON, reason="needs Triton + CUDA (run bench/colab_gpu.ipynb)")


def _qkv(shape, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=shape) * scale for _ in range(3)]


def test_naive_attention_is_softmax_definition():
    q, k, v = _qkv((5, 8))
    out = naive_attention(q, k, v, causal=True)
    np.testing.assert_allclose(out[0], v[0])            # first query sees only itself
    s = q[3] @ k[:4].T / np.sqrt(8)
    w = np.exp(s - s.max()); w /= w.sum()
    np.testing.assert_allclose(out[3], w @ v[:4])


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("T, bm, bn", [(1, 64, 64), (17, 16, 16), (100, 32, 16), (130, 64, 32), (64, 64, 64)])
def test_tiled_online_softmax_equals_naive(causal, T, bm, bn):
    q, k, v = _qkv((2, 3, T, 16), seed=T)
    np.testing.assert_allclose(tiled_attention_reference(q, k, v, causal, bm, bn),
                               naive_attention(q, k, v, causal), rtol=1e-10, atol=1e-12)


def test_tiled_is_stable_with_large_scores():
    # scores ~1e3: a softmax without the running max would overflow exp
    q, k, v = _qkv((1, 1, 70, 16), scale=30.0)
    out = tiled_attention_reference(q, k, v, True, 32, 16)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, naive_attention(q, k, v, True), rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------- GPU only

@needs_triton
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("T", [1, 17, 64, 200, 1000])
@pytest.mark.parametrize("D", [64, 128])
def test_triton_matches_fp32_reference(causal, T, D):
    import torch
    from kernels.attention_triton import triton_attention
    g = torch.Generator(device="cuda").manual_seed(T)
    q, k, v = (torch.randn(2, 4, T, D, device="cuda", generator=g, dtype=torch.float16) for _ in range(3))
    out = triton_attention(q, k, v, causal=causal)
    ref = naive_attention(*(t.double().cpu().numpy() for t in (q, k, v)), causal=causal)
    err = (out.double().cpu().numpy() - ref)
    assert np.abs(err).max() < 1e-2, np.abs(err).max()


@needs_triton
def test_triton_handles_noncontiguous_inputs():
    import torch
    from kernels.attention_triton import triton_attention
    x = torch.randn(1, 50, 3, 4, 64, device="cuda", dtype=torch.float16)   # [B, T, 3, H, D]
    q, k, v = (x[:, :, i].transpose(1, 2) for i in range(3))              # strided [B, H, T, D]
    ref = triton_attention(q.contiguous(), k.contiguous(), v.contiguous())
    torch.testing.assert_close(triton_attention(q, k, v), ref)


@needs_triton
def test_model_with_triton_attention_matches_torch_attention(gpt2_small):
    import torch
    from engine.tokenizer import encode
    from engine.torch_model import TorchGPT2
    hp, p = gpt2_small
    ids = encode("Alan Turing theorized that computers would one day become the most powerful machines") * 8
    a = TorchGPT2(p, hp, device="cuda", dtype=torch.float16, attention="triton").forward_dense(ids).float()
    b = TorchGPT2(p, hp, device="cuda", dtype=torch.float16, attention="torch").forward_dense(ids).float()
    assert (a - b).abs().max() / b.abs().max() < 1e-2
    assert (a.argmax(-1) == b.argmax(-1)).float().mean() > 0.95


def test_triton_option_refuses_without_gpu(gpt2_small):
    if HAS_TRITON:
        pytest.skip("GPU present")
    import torch
    from engine.torch_model import TorchGPT2
    hp, p = gpt2_small
    with pytest.raises(RuntimeError):
        TorchGPT2(p, hp, device="cpu", dtype=torch.float32, attention="triton")
