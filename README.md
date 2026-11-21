# LLM Inference Engine from Scratch

GPT-2 inference written from scratch (NumPy reference, then PyTorch + a Triton kernel), verified against HuggingFace and sped up with a KV cache, int8 quantization, continuous batching, a block-based KV cache and speculative decoding, all served through an OpenAI-compatible streaming API.

**Status:** Phase 1 (NumPy forward pass) done. See `CLAUDE.md` for the plan.

## Setup (Windows)

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m pytest -q tests
```

(Plain `py -3.12 -m venv .venv` + `pip install -r requirements.txt` works too.) The first run downloads `gpt2` `config.json` + `model.safetensors` (~550 MB) into `weights/` (gitignored). Pickled `.bin` files are never loaded.

```powershell
.venv\Scripts\python.exe generate.py "Alan Turing theorized that computers would one day become" --n 40
```

GPU benchmarks and the Triton kernel run on a free Colab/Kaggle T4 via `bench/colab_gpu.ipynb`.

## How it works

### Phase 1: NumPy forward pass (`engine/weights.py`, `engine/numpy_model.py`)

**Weights.** `load_gpt2` reads the safetensors file into a nested dict of float32 arrays: token embeddings `wte` [50257, 768], position embeddings `wpe` [1024, 768], 12 blocks, and a final layer norm `ln_f`. GPT-2 was written with a `Conv1D` layer that stores weights as **[in, out]**, so every projection is `x @ W + b` with no transpose (PyTorch's `nn.Linear` is [out, in]). 124,439,808 parameters, checked by a test.

**Forward pass** for T tokens (C = 768, H = 12 heads, D = 64):
1. `x = wte[ids] + wpe[0..T)` → [T, C].
2. 12 pre-norm blocks: `x = x + attn(ln_1(x))`, then `x = x + ffn(ln_2(x))`.
   - **Attention:** one matmul `c_attn` makes [T, 3C], split into q, k, v, reshaped to [H, T, D]. Scores `q kᵀ / √D` plus a **causal mask** (large negative above the diagonal, so token *t* only sees tokens ≤ *t*), softmax, `@ v`, merge heads, `c_proj`.
   - **FFN:** C → 4C, GELU, 4C → C.
3. Final `ln_f`, then logits = `x @ wteᵀ`: the output head reuses (is *tied to*) the input embedding matrix.

**Numerics.** Softmax subtracts the row max before `exp` (so no overflow). The mask uses `finfo.min`, not `-inf`, so a fully masked row can't produce `nan`. GELU is the tanh approximation GPT-2 was trained with. Layer norm uses ε = 1e-5.

**Gotcha found while building it:** under NumPy 2 (NEP 50), multiplying a float32 array by a `np.float64` scalar like `np.sqrt(2/np.pi)` silently upcasts the whole activation to float64. Using plain Python floats for constants took a 50-token forward pass from ~2.5 s to ~0.25 s. A test now asserts the logits stay float32.

**Generation.** `generate` is greedy (argmax) and has **no KV cache**: every new token re-runs the whole sequence, so the cost of step *t* grows with *t*. That's what Phase 2 fixes.

**Tests** (`tests/test_numpy_ops.py`, `tests/test_numpy_gpt2.py`): GELU vs exact erf-GELU, softmax stability at ±1000, layer norm stats, causal mask, "changing a later token doesn't change earlier outputs" (for both `mha` and the full model), weight shapes + parameter count, tokenizer round-trip, float32 logits, and the exact greedy continuation of the Turing prompt: *" the most powerful machines on the planet."*

## Honest notes so far
- Tokenizer is tiktoken's `gpt2` encoding, not my own BPE.
- Phase 1 has not been compared to HuggingFace logits yet; that's the first thing in Phase 2 (1e-4 max abs diff target).
- CPU only on this machine (no CUDA GPU); throughput numbers come in Phase 2 from `bench/` scripts.
