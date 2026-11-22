# LLM Inference Engine from Scratch

GPT-2 inference written from scratch (NumPy reference, then PyTorch + a Triton kernel), verified against HuggingFace and sped up with a KV cache, int8 quantization, continuous batching, a block-based KV cache and speculative decoding, all served through an OpenAI-compatible streaming API.

**Status:** Phase 3 (int8 weight-only quantization) done. See `CLAUDE.md` for the plan.

## Setup (Windows)

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q tests
```

(Plain `py -3.12 -m venv .venv` + `pip install` works too.) torch is the CPU wheel. On Windows, torch ≥ 2.12 needs a recent Visual C++ Redistributable (`winget install --id Microsoft.VCRedist.2015+.x64 -e`), otherwise `import torch` fails looking for `VCRUNTIME140_THREADS.dll`. `transformers` (in `requirements-dev.txt`) is the test oracle only and is never imported by `engine/`. The first run downloads `gpt2` `config.json` + `model.safetensors` (~550 MB) into `weights/` (gitignored). Pickled `.bin` files are never loaded.

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

### Phase 2: match HuggingFace + KV cache (`tests/test_vs_huggingface.py`, `KVCache` in `engine/numpy_model.py`)

**Matching HuggingFace.** HF's `GPT2LMHeadModel` (test-only oracle) is loaded from the same safetensors file, and logits are compared at **every position** for 5 prompts (1 to 61 tokens: prose, code, repeated text).
- **fp64, both sides: max abs diff < 1e-4** (the spec). Measured ~1e-13 to 2e-12, which shows the two implementations compute the same function.
- **fp32:** a raw 1e-4 bound would be testing rounding noise. GPT-2 logits reach |330|, where one fp32 ulp is ~3e-5, and HF's own fp32 logits differ from its fp64 logits by up to 9.6e-4. So fp32 is checked as max diff / max |logit| < 1e-5 (measured ≤ 3.4e-6), plus an identical argmax at every position.

**KV cache.** In attention, the keys and values of past tokens never change once computed, so recomputing them every step is wasted work.
- `KVCache` preallocates `k, v` of shape [n_layer, n_head, n_ctx, head_dim]. That is 75.5 MB fp32 for a full 1,024-token context (2 × 12 × 12 × 1024 × 64 × 4 bytes).
- **Prefill:** run the whole prompt once, writing each layer's K/V into slots `0..T)`.
- **Decode:** feed only the newest token. Its position is `cache.length`. Each layer writes its single K/V row at that slot and attends over slots `0..length]`. Cost per token goes from O(T) full-sequence work to one token's matmuls plus attention over T cached keys.
- The causal mask generalises to `causal_mask(tq, tk)`: tq new queries at the end of a tk-long sequence, where query *i* sees keys ≤ tk − tq + i. Decode is just tq = 1. Chunked prefill works too.
- `last_only=True` runs the 50257 × 768 output head on the final position only, which is all generation needs.

**Tests** (`tests/test_kv_cache.py`):
- Prefill logits equal uncached logits.
- Decode one token at a time: each step's logits equal row *t* of a full uncached forward.
- Chunked prefill equals single prefill.
- Greedy tokens with the cache are identical to greedy tokens without it, for 3 prompts.
- Mask offsets are correct, and overflowing n_ctx raises an error.

**Results** (`bench/kv_cache_bench.py` → `bench/results/kv_cache.md`, i5-1135G7 laptop on AC power, fp32 NumPy; medians of 20 cached / 5 uncached samples for A, 3 runs each for B):

| | L / N = 32 | 256 | 1024 |
|---|---|---|---|
| A. one decode step at context L: cached vs uncached tok/s | 30.3 vs 7.2 (**4.2×**) | 29.1 vs 1.31 (**22.2×**) | 25.4 vs 0.26 (**96.3×**) |
| B. generate N tokens end to end: cached vs uncached tok/s | 31.7 vs 8.4 (**3.8×**) | 27.8 vs 2.1 (**13.2×**) | 23.3 vs not run |

The speedup grows with length because uncached work per token grows linearly with context, while cached work stays nearly flat (the weights dominate; attention over T cached keys is small next to them at these lengths). End-to-end speedup is smaller than per-step speedup because it averages over all the shorter contexts along the way. Uncached N = 1014 takes tens of minutes per run, so it only runs with `--full`.

**Benchmarking on a laptop** (lessons from getting these numbers):
- Configs are timed **round-robin**, one sample of each per round, so CPU clock drift (turbo, thermals) is shared evenly instead of landing on whichever config ran last.
- Cached and uncached decode steps are timed as **separate groups**. A 35 ms cached step timed right after a 4 s uncached forward runs throttled; mixed together, that made cached L = 32 look slower than L = 256.
- **Plug the laptop in.** On battery the same benchmark ran ~40% slower and much noisier. The results file records the power source.

### Phase 3: int8 weight-only quantization (`QInt8` / `quantize_int8` in `engine/weights.py`, `linear` / `lm_head` in `engine/numpy_model.py`, `engine/eval.py`)

**Quantizing.** Every big matrix (the 48 block projections and `wte`) is stored as int8 plus one fp32 scale per **output channel**: `scale = max|w_channel| / 127`, `q = round(w / scale)` clipped to ±127 (symmetric, no zero point). The largest weight in each channel maps to exactly ±127, and the rounding error per weight is at most scale / 2. Per-channel scales matter because channels' ranges differ a lot: one scale for the whole matrix would let a single outlier column crush the precision of all the others. For the Conv1D weights [in, out] the channel axis is 1. For `wte` [vocab, C] it is 0 (one scale per vocab row), because as the tied output head each row produces one logit. `wpe`, biases and layer norms (~0.6% of parameters) stay fp32. A zero channel gets scale 1 so nothing divides by zero.

**Running it ("weight-only").** Activations stay fp32. Before each matmul the int8 weight is converted to fp32. Two tricks keep this from being painfully slow:
- **Scale after the matmul.** `x @ (q · s) = (x @ q) · s`, because s is constant down each output column. So the weight is only *cast*, and the scale multiplies the small [T, out] result instead of the [in, out] weight.
- **One reusable scratch buffer.** The cast writes into a single persistent fp32 buffer (9 MB, sized for the largest 768 × 3072 matrix) instead of allocating a new temporary per matmul. Fresh multi-MB allocations were page-faulted in on every call by Windows, which made the cast ~2× slower.
- **Chunked output head.** Dequantizing all of `wte` at once would create a 154 MB fp32 temporary and wipe out the memory win, so the head runs in chunks of 2048 vocab rows.

**Measuring perplexity** (`engine/eval.py`). Perplexity = exp(mean negative log-likelihood of each next token). The 287,644-token WikiText-2 raw test set is longer than GPT-2's 1024-token context, so it is scored in windows. Each window predicts its chunk of targets, and every token after the first is scored **exactly once**. (The common HuggingFace recipe silently skips the first token of every window after the first; a test catches that.) With stride < 1024, each window keeps up to 1024 tokens of earlier text as context: more context gives lower perplexity, at more compute.

**Tests** (`tests/test_int8.py`, `tests/test_perplexity.py`):
- Quantizer: error ≤ scale / 2 on both axes; each channel's max maps to ±127; zero channels are safe.
- `linear` with int8 equals a matmul with the dequantized weight; the scratch buffer can be reused across matrix shapes; the chunked head equals the full dequantized head for a vocab size that isn't a multiple of the chunk.
- Real model: int8 weights are 0.25–0.27× the fp32 bytes; per-position logit cosine similarity > 0.999 and top-1 agreement ≥ 90% for 3 prompts; int8 cached == uncached tokens; coherent generation.
- Correctness-ladder rung: int8 perplexity within 3% of fp32 on 2 windows of WikiText-2.
- Perplexity: stable log-softmax; strided windows score every token once; single-window perplexity matches HF's loss.

**Results** (`bench/int8_bench.py` → `bench/results/int8.md`, i5-1135G7 laptop on AC power; decode medians of 20 samples, end to end 3 runs):

| | fp32 | int8 | int8 / fp32 |
|---|---|---|---|
| stored weights | 474.7 MB | 121.8 MB | **0.26× (3.9× smaller)** |
| WikiText-2 test perplexity (287,643 tokens, ctx 1024, stride 1024) | 29.94 | 30.43 | **+1.64%** |
| decode step @ L = 32 / 256 / 1024 (tok/s) | 23.3 / 22.5 / 19.0 | 5.2 / 5.2 / 5.1 | 0.22–0.27× |
| generate 256 tokens end to end (tok/s) | 21.0 | 6.7 | 0.32× |
| extra memory during a 1024-token forward | 163 MB | 163 MB (+9 MB scratch) | 1.0× |

**Reading the results honestly.**
- **Memory is the real win:** weights shrink 3.9× for a 1.6% perplexity cost. On a GPU, weight memory is what limits model size and batch size.
- **Speed is worse here, and that is expected:** NumPy has no int8 × fp32 matmul, so every step re-casts ~124M int8 weights to fp32 and *then* does the same fp32 matmul as before. That is extra work and extra memory traffic, so decoding is ~4× slower. Real int8 inference wins speed because a fused kernel reads int8 from memory and dequantizes in registers; decode is memory-bandwidth bound, so reading 4× fewer bytes is the speedup. NumPy can't express that kernel; the GPU phases are where speed comes from.
- **Activation memory doesn't change:** activations and attention stay fp32, so peak extra memory during a forward pass is the same.
- The fp32 decode numbers here (~23 tok/s) are lower than Phase 2's (~30) on the same code path: different run and thermal state. Compare within a table, not across tables.

## Honest notes so far
- Tokenizer is tiktoken's `gpt2` encoding, not my own BPE.
- fp32 equality with HF is relative to logit scale (see Phase 2); the strict 1e-4 bound is enforced in fp64.
- CPU only on this machine (no CUDA GPU). All numbers so far are from an i5-1135G7 laptop CPU; GPU numbers come from the Colab notebook later.
- int8 is weight-only and slower than fp32 in NumPy (no fused int8 kernel); its win in this repo is memory, not speed (see Phase 3).
