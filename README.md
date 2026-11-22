# LLM Inference Engine from Scratch

GPT-2 inference written from scratch (NumPy reference, then PyTorch + a Triton kernel), verified against HuggingFace and sped up with a KV cache, int8 quantization, continuous batching, a block-based KV cache and speculative decoding, all served through an OpenAI-compatible streaming API.

**Status:** Phase 5 (speculative decoding + Triton attention) done, with CPU and T4 numbers. Phase 6 (ship: diagram, plots, final README) is next. See `CLAUDE.md` for the plan.

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

### Phase 4: PyTorch backend, paged KV cache, continuous batching, API (`engine/torch_model.py`, `engine/kv_cache.py`, `engine/sampler.py`, `engine/scheduler.py`, `server/app.py`)

**PyTorch backend.** The same math as the NumPy model, written with plain tensor ops (matmuls, a hand-written softmax and layer norm; no `nn.Module` zoo, no SDPA). It runs fp32 on CPU and fp16 on CUDA. Layer-norm statistics and softmax are computed in fp32 even in fp16 mode, because fp16 variance underflows. Two entry points:
- `forward_dense(ids)`: one sequence, no cache. This is the reference.
- `forward_batch(seqs, cache)`: many sequences over the paged cache. Each brings its *new* tokens (a whole prompt on prefill, one token on decode) and its block table.

**Paged KV cache** (`engine/kv_cache.py`). Phase 2's cache preallocated n_ctx = 1024 slots per sequence: 75.5 MB fp32, mostly empty for short requests, so 64 concurrent sequences would reserve 4.8 GB. Instead, all K/V live in one shared pool `[n_layer, num_blocks, 16, n_head, head_dim]`:
- Each sequence owns a **block table**, the list of block ids holding its tokens in order. Position p is stored at `blocks[p // 16] * 16 + p % 16`.
- Blocks come from a free list as a sequence grows (1.18 MB per 16 tokens in fp32) and go back when it finishes. Waste is at most 15 token slots per sequence.
- Any free block can serve any sequence, so blocks don't need to be contiguous. A test checks scattered, out-of-order blocks.

**Batched forward.** All new tokens of all sequences are flattened into one [N, 768] matrix, so every linear layer is one matmul for the whole batch. For attention:
1. Each layer writes the new K/V into the pool at their slots (`index_copy_`).
2. Each sequence's history is gathered through its block table into a padded [B, H, Lmax, 64] tensor.
3. Queries are padded to [B, H, Tmax, 64].
4. One mask removes both the future (key position > query position) and padding.

That gather is a copy of every cached K/V per layer per step. A paged-attention kernel would read the blocks in place; that is what the kernel phase is for.

**Sampler** (`engine/sampler.py`). Batched, with per-row settings:
- temperature 0 → argmax;
- otherwise sample from softmax(logits / T);
- top-p keeps the smallest set of most-likely tokens whose mass reaches p (the top token is always kept), then `torch.multinomial` renormalises;
- seeded with a `torch.Generator`.

**Scheduler** (`engine/scheduler.py`). Each `step()`:
1. **Admit** waiting requests while the batch has room (max_batch) and the pool has blocks for their prompts. A small headroom of one free block per running sequence is kept so the next decode step doesn't immediately run out. All newly admitted prompts are prefilled in one forward, and each gets its first token.
2. **Decode:** one batched forward for every request that was already running. Each feeds its newest token, then a token is sampled.
3. **Retire** finished requests (max_tokens, `<|endoftext|>`, or context full) and free their blocks.
- **`continuous`** admits every step, so a finished request's slot is refilled at once. **`static`** admits a new batch only when the whole previous batch is done. Finished requests stop computing, but their slots stay empty, so the baseline is a little generous to static.
- **Preemption:** if a decode step needs a block and none is free, the youngest running request is preempted. Its blocks are freed and it goes to the front of the queue. Later it is re-prefilled from prompt + tokens so far ("recompute" preemption). A test runs 5 requests in a pool that holds about half of them and checks the outputs are identical to unconstrained runs.
- `LLMEngine` runs the scheduler in one background thread. `submit()` and `abort()` are thread-safe, and per-request callbacks deliver each token as it is sampled.

**API server** (`server/app.py`, port 8001). Endpoints:
- `POST /v1/completions` takes `prompt` (string or token ids), `max_tokens`, `temperature`, `top_p` and `stream`. Non-streaming returns OpenAI's `text_completion` JSON with `usage`. `stream: true` sends Server-Sent Events, `data: {chunk}` per token and then `data: [DONE]`.
- `GET /v1/models`, `GET /health`, and `GET /metrics` (Prometheus text: running/waiting sequences, free blocks, tokens, preemptions, mean decode batch, median TTFT).

Implementation details:
- Each HTTP request registers callbacks that push tokens from the engine thread into an `asyncio.Queue` on the server loop (`call_soon_threadsafe`).
- A GPT-2 token can be half of a multi-byte UTF-8 character, so the streamer holds back text that ends in U+FFFD until the rest arrives.
- If the client disconnects mid-stream, the request is aborted and its blocks freed.
- `ignore_eos` (a vLLM-style extension) lets the load test fix output lengths.
- The official OpenAI Python client works against it, and a test proves it:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="unused")
for ch in c.completions.create(model="gpt2", prompt="Alan Turing theorized", max_tokens=30, stream=True):
    print(ch.choices[0].text, end="", flush=True)
```

```powershell
.venv\Scripts\python.exe -m server.app
```

(Flags: `--policy static`, `--max-batch`, `--num-blocks`, `--device`.)

**Performance gotcha found while building it.** On CPU (MKL), `x @ wte.T` for a few rows of x against the 50257 × 768 embedding took 58 ms. Computing `(wte @ x.T).T` gives the same numbers in 21 ms. Together with `index_select` for the KV gather, this made a batch-16 decode step 1.4× faster.

**Tests:**
- `test_torch_model.py`: torch fp64 vs NumPy fp64 max diff < 1e-4; fp32 relative + argmax (same policy as the HF test). The paged mixed-length batch equals dense; decoding across 16-token block boundaries equals dense at every step; non-contiguous blocks work; torch greedy equals NumPy greedy; CUDA fp16 vs fp32 within 1e-2 (Colab only).
- `test_paged_cache.py`: allocator, double free, out of blocks, reserve rounding, slot mapping.
- `test_sampler.py`: greedy, nucleus never leaves the top-p set and renormalises correctly, temperature-1 frequencies match softmax, per-row settings, seeding.
- `test_scheduler.py`: batched greedy equals single-sequence greedy (both policies); continuous refills slots while static waits; preemption keeps outputs identical; EOS and context-limit stops; abort; every block is always returned; the engine thread streams tokens.
- `test_server.py`: JSON shape and usage, SSE chunks join to the non-streaming text, concurrent requests share decode steps (mean batch > 1.5), 4xx validation, Prometheus format, the UTF-8 hold-back, the **official OpenAI client** (plain + streaming) against a live uvicorn server, and a client disconnect aborting the request and freeing its blocks.

**Results** (`bench/batching_bench.py` → `bench/results/batching.md`). 64 concurrent streaming requests over HTTP, prompts 16–128 tokens, outputs 16–256 tokens (seeded), max batch 16, 3 runs per policy:

| i5-1135G7 laptop CPU, fp32, AC power | static | continuous | continuous / static |
|---|---|---|---|
| throughput (generated tok/s) | 65.3 (64.6 to 68.5) | 63.7 (63.4 to 68.0) | 0.98× |
| TTFT p50 (s) | 39.5 (39.4 to 39.8) | 41.9 (33.0 to 43.6) | 1.06× |
| TTFT p90 (s) | 108.2 (104.6 to 109.1) | 102.5 (93.0 to 102.6) | 0.95× |
| time per output token p50 (ms) | 133 | 236 | |
| batched decode steps | 1150 | 719 | 0.63× |
| mean sequences per decode step | 7.8 | 12.5 | |

**On this CPU, continuous batching does not win, and the numbers show why.**
- Continuous batching keeps the batch full: 12.5 sequences per decode step instead of 7.8, and 37% fewer steps for the same 9,004 tokens. That only pays off if a step with 16 sequences costs about the same as a step with 1.
- On a GPU it does, because decode is bound by reading the weights from memory, and the extra rows are nearly free.
- On this 4-core laptop CPU, step cost grows almost in proportion to batch size. From the run above:
  - static: 138 s / 1,150 steps = 120 ms per step at 7.8 sequences;
  - continuous: 141 s / 719 steps = 197 ms per step at 12.5 sequences (including its prefills).
- So the work per token is nearly the same however the requests are packed: 15.3 vs 15.7 ms per generated token.
- So throughput and median TTFT come out even. The tail TTFT is ~5% better with continuous batching. Per-token latency is worse with it, because its steps are always full, while static batches drain down to a few sequences.
- The scheduler is doing its job: fewer steps and fuller batches. The hardware just doesn't reward that.
- The GPU run in `bench/colab_gpu.ipynb` measures the case continuous batching is built for. Until it has been run, there is no claim here that continuous batching is faster.

**GPU** (Colab Tesla T4, fp16, same workload and script; `bench/colab_gpu.ipynb` → `bench/results/batching_t4*.md`; median of 3 runs, range in the files):

| T4, fp16 | static | continuous | continuous / static |
|---|---|---|---|
| max batch 16: throughput (tok/s) | 380.2 | 554.4 | **1.46×** |
| max batch 16: TTFT p50 / p90 (s) | 7.63 / 18.16 | 4.45 / 10.44 | **0.58×** p50 |
| max batch 16: time per output token p50 (ms) | 21 | 23 | |
| max batch 64: throughput (tok/s) | 630.9 | 1150.9 | 1.82×* |
| max batch 64: TTFT p50 / p90 (s) | 2.69 / 9.89 | 0.29 / 0.30 | 0.11×* p50 |

**The GPU result shows what the CPU run couldn't.**
- On the T4 a decode step costs about the same with 16 sequences as with 7, because reading the weights dominates.
- So the 37% fewer, fuller steps of continuous batching turn straight into 1.46× throughput.
- Time to first token falls to 0.58×, because a freed slot is refilled on the next step instead of waiting for the slowest request in the batch.
- Per-token latency rises slightly (21 → 23 ms), because every step carries a full batch plus new prefills.

\*The max-batch-64 static row is not a clean baseline. The scheduler prefills at most 4,096 prompt tokens per step, and these 64 prompts total 4,851. So static admitted about 54 requests, and the remaining ~10 waited for a second full wave: 638 decode steps instead of about 255. Treat 1.82× as "continuous vs static under the same prefill budget", not as pure batching. **The max-batch-16 row is the headline number.**

A second Colab session (`bench/results/batching_t4_run2.md`, `batching_t4_b64_run2.md`) reproduced these numbers: 1.43× throughput and 0.59× TTFT p50 at max batch 16, and 1.86× at max batch 64.

### Phase 5: speculative decoding + Triton fused attention (`engine/speculative.py`, `kernels/attention_triton.py`)

**Speculative decoding.** At batch size 1, a decode step is limited by reading the weights, not by arithmetic. So a big model checking 5 tokens in one forward costs about the same as generating 1. A small **draft** model (gpt2, 124M) guesses k tokens greedily, and the **target** (gpt2-medium, 355M, same tokenizer) runs one forward over `[last, d1 … dk]`. That gives the target's own greedy choice g0 … gk at every position. Then:
- Accept the longest prefix where the draft agreed (d(i+1) == g(i)), say n tokens, and append g(n): the target's correction, or a free bonus token if all k matched. Each round emits n + 1 tokens for **one** target forward.
- Every emitted token is a target greedy token, so the **output is identical to the target's greedy output**. The draft only changes how many tokens come out per target forward.
- **KV rollback** is cheap with the paged cache: rejected tokens are dropped by lowering `table.length`, and their slots get overwritten next round. The draft cache also tracks which accepted tokens it has already seen (when all k match, the draft never fed the last one, so it is fed at the start of the next round).
- Only greedy verification is implemented, not the rejection-sampling variant for temperature > 0.

**Tests** (`tests/test_speculative.py`):
- Speculative output == gpt2-medium greedy output, token for token, for k = 1, 2, 4, 8 on 3 prompts (fp32).
- Using the same model as draft and target accepts every draft (k + 1 tokens per round).
- Edge cases: 1–7 output tokens, a 1-token prompt, and generating up to the 1024-token context limit (the draft length shrinks to fit).

**Results** (`bench/speculative_bench.py` → `bench/results/speculative.md`, laptop CPU fp32 on AC, 4 prompts × 128 tokens, 3 runs round-robin, median):

| | tok/s | vs target greedy | acceptance | tokens / target forward | output == target greedy |
|---|---|---|---|---|---|
| target greedy (gpt2-medium) | 7.52 | 1.00× | | 1 | |
| draft alone (gpt2) | 18.13 | 2.41× | | | |
| speculative k = 2 | 6.59 | 0.88× | 0.81 | 2.61 | yes |
| speculative k = 4 | 7.40 | 0.98× | 0.69 | 3.71 | yes |
| speculative k = 6 | 7.06 | 0.94× | 0.59 | 4.49 | yes |
| speculative k = 8 | 5.82 | 0.77× | 0.55 | 5.28 | yes |

**On this CPU, speculation saves target forwards (3.7× fewer at k = 4) but not time.** Speculative decoding needs two things, and the laptop provides neither:
1. **A cheap draft.** Here gpt2 costs 55 ms per token against gpt2-medium's 133 ms, a ratio of 0.41. Both share the same 50257 × 768+ output head, and per-forward overhead doesn't shrink with model size. Speculative decoding pays off when the draft is ~10× cheaper.
2. **Near-free verification.** At k = 4, a round takes 3.71 / 7.40 = 501 ms. The 4 draft steps take about 220 ms, which leaves ~280 ms for verifying 5 tokens: about 2.1× a single-token step. A CPU is compute-bound, so 5 tokens cost more than 1.
- The acceptance rates (0.55–0.81) are healthy. The math only works where verification is close to free, which is the GPU row below.
- The timing spread is wide (k = 4 ranged from 6.2 to 8.4 tok/s), because the laptop had ~1 GB of RAM free while holding both models.

**GPU** (Colab Tesla T4, fp16, same script and prompts → `bench/results/speculative_t4.md`):

| T4, fp16 | tok/s | vs target greedy | acceptance | tokens / target forward | output == target greedy |
|---|---|---|---|---|---|
| target greedy (gpt2-medium) | 31.98 | 1.00× | | 1 | |
| draft alone (gpt2) | 60.59 | 1.90× | | | |
| speculative k = 2 | 39.63 | **1.24×** | 0.81 | 2.61 | yes |
| speculative k = 4 | 37.99 | 1.19× | 0.69 | 3.71 | yes |
| speculative k = 6 | 33.88 | 1.06× | 0.59 | 4.49 | yes |
| speculative k = 8 | 32.48 | 1.02× | 0.55 | 5.22 | yes |

**On the GPU it pays off, and a two-number model predicts by how much.**
- On the T4, verifying k + 1 tokens costs about one target step, so the second CPU problem is gone.
- The first one remains. At batch size 1 a GPU forward is dominated by launching hundreds of small kernels per layer, so gpt2 (12 layers) costs about half of gpt2-medium (24 layers): c = 31.98 / 60.59 = 0.53.
- Expected speedup ≈ (tokens per round) / (1 + k·c):
  - k = 2: 2.61 / 2.06 = 1.27× (measured 1.24×);
  - k = 4: 3.71 / 3.11 = 1.19× (measured 1.19×);
  - k = 8: 5.22 / 5.24 = 1.0× (measured 1.02×).
- So k = 2 is best here. A bigger gap between draft and target (or CUDA graphs to cut launch overhead) would move the best k up.
- Output stayed identical to target greedy at every k, even in fp16.

**Triton fused attention** (`kernels/attention_triton.py`). Naive attention writes a [T, T] score matrix per head to GPU memory, reads it back for the softmax, writes the probabilities, then reads them again for `@ v`. At T = 4096 that is 32 MB per head in fp16. The kernel (FlashAttention-style forward) never stores it:
- One GPU program handles a 64-row tile of queries for one (batch, head) and walks over 64-column tiles of keys/values, stopping at the diagonal because the attention is causal.
- For each tile it computes the scores in fast on-chip memory and keeps a **running row max m** and **running denominator l**. When a new tile raises the max, it rescales the old partial output by exp(m_old − m_new). This "online softmax" gives exactly softmax over the whole row, without ever holding the row.
- It uses exp2 with log2(e) folded into the scale (the GPU's native exponent).
- Memory goes from O(T²) to O(T) per head, and the kernel reads Q, K, V once per tile pass instead of shuttling the scores through GPU memory.
- `kernels/attention_triton.py` also has `tiled_attention_reference`, the same loop in NumPy. It matches plain attention to 1e-10 for odd lengths and tile sizes, and stays finite with scores around 1e3. That way the algorithm is tested on the laptop; the Triton code itself is tested in Colab.
- `TorchGPT2(..., attention="triton")` uses the kernel for prefill on CUDA (the decode path still uses the paged gather).

**Tests** (`tests/test_attention_kernel.py`):
- NumPy mirror vs the definition (causal and non-causal, T from 1 to 130, uneven tiles, large scores).
- On CUDA: Triton fp16 vs an fp64 reference within 1e-2 for T in {1, 17, 64, 200, 1000} and D in {64, 128}; strided (non-contiguous) inputs; the full GPT-2 forward with Triton attention vs torch attention.

**Results** (`bench/attention_bench.py` → `bench/results/attention_t4.md`, Colab Tesla T4, fp16, batch 2 × 12 heads × head dim 64, causal, median of 20 runs). The Triton tests passed on the T4.

| seq len | PyTorch naive (ms) | Triton fused (ms) | torch SDPA, reference only (ms) | naive peak memory | Triton peak memory | max abs error naive / Triton |
|---|---|---|---|---|---|---|
| 256 | 0.44 | 3.16 | 0.15 | 15 MB | 0.8 MB | 1.4e-3 / 1.0e-3 |
| 1024 | 4.31 | 37.5 | 0.38 | 240 MB | 3 MB | 1.6e-3 / 1.0e-3 |
| 2048 | 17.1 | 145 | 1.16 | 960 MB | 6 MB | 1.8e-3 / 1.2e-3 |
| 4096 | 68.4 | 570 | 3.95 | 3,840 MB | 12 MB | 2.0e-3 / 1.0e-3 |

**First version: correct, memory-light, and slow.**
- **Correct:** error against an fp32 reference is 1.0–1.2e-3, the same as PyTorch's production fused kernel (SDPA), and lower than naive fp16 attention.
- **Memory:** extra memory grows linearly instead of quadratically. At 4096 tokens it needs 12 MB instead of 3,840 MB, **320× less**. That is the point of the online softmax.
- **Speed:** with its first settings (64 × 64 tiles, 4 warps, 2 pipeline stages) it was ~8× *slower* than naive attention: 570 vs 68 ms at 4096, only 0.09 TFLOP/s. The table above is that version.

**Diagnosis and tuning** (`bench/attention_tune.py` → `bench/results/attention_tune_t4.md`, Colab T4). The script times a plain Triton matmul against cuBLAS, inspects what the compiler emitted (registers, spills, and whether the PTX contains tensor-core `mma.sync` instructions), and sweeps 162 kernel configs, each checked for correctness. It found two separate problems:
1. **Register spills: the 12× problem.** The first config compiled to 64 registers per thread with 1,448 spills to local memory, so every tile iteration went back and forth to slow memory. The best config (64-row query tiles, **32-column** key tiles, **8 warps**, 1 stage) uses 221 registers and **0 spills**. Most of the fast configs have no spills; the slow ones all spill.
2. **No tensor cores on this T4 with this Triton: the remaining gap.** `mma.sync` appears **0 times** in the PTX of every config, including a plain Triton matmul. The `tl.dot` calls compile to scalar fp32 FMAs. Even Triton's plain 4096³ matmul reached only 1.1 TFLOP/s here, against 22.2 TFLOP/s for cuBLAS on the same GPU. So the kernel runs on the T4's CUDA cores, not its tensor cores. That limit comes from the T4 (Turing, compute capability 7.5) with this Triton build, not from the attention algorithm.

**After tuning** (now the kernel's default settings; interleaved timing, median of 10 runs):

| seq len | naive PyTorch (ms) | Triton, first config (ms) | Triton, tuned (ms) | tuned vs naive | max abs error |
|---|---|---|---|---|---|
| 256 | 0.436 | 3.451 | 0.377 | 1.16× | 9.9e-4 |
| 512 | 1.179 | 10.024 | 1.085 | 1.09× | 9.6e-4 |
| 1024 | 4.391 | 37.570 | 3.610 | 1.22× | 9.8e-4 |
| 2048 | 17.250 | 145.637 | 13.452 | 1.28× | 9.9e-4 |
| 4096 | 66.292 | 573.390 | 48.588 | **1.36×** | 9.5e-4 |

- Tuning made the kernel **11.8× faster** than its first version, and it now **beats naive PyTorch attention by 1.16–1.36×** while using 320× less memory at 4096 tokens.
- The gap to naive grows with length, because naive attention's O(T²) memory traffic grows faster.
- It is still ~12× slower than PyTorch's SDPA (3.95 ms at 4096), which reaches tensor cores through hand-written CUDA kernels.
- Closing that gap would need either a GPU where Triton emits MMA (Ampere or newer: Colab L4/A100, paid) or hand-written CUDA with `mma.sync`. Both are out of scope here.
- Honest resume line: *"wrote a fused causal attention kernel in Triton (online softmax, O(T) memory); diagnosed register spills from compiled PTX and tuned it from 8× slower to 1.36× faster than PyTorch attention on a T4; 320× less memory at 4K tokens."*

## Honest notes so far
- Tokenizer is tiktoken's `gpt2` encoding, not my own BPE.
- fp32 equality with HF is relative to logit scale (see Phase 2); the strict 1e-4 bound is enforced in fp64.
- CPU only on this machine (no CUDA GPU). All numbers so far are from an i5-1135G7 laptop CPU; GPU numbers come from the Colab notebook later.
- int8 is weight-only and slower than fp32 in NumPy (no fused int8 kernel); its win in this repo is memory, not speed (see Phase 3).
- The static-batching baseline drops finished requests from the compute (it only wastes their slots), which is kinder than padded static batching.
- The API covers what the benchmarks and the OpenAI client need: one prompt per request (n = 1), no `stop` sequences, no logprobs. One engine thread, one model, one device.
- Speculative decoding is greedy-only (no rejection sampling for temperature > 0) and batch size 1; it is not wired into the server.
- The Triton attention kernel beats naive PyTorch attention (1.16–1.36× on a T4) but is ~12× slower than PyTorch's SDPA: on the T4 with this Triton build, `tl.dot` compiles to scalar FMAs, not tensor-core MMA (see Phase 5). It is used only for prefill via `attention="triton"`, not by default.
