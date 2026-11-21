# LLM Inference Engine from Scratch: context for Claude Code

## What this project is
Project 4 of my resume set (I'm Garvit, CSE student; other projects: ../mini-perplexity, ../crdt-editor, ../mini-docker-k8s, which another Claude Code window is building right now; don't touch those folders).
GPT-2 inference written from scratch: load OpenAI's released GPT-2 weights into my own transformer code, prove it matches HuggingFace's logits, then make it fast one bottleneck at a time, measuring every step against my own previous version.
Features: NumPy reference forward pass, KV cache, int8 weight-only quantization, PyTorch fp16 backend, block-based (PagedAttention-style) KV cache, continuous batching scheduler, speculative decoding (GPT-2 small drafts for GPT-2 medium), a Triton fused-attention kernel, OpenAI-compatible streaming API.

Rules:
- My code does the model math. Don't use `transformers` model classes, `model.generate()`, vLLM, or `torch.nn.functional.scaled_dot_product_attention` inside the engine. `transformers` is allowed ONLY in tests, as the correctness oracle.
- Tokenizer: tiktoken `gpt2` encoding (stated honestly in the README; my own BPE is a stretch goal).
- Weights: safetensors from the HuggingFace Hub (`gpt2`, `gpt2-medium`). Never load pickled .bin files.

## Machine and environment
- Windows laptop, Python 3.12. This project runs natively on Windows (no WSL needed). Use a venv at `.venv` (`.venv\Scripts\python.exe`), same as mini-perplexity.
- Disk is LOW. Install the CPU-only PyTorch wheel (`--index-url https://download.pytorch.org/whl/cpu`) unless `nvidia-smi` shows a CUDA GPU. Put weights in one cache folder (`weights/`, gitignored); gpt2 + gpt2-medium safetensors is ~2 GB, so download gpt2-medium only when Phase 5 needs it. Check free space before big downloads and tell me if it's under 5 GB.
- GPU work: check `nvidia-smi` first. If there's no CUDA GPU (likely), the PyTorch backend must also run on CPU (fp32) so every test passes locally, and GPU benchmarks + the Triton kernel run on a free Colab/Kaggle T4: write `bench/colab_gpu.ipynb` that clones the repo from GitHub (GARVIT-bit), installs deps, runs the GPU tests and benchmarks, and writes CSVs to bench/results/. I'll run it and paste/commit the results. Triton doesn't run on Windows, so kernel tests skip when unavailable.
- Ports already in use on this machine: 8080, 8090, 7070, 27017, 27018, 6379, 6380. Use 8001 for this API server.
- No API keys or paid services needed. Never commit secrets.

## Layout (rename if something better emerges)
engine/weights.py (load_gpt2, quantize_int8), engine/numpy_model.py, engine/torch_model.py, engine/kv_cache.py (block allocator, block tables), engine/sampler.py (greedy/temperature/top-p), engine/scheduler.py (continuous batching), engine/speculative.py, kernels/attention_triton.py, server/app.py (FastAPI), bench/*.py + bench/results/*.csv|md, tests/.

## Build phases
1. NumPy forward pass: load gpt2 (124M) safetensors into NumPy dicts (note GPT-2's Conv1D weight layout), gelu, stable softmax, layer_norm, causal attention, mha, ffn, block, gpt2; greedy loop. Done when "Alan Turing theorized that computers would one day become" continues coherently.
2. Match HuggingFace + KV cache: logits within 1e-4 max abs diff for 5 prompts at every position; KV cache (prefill stores K/V, decode feeds only the newest token); cached == uncached tokens; bench tokens/sec at 32/256/1,024 tokens with vs without cache.
3. int8 weight-only quantization: per-output-channel scale = max|row|/127, dequantize before matmul; measure memory, tokens/sec, perplexity on WikiText-2 test (fp32 vs int8).
4. PyTorch backend + block KV cache + continuous batching + API: torch model matches NumPy; blocks of 16 tokens with per-sequence block tables, freed on finish; scheduler admits while blocks are free, one batched decode step for all running sequences; FastAPI `/v1/completions` (stream via SSE, max_tokens, temperature, top_p), `/v1/models`, `/metrics`, `/health`; load test 64 concurrent requests with random output lengths, static vs continuous batching (throughput, p50 TTFT). The OpenAI Python client pointed at base_url should work.
5. Speculative decoding + Triton: gpt2 drafts k tokens, gpt2-medium verifies in one pass; greedy speculative output identical to target greedy; acceptance rate + speedup for k = 2, 4, 6, 8. Triton fused attention (tiled, online softmax, causal) vs PyTorch attention, within fp16 tolerance, benchmarked at seq 256 to 4,096 (Colab).
6. Ship: README with architecture diagram, results table for every optimisation (each row from a bench script), plots, demo steps, honest limits (old GPT-2 architecture, no RoPE/GQA, tiktoken tokenizer, single GPU, where numbers came from: CPU model / T4).

Correctness ladder (keep all of these as tests): NumPy vs HF; torch vs NumPy; cached vs uncached; int8 vs fp32 perplexity; speculative vs target greedy; Triton vs PyTorch attention. Tolerances: fp32 ~1e-4, fp16 ~1e-2. Benchmarks: warm-up, GPU sync before timing, several runs with median and spread, fixed seed, machine details in the results file.

## How I want to work (same as my other projects)
- Build without long explanations; tests after every module, run them before moving on; short summary at the end of each phase.
- Never drop or weaken tests to go faster.
- I still need to understand what I ship: add a short "how it works" README section per phase so I can catch up before interviews.
- Every resume number comes from a script in bench/ that I can rerun. Never invent numbers; leave placeholders for GPU numbers until I run the Colab notebook.
- Ask before downloads over 1 GB.
