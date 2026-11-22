"""Benchmark: fp32 vs int8 weight-only quantization (NumPy GPT-2 124M, CPU).

  1. memory: bytes of stored weights, and peak extra memory allocated during a
     forward pass (tracemalloc; NumPy reports its buffers to it)
  2. speed: cached decode step at context L, and end-to-end cached generation;
     fp32 and int8 timed round-robin
  3. quality: perplexity on the full WikiText-2 (raw) test set, sliding window
     ctx=1024 (see engine/eval.py)

    .venv\\Scripts\\python.exe bench\\int8_bench.py                  # full run (~1 h on a laptop CPU)
    .venv\\Scripts\\python.exe bench\\int8_bench.py --ppl-tokens 4096  # quick check
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.common import RESULTS, machine_info, machine_md, time_interleaved  # noqa: E402
from engine.data import wikitext2_test_text  # noqa: E402
from engine.eval import perplexity  # noqa: E402
from engine.numpy_model import KVCache, generate, gpt2, scratch_nbytes  # noqa: E402
from engine.tokenizer import encode  # noqa: E402
from engine.weights import load_gpt2, params_nbytes, quantize_int8  # noqa: E402

LENGTHS = [32, 256, 1024]
PROMPT = "Alan Turing theorized that computers would one day become"
SEED = 0
MB = 1024 * 1024


def peak_alloc(fn) -> int:
    """Peak bytes allocated above the starting point while fn() runs."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak - base


def decode_fn(hp, p, L: int):
    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, hp["n_vocab"], size=L).tolist()
    cache = KVCache(hp)
    gpt2(ids[:-1], p, hp, cache=cache, last_only=True)

    def step():
        cache.length = L - 1
        gpt2(ids[-1:], p, hp, cache=cache, last_only=True)
    return step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20, help="samples per decode-step config")
    ap.add_argument("--e2e-runs", type=int, default=3)
    ap.add_argument("--e2e-tokens", type=int, default=256)
    ap.add_argument("--ppl-tokens", type=int, default=0, help="limit WikiText tokens (0 = full test set)")
    ap.add_argument("--stride", type=int, default=1024)
    args = ap.parse_args()

    if machine_info()["power"] == "battery":
        print("WARNING: on battery power; laptop throttles, numbers will be low and noisy", flush=True)

    hp, p32 = load_gpt2("gpt2")
    t0 = time.perf_counter()
    p8 = quantize_int8(p32)
    quant_s = time.perf_counter() - t0
    models = {"fp32": p32, "int8": p8}
    rows = []

    # 1. memory
    mem = {}
    for name, p in models.items():
        ids256 = list(range(256))
        mem[name] = {
            "weights_mb": params_nbytes(p) / MB,
            "peak_decode_mb": peak_alloc(decode_fn(hp, p, 256)) / MB,   # one step, L=256
            "peak_prefill1024_mb": peak_alloc(lambda: gpt2(list(range(1024)), p, hp, last_only=True)) / MB,
        }
        del ids256
        # int8 keeps one reusable fp32 dequant buffer alive (engine.numpy_model._scratch)
        mem[name]["dequant_scratch_mb"] = (scratch_nbytes() / MB) if name == "int8" else 0.0
        for k, v in mem[name].items():
            rows.append({"section": "memory", "model": name, "metric": k, "value": round(v, 2), "extra": ""})
        print(f"memory {name}: {mem[name]}", flush=True)

    # 2. speed
    fns = {(L, name): decode_fn(hp, p, L) for L in LENGTHS for name, p in models.items()}
    stats = time_interleaved(fns, runs=args.runs)
    for (L, name), s in stats.items():
        rows.append({"section": "decode_step", "model": name, "metric": f"tok_per_s@L={L}",
                     "value": round(1 / s["median"], 3),
                     "extra": f"min {1 / s['max']:.3f} max {1 / s['min']:.3f} runs {s['runs']}"})
    for L in LENGTHS:
        f, q = stats[(L, "fp32")], stats[(L, "int8")]
        print(f"decode step L={L}: fp32 {1/f['median']:.2f} tok/s | int8 {1/q['median']:.2f} tok/s", flush=True)

    ids = encode(PROMPT)
    N = args.e2e_tokens
    e2e = time_interleaved({name: (lambda p=p: generate(ids, p, hp, N, use_cache=True))
                            for name, p in models.items()}, runs=args.e2e_runs)
    for name, s in e2e.items():
        rows.append({"section": "e2e_generate", "model": name, "metric": f"tok_per_s@N={N}",
                     "value": round(N / s["median"], 3),
                     "extra": f"min {N / s['max']:.3f} max {N / s['min']:.3f} runs {s['runs']}"})
        print(f"e2e N={N}: {name} {N / s['median']:.2f} tok/s", flush=True)

    # 3. perplexity
    wt = encode(wikitext2_test_text())
    if args.ppl_tokens:
        wt = wt[:args.ppl_tokens]
    ppl = {}
    for name, p in models.items():
        t0 = time.perf_counter()
        last = [0.0]

        def prog(i, n, cur, name=name):
            if time.perf_counter() - last[0] > 60 or i == n:
                last[0] = time.perf_counter()
                print(f"  ppl {name}: window {i}/{n}, running ppl {cur:.3f}", flush=True)
        r = perplexity(wt, p, hp, ctx=1024, stride=args.stride, progress=prog)
        r["seconds"] = time.perf_counter() - t0
        ppl[name] = r
        rows.append({"section": "perplexity", "model": name, "metric": "wikitext2_test_ppl",
                     "value": round(r["ppl"], 4),
                     "extra": f"tokens {r['tokens']} ctx 1024 stride {args.stride} time {r['seconds']:.0f}s"})
        print(f"ppl {name}: {r['ppl']:.4f} over {r['tokens']} tokens ({r['seconds']:.0f}s)", flush=True)

    # write
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "int8.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def ds(L, name):
        s = stats[(L, name)]
        return f"{1 / s['median']:.2f} ({1 / s['max']:.2f} to {1 / s['min']:.2f})"

    full = "full test set" if not args.ppl_tokens else f"first {args.ppl_tokens} tokens only"
    lines = [
        "# int8 weight-only quantization benchmark (NumPy GPT-2 124M, CPU)", "",
        "Generated by `bench/int8_bench.py`. Per-output-channel symmetric int8 "
        "(scale = max|w| / 127), dequantized to fp32 before each matmul. "
        f"Quantizing took {quant_s:.2f}s. Speed: configs timed round-robin after 1 warm-up, "
        "median (min to max).", "",
        "## Machine", machine_md(machine_info()), "",
        "## Memory", "",
        "| | fp32 | int8 | int8 / fp32 |", "|---|---|---|---|",
    ]
    labels = {"weights_mb": "stored weights (MB)",
              "dequant_scratch_mb": "persistent dequant scratch buffer (MB)",
              "peak_decode_mb": "peak extra alloc, 1 decode step @ L=256 (MB)",
              "peak_prefill1024_mb": "peak extra alloc, 1024-token forward (MB)"}
    for k, lab in labels.items():
        a, b = mem["fp32"][k], mem["int8"][k]
        ratio = f"{b / a:.2f}" if a else "n/a"
        lines.append(f"| {lab} | {a:.1f} | {b:.1f} | {ratio} |")
    lines += ["", "## Speed (tokens/s, KV cache on)", "",
              "| | fp32 | int8 | int8 / fp32 |", "|---|---|---|---|"]
    for L in LENGTHS:
        f, q = stats[(L, "fp32")]["median"], stats[(L, "int8")]["median"]
        lines.append(f"| decode step @ L={L} | {ds(L, 'fp32')} | {ds(L, 'int8')} | {f / q:.2f} |")
    f, q = e2e["fp32"]["median"], e2e["int8"]["median"]
    lines.append(f"| generate {N} tokens end to end | {N / f:.2f} | {N / q:.2f} | {f / q:.2f} |")
    lines += ["", f"## Perplexity, WikiText-2 raw test ({full}), ctx 1024, stride {args.stride}", "",
              "| | fp32 | int8 | change |", "|---|---|---|---|",
              f"| perplexity | {ppl['fp32']['ppl']:.3f} | {ppl['int8']['ppl']:.3f} | "
              f"{100 * (ppl['int8']['ppl'] / ppl['fp32']['ppl'] - 1):+.2f}% |",
              f"| tokens scored | {ppl['fp32']['tokens']} | {ppl['int8']['tokens']} | |",
              f"| eval time (s) | {ppl['fp32']['seconds']:.0f} | {ppl['int8']['seconds']:.0f} | |", ""]
    (RESULTS / "int8.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / "int8_ppl.json").write_text(json.dumps(ppl, indent=2))
    print(f"wrote {RESULTS / 'int8.md'}")


if __name__ == "__main__":
    main()
