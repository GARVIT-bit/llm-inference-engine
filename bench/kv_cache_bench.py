"""Benchmark: NumPy GPT-2 (124M, fp32, CPU) with vs without KV cache.

Two measurements:
  A. decode-step throughput at context length L: time to produce ONE next token when
     L-1 tokens came before. Cached = one decode step against a cache holding L-1
     tokens. Uncached = a full forward over all L tokens. tokens/s = 1 / median.
  B. end-to-end greedy generation of N new tokens from a short prompt
     (prefill + N steps), tokens/s = N / median total time.
     Uncached N=1024 takes tens of minutes per run, so it only runs with --full.

    .venv\\Scripts\\python.exe bench\\kv_cache_bench.py            # writes bench/results/kv_cache.{csv,md}
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.common import RESULTS, machine_info, machine_md, time_interleaved  # noqa: E402
from engine.numpy_model import KVCache, generate, gpt2  # noqa: E402
from engine.tokenizer import encode  # noqa: E402
from engine.weights import load_gpt2  # noqa: E402

LENGTHS = [32, 256, 1024]
PROMPT = "Alan Turing theorized that computers would one day become"
SEED = 0


def decode_step_fns(hp, p, L: int) -> dict:
    rng = np.random.default_rng(SEED)
    ids = rng.integers(0, hp["n_vocab"], size=L).tolist()

    cache = KVCache(hp)
    gpt2(ids[:-1], p, hp, cache=cache, last_only=True)  # fill L-1 tokens once

    def cached_step():
        cache.length = L - 1                              # overwrite slot L-1 each run
        gpt2(ids[-1:], p, hp, cache=cache, last_only=True)

    def uncached_step():
        gpt2(ids, p, hp, last_only=True)

    return {(L, "cached"): cached_step, (L, "uncached"): uncached_step}


def e2e_fn(hp, p, n_new: int, use_cache: bool):
    ids = encode(PROMPT)
    return lambda: generate(ids, p, hp, n_new, use_cache=use_cache)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5, help="timed runs for decode-step bench")
    ap.add_argument("--e2e-runs", type=int, default=3)
    ap.add_argument("--full", action="store_true", help="also run uncached end-to-end at N=1024 (slow)")
    args = ap.parse_args()

    hp, p = load_gpt2("gpt2")
    prompt_len = len(encode(PROMPT))
    rows = []

    # A. decode step. Cached and uncached are timed as two separate round-robin groups:
    # on a power-limited laptop, a ~60 ms cached step timed right after a multi-second
    # uncached forward runs throttled, which skewed the cached-vs-L comparison.
    # Cached steps are short and noisy, so they get 4x the samples.
    fns = {}
    for L in LENGTHS:
        fns.update(decode_step_fns(hp, p, L))
    stats = time_interleaved({k: f for k, f in fns.items() if k[1] == "cached"}, runs=4 * args.runs)
    stats |= time_interleaved({k: f for k, f in fns.items() if k[1] == "uncached"}, runs=args.runs)
    for L in LENGTHS:
        c, u = stats[(L, "cached")], stats[(L, "uncached")]
        for mode, s in (("cached", c), ("uncached", u)):
            rows.append({"bench": "decode_step", "tokens": L, "mode": mode,
                         "median_s": s["median"], "min_s": s["min"], "max_s": s["max"],
                         "tok_per_s": 1 / s["median"], "runs": s["runs"]})
        print(f"decode step @ L={L:5d}: cached {1/c['median']:7.2f} tok/s | uncached {1/u['median']:7.2f} tok/s "
              f"| speedup {u['median']/c['median']:.1f}x", flush=True)

    # B. end-to-end: also round-robin. Uncached 256 is minutes per run, so it gets no warm-up
    # (section A already warmed the process); uncached 1014 only runs with --full.
    fns, n_eff = {}, {}
    for N in LENGTHS:
        n_eff[N] = min(N, hp["n_ctx"] - prompt_len)
        fns[(N, "cached")] = e2e_fn(hp, p, n_eff[N], True)
        if N < 1024 or args.full:
            fns[(N, "uncached")] = e2e_fn(hp, p, n_eff[N], False)
    warm = {k for k in fns if k[1] == "cached" or k[0] <= 32}
    stats = time_interleaved(fns, runs=args.e2e_runs, warmup=warm)
    for N in LENGTHS:
        for mode in ("cached", "uncached"):
            s = stats.get((N, mode))
            if s is None:
                rows.append({"bench": "e2e_generate", "tokens": n_eff[N], "mode": mode, "median_s": "",
                             "min_s": "", "max_s": "", "tok_per_s": "", "runs": 0})
                print(f"e2e N={n_eff[N]}: {mode} skipped (use --full)", flush=True)
                continue
            rows.append({"bench": "e2e_generate", "tokens": n_eff[N], "mode": mode,
                         "median_s": s["median"], "min_s": s["min"], "max_s": s["max"],
                         "tok_per_s": n_eff[N] / s["median"], "runs": s["runs"]})
            print(f"e2e N={n_eff[N]}: {mode:8s} {n_eff[N]/s['median']:7.2f} tok/s "
                  f"({s['median']:.1f}s, runs={s['runs']})", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "kv_cache.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def fmt(x, spec):
        return format(x, spec) if x != "" else "n/a"

    lines = ["# KV cache benchmark (NumPy GPT-2 124M, fp32, CPU)", "",
             "Generated by `bench/kv_cache_bench.py`. Greedy decoding, prompt "
             f"({prompt_len} tokens): \"{PROMPT}\". Seed {SEED} for decode-step inputs. "
             "1 warm-up call per config (except uncached N >= 256), then configs are timed "
             "round-robin (one sample of each per round) so CPU clock drift hits all equally. "
             f"Section A times cached ({4 * args.runs} samples) and uncached ({args.runs} samples) "
             "as separate groups so cheap steps never follow heavy ones. "
             "Reported: median (min to max).", "",
             "## Machine", machine_md(machine_info()), "",
             "## A. One decode step at context length L", "",
             "| L | cached tok/s | uncached tok/s | speedup |", "|---|---|---|---|"]
    ds = {(r["tokens"], r["mode"]): r for r in rows if r["bench"] == "decode_step"}
    for L in LENGTHS:
        c, u = ds[(L, "cached")], ds[(L, "uncached")]
        lines.append(f"| {L} | {c['tok_per_s']:.2f} ({1/c['max_s']:.2f} to {1/c['min_s']:.2f}) "
                     f"| {u['tok_per_s']:.2f} ({1/u['max_s']:.2f} to {1/u['min_s']:.2f}) "
                     f"| {u['median_s']/c['median_s']:.1f}x |")
    lines += ["", "## B. End-to-end generation of N new tokens (prefill + N steps)", "",
              "| N | cached tok/s | uncached tok/s | speedup | runs (cached/uncached) |", "|---|---|---|---|---|"]
    e2e = {(r["tokens"], r["mode"]): r for r in rows if r["bench"] == "e2e_generate"}

    def cell(r, n):
        if r["median_s"] == "":
            return "not run (`--full`)"
        return f"{n / r['median_s']:.2f} ({n / r['max_s']:.2f} to {n / r['min_s']:.2f})"

    for N in sorted({k[0] for k in e2e}):
        c, u = e2e[(N, "cached")], e2e[(N, "uncached")]
        sp = f"{u['median_s']/c['median_s']:.1f}x" if u["median_s"] != "" else "n/a"
        lines.append(f"| {N} | {cell(c, N)} | {cell(u, N)} | {sp} | {c['runs']}/{u['runs']} |")
    lines += ["", "Note: laptop CPU with turbo boost and background load; run-to-run spread is large "
              "(see min to max). Compare speedups more than absolute tok/s.", ""]
    (RESULTS / "kv_cache.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS / 'kv_cache.csv'} and kv_cache.md")


if __name__ == "__main__":
    main()
