"""Load test: static vs continuous batching through the real HTTP server.

For each policy, start `python -m server.app --policy <p>` on port 8001, send N
concurrent streaming /v1/completions requests (prompt and output lengths drawn from
a fixed seed), and measure from the client side:
  - throughput = generated tokens / wall time (first request sent -> last one done)
  - TTFT       = time from sending a request to receiving its first token
  - TPOT       = (end - first token) / (tokens - 1), time per output token after the first
Policies are run round-robin (static, continuous, static, ...) to share CPU drift.

    .venv\\Scripts\\python.exe bench\\batching_bench.py            # 64 requests x 3 runs per policy
    .venv\\Scripts\\python.exe bench\\batching_bench.py --requests 16 --runs 1 --max-out 64   # smoke
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import torch  # noqa: F401  (machine_info reports torch version / GPU)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.common import RESULTS, machine_info, machine_md  # noqa: E402
from engine.data import wikitext2_test_text  # noqa: E402
from engine.tokenizer import encode  # noqa: E402

POLICIES = ["static", "continuous"]


def make_workload(n: int, seed: int, min_in: int, max_in: int, min_out: int, max_out: int):
    rng = np.random.default_rng(seed)
    text = encode(wikitext2_test_text())
    work = []
    for _ in range(n):
        L = int(rng.integers(min_in, max_in + 1))
        start = int(rng.integers(0, len(text) - L))
        work.append((text[start:start + L], int(rng.integers(min_out, max_out + 1))))
    return work


def start_server(policy: str, port: int, args) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "server.app", "--policy", policy, "--port", str(port),
           "--max-batch", str(args.max_batch), "--num-blocks", str(args.num_blocks)]
    if args.device:
        cmd += ["--device", args.device]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    while time.time() - t0 < 120:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited: {proc.stdout.read()}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError("server did not come up")


async def one_request(client, url, prompt_ids, max_tokens, t_start):
    sent = time.perf_counter()
    first = None
    async with client.stream("POST", url, json={
            "model": "gpt2", "prompt": prompt_ids, "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "ignore_eos": True}) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            obj = json.loads(line[6:])
            if "error" in obj:
                raise RuntimeError(obj["error"])
            if first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    return {"sent": sent - t_start, "ttft": first - sent, "end": end - t_start,
            "latency": end - sent}


async def load_test(port: int, work) -> dict:
    url = f"http://127.0.0.1:{port}/v1/completions"
    limits = httpx.Limits(max_connections=len(work) + 8)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        # warm-up (not measured)
        await one_request(client, url, work[0][0], 4, time.perf_counter())
        m0 = (await client.get(f"http://127.0.0.1:{port}/metrics")).text
        t0 = time.perf_counter()
        res = await asyncio.gather(*(one_request(client, url, p, n, t0) for p, n in work))
        wall = time.perf_counter() - t0
        m1 = (await client.get(f"http://127.0.0.1:{port}/metrics")).text
    def val(text, name):
        return float(next(l.split()[1] for l in text.splitlines() if l.startswith(name + " ")))
    gen = val(m1, "llm_generated_tokens_total") - val(m0, "llm_generated_tokens_total")
    steps = val(m1, "llm_decode_steps_total") - val(m0, "llm_decode_steps_total")
    ttft = sorted(r["ttft"] for r in res)
    tpot = sorted((r["latency"] - r["ttft"]) / max(1, n - 1) for r, (_, n) in zip(res, work))
    q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))]
    expected = sum(n for _, n in work)
    assert gen == expected, (gen, expected)
    return {"wall_s": wall, "tokens": gen, "tok_per_s": gen / wall, "decode_steps": steps,
            "ttft_p50": q(ttft, 0.5), "ttft_p90": q(ttft, 0.9), "tpot_p50": q(tpot, 0.5),
            "preemptions": val(m1, "llm_preemptions_total") - val(m0, "llm_preemptions_total")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--num-blocks", type=int, default=384)
    ap.add_argument("--min-in", type=int, default=16)
    ap.add_argument("--max-in", type=int, default=128)
    ap.add_argument("--min-out", type=int, default=16)
    ap.add_argument("--max-out", type=int, default=256)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="server device (default: cuda if available)")
    ap.add_argument("--out", default="batching", help="results file stem in bench/results/")
    args = ap.parse_args()

    info = machine_info()
    if info["power"] == "battery":
        print("WARNING: on battery power; numbers will be low and noisy", flush=True)
    work = make_workload(args.requests, args.seed, args.min_in, args.max_in, args.min_out, args.max_out)
    print(f"{len(work)} requests, {sum(len(p) for p, _ in work)} prompt tokens, "
          f"{sum(n for _, n in work)} output tokens", flush=True)

    runs = {p: [] for p in POLICIES}
    for i in range(args.runs):
        for policy in POLICIES:
            proc = start_server(policy, args.port, args)
            try:
                r = asyncio.run(load_test(args.port, work))
            finally:
                proc.terminate()
                proc.wait(30)
            runs[policy].append(r)
            print(f"run {i + 1} {policy:10s}: {r['tok_per_s']:.1f} tok/s, wall {r['wall_s']:.1f}s, "
                  f"TTFT p50 {r['ttft_p50']:.2f}s p90 {r['ttft_p90']:.2f}s, "
                  f"TPOT p50 {1000 * r['tpot_p50']:.0f} ms, decode steps {r['decode_steps']:.0f}, "
                  f"preemptions {r['preemptions']:.0f}", flush=True)

    keys = ["tok_per_s", "wall_s", "ttft_p50", "ttft_p90", "tpot_p50", "decode_steps", "preemptions"]
    summary = {p: {k: {"median": statistics.median(r[k] for r in runs[p]),
                       "min": min(r[k] for r in runs[p]), "max": max(r[k] for r in runs[p])}
                   for k in keys} for p in POLICIES}

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{args.out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "run"] + keys)
        for p in POLICIES:
            for i, r in enumerate(runs[p]):
                w.writerow([p, i + 1] + [round(r[k], 4) for k in keys])

    def cell(p, k, fmt, scale=1.0):
        s = summary[p][k]
        return f"{s['median'] * scale:{fmt}} ({s['min'] * scale:{fmt}} to {s['max'] * scale:{fmt}})"
    S, C = summary["static"], summary["continuous"]
    lines = [
        "# Static vs continuous batching (HTTP load test, GPT-2 124M, PyTorch)", "",
        f"Generated by `bench/batching_bench.py`. {args.requests} concurrent streaming requests "
        f"sent at once; prompt lengths {args.min_in}-{args.max_in} tokens (WikiText-2 test slices), "
        f"output lengths {args.min_out}-{args.max_out} tokens (uniform, seed {args.seed}, greedy, "
        f"`ignore_eos`). Server on {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}: max batch {args.max_batch}, KV cache {args.num_blocks} blocks x 16 tokens. "
        f"{args.runs} runs per policy, round-robin, 1 warm-up request each; median (min to max). "
        "Client-side timing.", "",
        "## Machine", machine_md(info), "",
        "## Results", "",
        "| | static | continuous | continuous / static |", "|---|---|---|---|",
        f"| throughput (generated tok/s) | {cell('static', 'tok_per_s', '.1f')} | "
        f"{cell('continuous', 'tok_per_s', '.1f')} | {C['tok_per_s']['median'] / S['tok_per_s']['median']:.2f}x |",
        f"| wall time (s) | {cell('static', 'wall_s', '.1f')} | {cell('continuous', 'wall_s', '.1f')} | |",
        f"| TTFT p50 (s) | {cell('static', 'ttft_p50', '.2f')} | {cell('continuous', 'ttft_p50', '.2f')} | "
        f"{C['ttft_p50']['median'] / S['ttft_p50']['median']:.2f}x |",
        f"| TTFT p90 (s) | {cell('static', 'ttft_p90', '.2f')} | {cell('continuous', 'ttft_p90', '.2f')} | |",
        f"| time per output token p50 (ms) | {cell('static', 'tpot_p50', '.0f', 1000)} | "
        f"{cell('continuous', 'tpot_p50', '.0f', 1000)} | |",
        f"| batched decode steps | {cell('static', 'decode_steps', '.0f')} | "
        f"{cell('continuous', 'decode_steps', '.0f')} | |",
        f"| preemptions | {cell('static', 'preemptions', '.0f')} | {cell('continuous', 'preemptions', '.0f')} | |",
        "",
    ]
    (RESULTS / f"{args.out}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS / (args.out + '.md')}")


if __name__ == "__main__":
    main()
