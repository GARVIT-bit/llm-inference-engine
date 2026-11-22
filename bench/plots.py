"""Plots for the README, drawn only from the committed files in bench/results/.

    .venv\\Scripts\\python.exe bench\\plots.py        # writes docs/img/*.png

No numbers are typed in here: every point is read from a results file that a bench
script wrote. The exact values are also in the README tables (the plots are a view).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "bench" / "results"
OUT = ROOT / "docs" / "img"

# validated categorical palette (fixed order) + neutral ink, light surface
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 7, "axes.axisbelow": True,
})


def rows(name):
    with open(RES / name, newline="") as f:
        return list(csv.DictReader(f))


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                fontsize=9, color=INK2)
    ax.plot([x], [y], "o", color=color, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def kv_cache():
    r = [x for x in rows("kv_cache.csv") if x["bench"] == "decode_step"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for mode, color in (("cached", C1), ("uncached", C2)):
        pts = sorted((int(x["tokens"]), float(x["tok_per_s"])) for x in r if x["mode"] == mode)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=color, label=mode, markeredgecolor=SURFACE, markeredgewidth=2)
        label_end(ax, xs[-1], ys[-1], f"{mode} {ys[-1]:.2f} tok/s", color)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([32, 256, 1024]); ax.set_xticklabels(["32", "256", "1024"])
    ax.set_xlabel("context length L (tokens)"); ax.set_ylabel("decode tokens/s (log)")
    ax.set_title("KV cache: one decode step, NumPy GPT-2 124M, laptop CPU")
    ax.legend(loc="lower left"); ax.set_xlim(right=4096)
    save(fig, "kv_cache.png")


def batching():
    def med(name, policy, key):
        v = sorted(float(x[key]) for x in rows(name) if x["policy"] == policy)
        return v[len(v) // 2]
    setups = [("laptop CPU\nmax batch 16", "batching.csv"), ("T4 GPU\nmax batch 16", "batching_t4.csv")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, key, title, fmt in ((axes[0], "tok_per_s", "Throughput (tok/s, higher is better)", "{:.0f}"),
                                (axes[1], "ttft_p50", "Median time to first token (s, lower is better)", "{:.1f}")):
        for i, (lab, f) in enumerate(setups):
            for j, (pol, color) in enumerate((("static", C2), ("continuous", C1))):
                v = med(f, pol, key)
                x = i + (j - 0.5) * 0.36
                ax.bar(x, v, width=0.34, color=color, label=pol if i == 0 else None)
                ax.annotate(fmt.format(v), (x, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=9, color=INK2)
        ax.set_xticks(range(len(setups))); ax.set_xticklabels([s for s, _ in setups])
        ax.set_title(title); ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left")
    fig.suptitle("Static vs continuous batching: 64 concurrent HTTP requests", fontweight="bold")
    fig.tight_layout()
    save(fig, "batching.png")


def speculative():
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, lab, color, dy in (("speculative.csv", "laptop CPU (fp32)", C2, -14), ("speculative_t4.csv", "T4 GPU (fp16)", C1, -14)):
        pts = sorted((int(x["k"]), float(x["speedup_vs_target"])) for x in rows(name) if x["k"])
        xs, ys = zip(*pts)
        best = max(pts, key=lambda p: p[1])
        ax.plot(xs, ys, "-o", color=color, label=f"{lab}: best {best[1]:.2f}x at k={best[0]}",
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
    ax.annotate("target greedy = 1.0x", (3, 1.0), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8, color=INK2)
    ax.set_ylim(0.72, 1.3)
    ax.set_xticks([2, 4, 6, 8]); ax.set_xlabel("draft length k")
    ax.set_ylabel("speedup vs gpt2-medium greedy")
    ax.set_title("Speculative decoding: gpt2 drafts for gpt2-medium (output identical)")
    ax.legend(loc="lower left")
    save(fig, "speculative.png")


def attention():
    r = rows("attention_t4.csv")
    md = (RES / "attention_tune_t4.md").read_text(encoding="utf-8")
    tuned = {}
    for line in md.split("## Best config", 1)[1].splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            tuned[int(m.group(1))] = float(m.group(4))
    series = [
        ("PyTorch naive", C2, {int(x["seq_len"]): float(x["ms"]) for x in r if x["method"] == "pytorch naive"}),
        ("Triton, first config", C4, {int(x["seq_len"]): float(x["ms"]) for x in r if x["method"] == "triton fused"}),
        ("Triton, tuned", C1, tuned),
        ("torch SDPA (reference)", C3, {int(x["seq_len"]): float(x["ms"]) for x in r if x["method"].startswith("torch SDPA")}),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    for lab, color, d in series:
        xs = sorted(d); ys = [d[x] for x in xs]
        ax.plot(xs, ys, "-o", color=color, label=lab, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{ys[-1]:.1f} ms", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([256, 512, 1024, 2048, 4096]); ax.set_xticklabels(["256", "512", "1K", "2K", "4K"])
    ax.set_xlim(right=9000)
    ax.set_xlabel("sequence length"); ax.set_ylabel("time per call (ms, log)")
    ax.set_title("Time (lower is better)"); ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    for lab, color, key in (("PyTorch naive", C2, "pytorch naive"), ("Triton fused", C1, "triton fused")):
        d = sorted((int(x["seq_len"]), float(x["peak_mb"])) for x in r if x["method"] == key)
        xs, ys = zip(*d)
        ax.plot(xs, ys, "-o", color=color, label=lab, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{ys[-1]:,.0f} MB", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([256, 512, 1024, 2048, 4096]); ax.set_xticklabels(["256", "512", "1K", "2K", "4K"])
    ax.set_xlim(right=9000)
    ax.set_xlabel("sequence length"); ax.set_ylabel("peak extra GPU memory (MB, log)")
    ax.set_title("Memory (lower is better)"); ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Causal attention on a T4 (fp16, batch 2 x 12 heads x D 64)", fontweight="bold")
    fig.tight_layout()
    save(fig, "attention.png")


def int8():
    r = {(x["model"], x["metric"]): float(x["value"]) for x in rows("int8.csv")}
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, metric, title, fmt in (
            (axes[0], "weights_mb", "Stored weights (MB)", "{:.0f}"),
            (axes[1], "wikitext2_test_ppl", "WikiText-2 perplexity", "{:.2f}"),
            (axes[2], "tok_per_s@N=256", "Generate 256 tokens (tok/s)", "{:.1f}")):
        for i, (m, color) in enumerate((("fp32", C2), ("int8", C1))):
            v = r[(m, metric)]
            ax.bar(i, v, width=0.6, color=color)
            ax.annotate(fmt.format(v), (i, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=9, color=INK2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["fp32", "int8"]); ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    fig.suptitle("int8 weight-only quantization, NumPy GPT-2 124M, laptop CPU", fontweight="bold")
    fig.tight_layout()
    save(fig, "int8.png")


if __name__ == "__main__":
    kv_cache()
    int8()
    batching()
    speculative()
    attention()
