"""Greedy speculative decoding: a small draft model guesses k tokens, the big target
model checks all of them in ONE forward pass.

Decode is memory-bound: one target forward over k+1 tokens costs about the same as a
forward over 1 token (the weights are read once either way). So if the draft guesses
right, we get several target tokens for the price of one target forward.

One round, with `last` = the newest token (not yet in the target cache):
  1. draft:  feed last (and anything it hasn't seen), then greedily propose d1..dk
  2. target: one forward over [last, d1, ..., dk] -> greedy tokens g0..gk, where g_i is
             the target's choice after d_i (g0 after `last`)
  3. accept the longest prefix with d_{i+1} == g_i, say n tokens, then append g_n
             (the target's correction, or a free bonus token if all k matched)
  4. roll both KV caches back to the accepted tokens (just lower table.length; the
             stale K/V slots are overwritten next round)
Every emitted token is a target greedy token, so the output is exactly the target's
greedy output; the draft only decides how many come out per target forward.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.kv_cache import BlockTable
from engine.torch_model import TorchGPT2


@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    generated: int = 0
    target_forwards: int = 0       # incl. the prefill
    draft_forwards: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.generated / self.rounds if self.rounds else 0.0


class _Seq:
    """One sequence on one model, with its own paged cache."""

    def __init__(self, model: TorchGPT2, max_tokens: int):
        self.model = model
        self.cache = model.new_cache(num_blocks=-(-max_tokens // 16), block_size=16)
        self.table = BlockTable()

    @property
    def length(self) -> int:
        return self.table.length

    def forward(self, toks: list[int], all_logits: bool = False):
        self.cache.reserve(self.table, self.table.length + len(toks))
        out = self.model.forward_batch([(toks, self.table)], self.cache, all_logits=all_logits)
        return out[0]                       # [V] (last token) or [T, V]

    def rollback(self, length: int) -> None:
        assert length <= self.table.length
        self.table.length = length


def greedy_generate(model: TorchGPT2, prompt: list[int], n_new: int) -> list[int]:
    """Baseline: one target forward per token (prefill, then decode)."""
    _check(model, prompt, n_new)
    s = _Seq(model, len(prompt) + n_new)
    logits = s.forward(prompt)
    out = []
    while True:
        out.append(int(logits.argmax()))
        if len(out) == n_new:
            return out
        logits = s.forward([out[-1]])


def speculative_generate(target: TorchGPT2, draft: TorchGPT2, prompt: list[int], n_new: int,
                         k: int, stats: SpecStats | None = None) -> list[int]:
    if k < 1:
        raise ValueError("k must be >= 1")
    if target.hp["n_vocab"] != draft.hp["n_vocab"]:
        raise ValueError("draft and target must share a vocabulary")
    _check(target, prompt, n_new)
    n_ctx = min(target.hp["n_ctx"], draft.hp["n_ctx"])
    stats = stats if stats is not None else SpecStats()
    room = min(n_ctx, len(prompt) + n_new + k + 1)
    T, D = _Seq(target, room), _Seq(draft, room)

    # both caches hold the prompt except its last token; that token is "pending"
    if len(prompt) > 1:
        T.forward(prompt[:-1])
        D.forward(prompt[:-1])
        stats.target_forwards += 1
        stats.draft_forwards += 1
    last = prompt[-1]          # newest token, not yet in the target cache
    pending = [prompt[-1]]     # tokens the draft cache hasn't consumed yet
    out: list[int] = []

    while len(out) < n_new:
        # draft no further than the output length and the context window allow
        kk = min(k, n_new - len(out) - 1, n_ctx - T.length - 1)

        drafts: list[int] = []
        d_before = D.length
        n_pending = len(pending)
        if kk > 0:
            logits = D.forward(pending)
            stats.draft_forwards += 1
            for i in range(kk):
                drafts.append(int(logits.argmax()))
                if i + 1 < kk:
                    logits = D.forward([drafts[-1]])
                    stats.draft_forwards += 1
            # the draft cache now holds pending + drafts[:kk-1]

        t_before = T.length
        g = T.forward([last] + drafts, all_logits=True).argmax(-1).tolist()   # kk+1 target tokens
        stats.target_forwards += 1
        n = 0
        while n < kk and drafts[n] == g[n]:
            n += 1
        new = drafts[:n] + [g[n]]

        T.rollback(t_before + 1 + n)                     # keep `last` + accepted drafts
        if kk > 0:
            consumed = min(n, kk - 1)                    # accepted drafts the draft already fed
            D.rollback(d_before + n_pending + consumed)
            pending = drafts[consumed:n] + [g[n]]
        else:
            pending = pending + [g[n]]                   # draft didn't run; catch up next round
        last = g[n]
        out += new
        stats.rounds += 1
        stats.drafted += kk
        stats.accepted += n
        stats.generated += len(new)
    return out


def _check(model: TorchGPT2, prompt: list[int], n_new: int) -> None:
    if not prompt:
        raise ValueError("empty prompt")
    if n_new < 1:
        raise ValueError("n_new must be >= 1")
    if len(prompt) + n_new > model.hp["n_ctx"]:
        raise ValueError(f"prompt + n_new = {len(prompt) + n_new} exceeds n_ctx {model.hp['n_ctx']}")
