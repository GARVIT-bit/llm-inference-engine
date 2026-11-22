"""Batched token sampling: greedy, temperature, top-p (nucleus). Each row has its own settings."""
from __future__ import annotations

import torch


def sample(logits: torch.Tensor, temperature, top_p, generator: torch.Generator | None = None) -> torch.Tensor:
    """logits [B, V]; temperature, top_p: one float per row. Returns token ids [B].

    temperature == 0  -> greedy (argmax).
    temperature  > 0  -> sample from softmax(logits / temperature) ...
    top_p < 1         -> ... restricted to the smallest set of most-likely tokens whose
                         probability mass reaches top_p (the top token is always kept).
    """
    logits = logits.float()
    dev = logits.device
    temp = torch.as_tensor(temperature, dtype=torch.float32, device=dev).reshape(-1)
    top_p = torch.as_tensor(top_p, dtype=torch.float32, device=dev).reshape(-1)
    if temp.shape[0] != logits.shape[0] or top_p.shape[0] != logits.shape[0]:
        raise ValueError("need one temperature and one top_p per row")
    if (temp < 0).any() or (top_p <= 0).any() or (top_p > 1).any():
        raise ValueError("temperature must be >= 0 and top_p in (0, 1]")

    out = logits.argmax(-1)
    rows = (temp > 0).nonzero().squeeze(-1)
    if rows.numel() == 0:
        return out
    probs = torch.softmax(logits[rows] / temp[rows, None], dim=-1)
    p_sorted, idx = probs.sort(dim=-1, descending=True)
    # drop a token if the mass of the tokens strictly ahead of it already reaches top_p
    ahead = p_sorted.cumsum(-1) - p_sorted
    p_sorted = p_sorted.masked_fill(ahead >= top_p[rows, None], 0.0)
    choice = torch.multinomial(p_sorted, 1, generator=generator)    # renormalises internally
    out[rows] = idx.gather(-1, choice).squeeze(-1)
    return out
