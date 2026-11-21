"""Load OpenAI's GPT-2 weights (HuggingFace safetensors) into plain NumPy dicts.

GPT-2 uses a `Conv1D` layer, which stores weights as (in_features, out_features),
so `x @ W + b` works directly with no transpose (unlike nn.Linear's (out, in)).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"


def download(model_name: str = "gpt2") -> Path:
    """Fetch config.json + model.safetensors only (never pickled .bin files)."""
    from huggingface_hub import hf_hub_download

    out = WEIGHTS_DIR / model_name
    for fname in ("config.json", "model.safetensors"):
        hf_hub_download(model_name, fname, local_dir=out)
    return out


def load_gpt2(model_name: str = "gpt2", dtype=np.float32) -> tuple[dict, dict]:
    """Return (hparams, params).

    params = {
      "wte": [n_vocab, n_embd], "wpe": [n_ctx, n_embd],
      "blocks": [ {ln_1:{g,b}, attn:{c_attn:{w,b}, c_proj:{w,b}},
                   ln_2:{g,b}, mlp:{c_fc:{w,b}, c_proj:{w,b}}} , ...],
      "ln_f": {g, b},
    }
    """
    path = WEIGHTS_DIR / model_name
    if not (path / "model.safetensors").exists():
        path = download(model_name)

    cfg = json.loads((path / "config.json").read_text())
    hparams = {
        "n_vocab": cfg["vocab_size"],
        "n_ctx": cfg["n_positions"],
        "n_embd": cfg["n_embd"],
        "n_head": cfg["n_head"],
        "n_layer": cfg["n_layer"],
        "eps": cfg.get("layer_norm_epsilon", 1e-5),
    }

    raw = load_file(str(path / "model.safetensors"))
    # Some checkpoints prefix keys with "transformer."; normalise.
    raw = {k.removeprefix("transformer."): v for k, v in raw.items()}

    def t(name: str) -> np.ndarray:
        return np.ascontiguousarray(raw[name], dtype=dtype)

    def ln(prefix: str) -> dict:
        return {"g": t(f"{prefix}.weight"), "b": t(f"{prefix}.bias")}

    def lin(prefix: str) -> dict:  # Conv1D: weight already [in, out]
        return {"w": t(f"{prefix}.weight"), "b": t(f"{prefix}.bias")}

    blocks = []
    for i in range(hparams["n_layer"]):
        p = f"h.{i}"
        blocks.append({
            "ln_1": ln(f"{p}.ln_1"),
            "attn": {"c_attn": lin(f"{p}.attn.c_attn"), "c_proj": lin(f"{p}.attn.c_proj")},
            "ln_2": ln(f"{p}.ln_2"),
            "mlp": {"c_fc": lin(f"{p}.mlp.c_fc"), "c_proj": lin(f"{p}.mlp.c_proj")},
        })

    params = {"wte": t("wte.weight"), "wpe": t("wpe.weight"), "blocks": blocks, "ln_f": ln("ln_f")}
    return hparams, params
