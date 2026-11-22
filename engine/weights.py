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


# ---------------------------------------------------------------- int8 quantization

class QInt8:
    """Weight-only int8 tensor with one fp32 scale per OUTPUT channel.

    q:     int8, same shape as the original weight, values in [-127, 127]
    scale: fp32, shaped to broadcast against q (keepdims), so  w ~= q * scale

    Conv1D weights are [in, out]: output channel = column, scale shape [1, out].
    The tied embedding wte is [vocab, C]; as the output head (x @ wte.T) its output
    channel is a ROW, scale shape [vocab, 1]. Per-row scales also let the embedding
    lookup dequantize only the rows it needs.
    """

    __slots__ = ("q", "scale")

    def __init__(self, q: np.ndarray, scale: np.ndarray):
        self.q, self.scale = q, scale

    @classmethod
    def quantize(cls, w: np.ndarray, out_axis: int) -> "QInt8":
        reduce_axis = 1 - out_axis                      # 2-D weights only
        amax = np.max(np.abs(w), axis=reduce_axis, keepdims=True)
        scale = (amax / 127.0).astype(np.float32)
        scale[scale == 0] = 1.0                         # all-zero channel: avoid 0/0
        q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
        return cls(q, scale)

    def dequantize(self) -> np.ndarray:
        return self.q.astype(np.float32) * self.scale

    @property
    def shape(self):
        return self.q.shape

    @property
    def nbytes(self) -> int:
        return self.q.nbytes + self.scale.nbytes


def quantize_int8(params: dict) -> dict:
    """Return a copy of params with every matmul weight stored as QInt8.

    Quantized: the 4 projection weights per block + wte (embedding/output head),
    i.e. ~99.4% of parameters. Kept fp32: biases, layer norms, wpe (small, and
    position embeddings are added straight into the residual stream).
    """
    def lin(d: dict) -> dict:
        return {"w": QInt8.quantize(d["w"], out_axis=1), "b": d["b"]}

    blocks = []
    for blk in params["blocks"]:
        blocks.append({
            "ln_1": blk["ln_1"],
            "attn": {k: lin(v) for k, v in blk["attn"].items()},
            "ln_2": blk["ln_2"],
            "mlp": {k: lin(v) for k, v in blk["mlp"].items()},
        })
    return {"wte": QInt8.quantize(params["wte"], out_axis=0), "wpe": params["wpe"],
            "blocks": blocks, "ln_f": params["ln_f"]}


def params_nbytes(params) -> int:
    """Bytes held by all weight tensors (fp32 arrays or QInt8)."""
    if isinstance(params, (np.ndarray, QInt8)):
        return params.nbytes
    if isinstance(params, dict):
        return sum(params_nbytes(v) for v in params.values())
    if isinstance(params, list):
        return sum(params_nbytes(v) for v in params)
    return 0
