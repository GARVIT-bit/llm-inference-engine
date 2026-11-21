"""CLI: greedy-generate with the NumPy GPT-2.

    .venv\\Scripts\\python.exe generate.py "Alan Turing theorized that computers would one day become" --n 20
"""
import argparse
import time

from engine.numpy_model import generate
from engine.tokenizer import decode, encode
from engine.weights import load_gpt2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--n", type=int, default=20, help="tokens to generate")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()

    hparams, params = load_gpt2(args.model)
    ids = encode(args.prompt)
    t0 = time.perf_counter()
    out = generate(ids, params, hparams, args.n)
    dt = time.perf_counter() - t0
    print(args.prompt + decode(out))
    print(f"\n[{args.n} tokens in {dt:.2f}s, {args.n / dt:.1f} tok/s, no KV cache]")


if __name__ == "__main__":
    main()
