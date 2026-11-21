"""Tokenizer: tiktoken's `gpt2` encoding (same BPE vocab OpenAI trained GPT-2 with)."""
from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def get_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def encode(text: str) -> list[int]:
    return get_tokenizer().encode(text, allowed_special={"<|endoftext|>"})


def decode(ids) -> str:
    return get_tokenizer().decode(list(ids))
