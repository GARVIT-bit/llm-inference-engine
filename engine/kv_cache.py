"""Block-based (PagedAttention-style) KV cache.

Instead of one contiguous n_ctx-long buffer per sequence, K/V live in a shared pool
of fixed-size blocks (16 tokens each). A sequence owns a *block table*: the list of
block ids holding its tokens, in order. Token at position p lives in block
table.blocks[p // block_size] at offset p % block_size. Blocks are allocated as a
sequence grows and returned to the pool when it finishes, so memory is used only for
tokens that exist, and any free block can serve any sequence.
"""
from __future__ import annotations

import math

import torch


class OutOfBlocks(RuntimeError):
    pass


class BlockAllocator:
    """Free list of block ids. Detects double frees."""

    def __init__(self, num_blocks: int):
        if num_blocks < 1:
            raise ValueError("need at least one block")
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))   # pop() hands out 0, 1, 2, ...
        self._used: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocks(f"need {n} blocks, {len(self._free)} free")
        out = [self._free.pop() for _ in range(n)]
        self._used.update(out)
        return out

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if b not in self._used:
                raise ValueError(f"block {b} is not allocated (double free?)")
            self._used.remove(b)
            self._free.append(b)


class BlockTable:
    """One sequence's view of the cache: which blocks it owns and how many tokens are stored."""

    __slots__ = ("blocks", "length")

    def __init__(self):
        self.blocks: list[int] = []
        self.length = 0          # tokens whose K/V are already written


class PagedKVCache:
    """K/V pool: k, v of shape [n_layer, num_blocks, block_size, n_head, head_dim]."""

    def __init__(self, hparams: dict, num_blocks: int, block_size: int = 16,
                 device="cpu", dtype=torch.float32):
        H = hparams["n_head"]
        D = hparams["n_embd"] // H
        shape = (hparams["n_layer"], num_blocks, block_size, H, D)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)

    @property
    def num_blocks(self) -> int:
        return self.allocator.num_blocks

    @property
    def num_free(self) -> int:
        return self.allocator.num_free

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def blocks_for(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / self.block_size)

    def blocks_needed(self, table: BlockTable, n_tokens: int) -> int:
        """Extra blocks `table` needs to hold n_tokens in total."""
        return max(0, self.blocks_for(n_tokens) - len(table.blocks))

    def can_fit(self, table: BlockTable, n_tokens: int) -> bool:
        return self.blocks_needed(table, n_tokens) <= self.num_free

    def reserve(self, table: BlockTable, n_tokens: int) -> None:
        """Make sure `table` has blocks for n_tokens tokens (raises OutOfBlocks)."""
        need = self.blocks_needed(table, n_tokens)
        if need:
            table.blocks.extend(self.allocator.allocate(need))

    def free(self, table: BlockTable) -> None:
        self.allocator.free(table.blocks)
        table.blocks = []
        table.length = 0

    def slots(self, table: BlockTable, start: int, end: int) -> list[int]:
        """Flat slot index (block * block_size + offset) of positions start..end-1."""
        bs = self.block_size
        if end > len(table.blocks) * bs:
            raise ValueError(f"positions up to {end} need {self.blocks_for(end)} blocks, "
                             f"table has {len(table.blocks)}")
        return [table.blocks[p // bs] * bs + p % bs for p in range(start, end)]
