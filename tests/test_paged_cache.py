"""Block allocator and block tables."""
import pytest
import torch

from engine.kv_cache import BlockAllocator, BlockTable, OutOfBlocks, PagedKVCache

HP = {"n_layer": 2, "n_head": 2, "n_embd": 8, "n_ctx": 64}


def test_allocator_alloc_free_reuse():
    a = BlockAllocator(4)
    x = a.allocate(3)
    assert x == [0, 1, 2] and a.num_free == 1
    a.free([1])
    assert a.num_free == 2
    assert sorted(a.allocate(2)) == [1, 3]
    assert a.num_free == 0


def test_allocator_errors():
    a = BlockAllocator(2)
    with pytest.raises(OutOfBlocks):
        a.allocate(3)
    b = a.allocate(1)
    a.free(b)
    with pytest.raises(ValueError):
        a.free(b)                      # double free
    with pytest.raises(ValueError):
        a.free([1])                    # never allocated


def test_reserve_grows_by_whole_blocks_and_free_returns_them():
    c = PagedKVCache(HP, num_blocks=10, block_size=16)
    assert c.k.shape == (2, 10, 16, 2, 4)
    assert c.nbytes == 2 * 2 * 10 * 16 * 8 * 4
    t = BlockTable()
    assert c.blocks_needed(t, 1) == 1 and c.blocks_needed(t, 16) == 1 and c.blocks_needed(t, 17) == 2
    c.reserve(t, 16)
    assert len(t.blocks) == 1
    c.reserve(t, 16)                   # already enough: no-op
    assert len(t.blocks) == 1
    c.reserve(t, 33)
    assert len(t.blocks) == 3 and c.num_free == 7
    t.length = 33
    c.free(t)
    assert t.blocks == [] and t.length == 0 and c.num_free == 10


def test_slots_follow_block_table():
    c = PagedKVCache(HP, num_blocks=8, block_size=4)
    t = BlockTable()
    t.blocks = [5, 2]
    assert c.slots(t, 0, 6) == [20, 21, 22, 23, 8, 9]
    with pytest.raises(ValueError):
        c.slots(t, 0, 9)


def test_can_fit_and_out_of_blocks():
    c = PagedKVCache(HP, num_blocks=2, block_size=16)
    t = BlockTable()
    assert c.can_fit(t, 32) and not c.can_fit(t, 33)
    with pytest.raises(OutOfBlocks):
        c.reserve(t, 33)
    assert t.blocks == [] and c.num_free == 2   # failed reserve takes nothing


def test_dtype_and_device():
    c = PagedKVCache(HP, num_blocks=1, dtype=torch.float16)
    assert c.k.dtype == torch.float16 and c.k.device.type == "cpu"
