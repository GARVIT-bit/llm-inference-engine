"""Continuous batching: batching must not change results, blocks must always come back."""
import threading

import pytest
import torch

from engine.numpy_model import generate as np_generate
from engine.scheduler import EOS, LLMEngine, Request, SamplingParams, Scheduler
from engine.tokenizer import encode
from engine.torch_model import TorchGPT2

PROMPTS = [
    "Alan Turing theorized that computers would one day become",
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Once upon a time, in a small village by the sea, there lived",
    "1, 2, 3, 4, 5,",
]


@pytest.fixture(scope="module")
def tm(gpt2_small):
    hp, p = gpt2_small
    return TorchGPT2(p, hp, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def reference(gpt2_small):
    """Greedy continuations from the NumPy engine, one prompt at a time."""
    hp, p = gpt2_small
    return {pr: np_generate(encode(pr), p, hp, 24, use_cache=True) for pr in PROMPTS}


def greedy(n):
    return SamplingParams(max_tokens=n, temperature=0.0, stop_at_eos=False)


def run(tm, reqs, **kw):
    sched = Scheduler(tm, tm.new_cache(kw.pop("num_blocks", 64), kw.pop("block_size", 16)), **kw)
    for r in reqs:
        sched.add(r)
    steps = 0
    while sched.has_work:
        sched.step()
        steps += 1
        assert steps < 10_000
    return sched


@pytest.mark.parametrize("policy", ["continuous", "static"])
def test_batched_greedy_equals_single_sequence_greedy(tm, reference, policy):
    lengths = [24, 5, 17, 24, 9]                     # different lengths -> finish at different steps
    reqs = [Request(encode(p), greedy(n)) for p, n in zip(PROMPTS, lengths)]
    sched = run(tm, reqs, max_batch=3, policy=policy)
    for r, p, n in zip(reqs, PROMPTS, lengths):
        assert r.finish_reason == "length"
        assert r.output_ids == reference[p][:n], p
    assert sched.cache.num_free == sched.cache.num_blocks         # every block returned
    assert sched.stats["generated_tokens"] == sum(lengths)


def test_continuous_refills_slots_static_waits(tm):
    """max_batch=2, request 0 is short: continuous admits request 2 as soon as 0 finishes,
    static waits for request 1 (long) to finish too."""
    def make():
        return [Request(encode(PROMPTS[i]), greedy(n)) for i, n in ((0, 2), (1, 12), (2, 3))]
    for policy, expect_overlap in (("continuous", True), ("static", False)):
        reqs = make()
        sched = Scheduler(tm, tm.new_cache(64), max_batch=2, policy=policy)
        for r in reqs:
            sched.add(r)
        overlapped = False
        while sched.has_work:
            sched.step()
            if reqs[1].status == "running" and reqs[2].status in ("running", "finished"):
                overlapped = True
        assert overlapped == expect_overlap, policy
        if policy == "static":
            assert reqs[2].t_first_token > reqs[1].t_finish


def test_preemption_under_block_pressure_keeps_outputs(tm, reference):
    # 4-token blocks, 24 blocks = 96 token slots for 5 sequences of ~10-16 prompt + 24
    # generated tokens: running all at once needs ~170 slots, so the scheduler must preempt
    reqs = [Request(encode(p), greedy(24)) for p in PROMPTS]
    sched = run(tm, reqs, num_blocks=24, block_size=4, max_batch=5)
    assert sched.stats["preemptions"] > 0
    for r, p in zip(reqs, PROMPTS):
        assert r.output_ids == reference[p], (p, r.preemptions)
    assert sched.cache.num_free == 24


def test_eos_stops_and_frees(tm):
    # GPT-2 rarely emits EOS on cue, so use a stub model whose logits always favour EOS
    class Stub:
        hp = {"n_ctx": 1024}
        device = torch.device("cpu")

        def forward_batch(self, seqs, cache, all_logits=False):
            for toks, tab in seqs:
                tab.length += len(toks)
            logits = torch.zeros(len(seqs), 50257)
            logits[:, EOS] = 1.0
            return logits
    cache = tm.new_cache(8)
    sched = Scheduler(Stub(), cache)
    r = Request([1, 2, 3], SamplingParams(max_tokens=10, temperature=0.0))
    r2 = Request([1, 2, 3], SamplingParams(max_tokens=3, temperature=0.0, stop_at_eos=False))
    sched.add(r); sched.add(r2)
    while sched.has_work:
        sched.step()
    assert r.finish_reason == "stop" and r.output_ids == [EOS]
    assert r2.finish_reason == "length" and r2.output_ids == [EOS] * 3
    assert cache.num_free == 8


def test_stops_at_context_limit(tm):
    hp = tm.hp
    prompt = [464] * (hp["n_ctx"] - 3)
    r = Request(prompt, greedy(50))
    run(tm, [r], num_blocks=64)
    assert r.finish_reason == "length" and len(r.output_ids) == 3


def test_rejects_bad_requests(tm):
    sched = Scheduler(tm, tm.new_cache(8))
    with pytest.raises(ValueError):
        sched.add(Request([], greedy(1)))
    with pytest.raises(ValueError):
        sched.add(Request([1] * 1024, greedy(1)))
    with pytest.raises(ValueError):
        sched.add(Request([1] * 200, greedy(1)))       # 8 blocks x 16 = 128 slots
    with pytest.raises(ValueError):
        sched.add(Request([1], greedy(0)))


def test_abort_frees_blocks(tm):
    sched = Scheduler(tm, tm.new_cache(16), max_batch=1)
    a, b = Request(encode(PROMPTS[0]), greedy(50)), Request(encode(PROMPTS[1]), greedy(50))
    sched.add(a); sched.add(b)
    sched.step()
    assert a.status == "running" and b.status == "waiting"
    sched.abort(a); sched.abort(b)
    assert a.finish_reason == b.finish_reason == "abort"
    assert not sched.has_work and sched.cache.num_free == 16


def test_engine_thread_streams_tokens(tm, reference):
    eng = LLMEngine(Scheduler(tm, tm.new_cache(64), max_batch=4)).start()
    try:
        done = threading.Event()
        streamed = {}
        left = [len(PROMPTS)]

        def fin(r):
            left[0] -= 1
            if left[0] == 0:
                done.set()
        reqs = []
        for p in PROMPTS:
            r = Request(encode(p), greedy(10), on_finish=fin)
            streamed[r.id] = []
            r.on_token = streamed[r.id].append
            reqs.append(eng.submit(r))
        assert done.wait(120)
        for r, p in zip(reqs, PROMPTS):
            assert streamed[r.id] == r.output_ids == reference[p][:10]
        m = eng.scheduler.metrics()
        assert m["finished"] == len(PROMPTS) and m["free_blocks"] == 64 and m["ttft_p50_s"] > 0
    finally:
        eng.stop()


def test_sampling_is_seeded(tm):
    def go():
        reqs = [Request(encode(p), SamplingParams(max_tokens=8, temperature=0.9, top_p=0.9))
                for p in PROMPTS[:3]]
        run(tm, reqs, max_batch=3, seed=123)
        return [r.output_ids for r in reqs]
    assert go() == go()
