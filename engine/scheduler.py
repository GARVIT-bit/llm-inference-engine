"""Continuous batching over the paged KV cache.

Every call to Scheduler.step():
  1. admit waiting requests while there is batch room and enough free blocks for their
     prompts, and prefill them (one forward over all newly admitted prompts); each gets
     its first token;
  2. run ONE batched decode step for every request that was already running: each feeds
     its newest token and samples the next;
  3. retire finished requests and return their blocks to the pool.

policy="continuous": admission happens every step, so a finished request's slot is
refilled right away. policy="static": a new batch is admitted only once the whole
previous batch has finished (classic static batching; finished requests drop out of
the compute, but their slots stay empty until the batch is done).

If a decode step needs a new block and none are free, the youngest running request is
preempted: its blocks are freed and it goes back to the front of the queue, to be
re-prefilled later from prompt + tokens generated so far (recompute preemption).
"""
from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import torch

from engine.kv_cache import BlockTable, PagedKVCache
from engine.sampler import sample
from engine.torch_model import TorchGPT2

EOS = 50256  # <|endoftext|>


@dataclass
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    stop_at_eos: bool = True


_ids = itertools.count()


@dataclass(eq=False)
class Request:
    prompt_ids: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    on_token: Callable[[int], None] | None = None          # called from the engine thread
    on_finish: Callable[["Request"], None] | None = None
    id: int = field(default_factory=lambda: next(_ids))
    output_ids: list[int] = field(default_factory=list)
    table: BlockTable = field(default_factory=BlockTable)
    status: str = "new"            # new | waiting | running | finished
    finish_reason: str | None = None   # "length" | "stop" | "abort" | "error"
    preemptions: int = 0
    t_arrival: float = 0.0
    t_first_token: float | None = None
    t_finish: float | None = None

    @property
    def context_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids


class Scheduler:
    def __init__(self, model: TorchGPT2, cache: PagedKVCache, max_batch: int = 16,
                 policy: str = "continuous", max_prefill_tokens: int = 4096, seed: int = 0):
        if policy not in ("continuous", "static"):
            raise ValueError(policy)
        self.model, self.cache = model, cache
        self.max_batch, self.policy = max_batch, policy
        self.max_prefill_tokens = max_prefill_tokens
        self.n_ctx = model.hp["n_ctx"]
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.generator = torch.Generator(device=model.device).manual_seed(seed)
        self.stats = {"requests": 0, "finished": 0, "aborted": 0, "preemptions": 0,
                      "prompt_tokens": 0, "generated_tokens": 0, "steps": 0,
                      "decode_steps": 0, "decode_batch_sum": 0, "prefill_steps": 0}
        self.ttfts: deque[float] = deque(maxlen=10_000)

    # ------------------------------------------------------------------ public

    def add(self, req: Request) -> None:
        n = len(req.prompt_ids)
        if n == 0:
            raise ValueError("empty prompt")
        if n >= self.n_ctx:
            raise ValueError(f"prompt has {n} tokens, model context is {self.n_ctx}")
        if self.cache.blocks_for(n + 1) > self.cache.num_blocks:
            raise ValueError("prompt does not fit in the KV cache")
        if req.params.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        req.status = "waiting"
        req.t_arrival = req.t_arrival or time.perf_counter()
        self.waiting.append(req)
        self.stats["requests"] += 1
        self.stats["prompt_tokens"] += n

    def abort(self, req: Request) -> None:
        if req.status == "waiting":
            self.waiting.remove(req)
        elif req.status == "running":
            self.running.remove(req)
            self.cache.free(req.table)
        else:
            return
        self.stats["aborted"] += 1
        self._finish(req, "abort")

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def step(self) -> None:
        if not self.has_work:
            return
        self.stats["steps"] += 1
        decoding = list(self.running)
        admitted = self._admit()
        if admitted:
            self.stats["prefill_steps"] += 1
            logits = self.model.forward_batch([(r.context_ids, r.table) for r in admitted], self.cache)
            self._emit(admitted, logits)
        decoding = self._reserve_decode_slots(decoding)
        if decoding:
            self.stats["decode_steps"] += 1
            self.stats["decode_batch_sum"] += len(decoding)
            logits = self.model.forward_batch([([r.output_ids[-1]], r.table) for r in decoding], self.cache)
            self._emit(decoding, logits)

    # ------------------------------------------------------------------ internals

    def _admit(self) -> list[Request]:
        if self.policy == "static" and self.running:
            return []
        admitted, budget = [], self.max_prefill_tokens
        while self.waiting and len(self.running) < self.max_batch:
            r = self.waiting[0]
            n = len(r.context_ids)
            if admitted and n > budget:
                break
            # keep one free block per running request as headroom for its next decode
            # step, unless nothing is running (then the request just has to fit)
            headroom = len(self.running) if self.running else 0
            if self.cache.blocks_needed(r.table, n) > self.cache.num_free - headroom:
                break
            self.waiting.popleft()
            self.cache.reserve(r.table, n)
            r.status = "running"
            self.running.append(r)
            admitted.append(r)
            budget -= n
        return admitted

    def _reserve_decode_slots(self, decoding: list[Request]) -> list[Request]:
        """Make room for one more token per decoding request, oldest first; preempt the
        youngest running request whenever the pool is out of blocks."""
        ready = []
        for r in decoding:
            if r.status != "running":
                continue                        # preempted earlier in this loop
            while self.cache.blocks_needed(r.table, r.table.length + 1) > self.cache.num_free:
                victim = self.running[-1]
                self._preempt(victim)
                if victim is r:
                    break
            else:
                self.cache.reserve(r.table, r.table.length + 1)
                ready.append(r)
        return ready

    def _preempt(self, r: Request) -> None:
        self.running.remove(r)
        self.cache.free(r.table)
        r.status = "waiting"
        r.preemptions += 1
        self.stats["preemptions"] += 1
        self.waiting.appendleft(r)

    def _emit(self, reqs: list[Request], logits: torch.Tensor) -> None:
        toks = sample(logits, [r.params.temperature for r in reqs],
                      [r.params.top_p for r in reqs], self.generator).tolist()
        now = time.perf_counter()
        for r, t in zip(reqs, toks):
            if r.status != "running":
                continue
            r.output_ids.append(t)
            self.stats["generated_tokens"] += 1
            if r.t_first_token is None:
                r.t_first_token = now
                self.ttfts.append(now - r.t_arrival)
            if r.on_token:
                r.on_token(t)
            if r.params.stop_at_eos and t == EOS:
                reason = "stop"
            elif len(r.output_ids) >= r.params.max_tokens:
                reason = "length"
            elif len(r.context_ids) >= self.n_ctx:   # no position left for the next token
                reason = "length"
            elif self.cache.blocks_for(len(r.context_ids) + 1) > self.cache.num_blocks:
                reason = "length"                     # would not fit even alone in the pool
            else:
                continue
            self.running.remove(r)
            self.cache.free(r.table)
            self.stats["finished"] += 1
            self._finish(r, reason)

    def _finish(self, r: Request, reason: str) -> None:
        r.status, r.finish_reason, r.t_finish = "finished", reason, time.perf_counter()
        if r.on_finish:
            r.on_finish(r)

    def metrics(self) -> dict:
        s = dict(self.stats)
        s.update(running=len(self.running), waiting=len(self.waiting),
                 free_blocks=self.cache.num_free, total_blocks=self.cache.num_blocks,
                 block_size=self.cache.block_size)
        s["avg_decode_batch"] = s["decode_batch_sum"] / s["decode_steps"] if s["decode_steps"] else 0.0
        t = sorted(self.ttfts)
        s["ttft_p50_s"] = t[len(t) // 2] if t else 0.0
        return s


class LLMEngine:
    """Runs a Scheduler in a background thread. submit()/abort() are thread-safe."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self._cv = threading.Condition()
        self._inbox: list[Request] = []
        self._aborts: list[Request] = []
        self._stop = False
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def submit(self, req: Request) -> Request:
        req.t_arrival = time.perf_counter()
        with self._cv:
            self._inbox.append(req)
            self._cv.notify()
        return req

    def abort(self, req: Request) -> None:
        with self._cv:
            self._aborts.append(req)
            self._cv.notify()

    def start(self) -> "LLMEngine":
        self._thread = threading.Thread(target=self._loop, name="llm-engine", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        if self._thread:
            self._thread.join()

    def run_until_done(self) -> None:
        """Synchronous driver (tests, in-process benchmarks): no thread needed."""
        self._drain()
        while self.scheduler.has_work:
            self.scheduler.step()

    def _drain(self) -> None:
        with self._cv:
            inbox, aborts = self._inbox, self._aborts
            self._inbox, self._aborts = [], []
        for r in inbox:
            try:
                self.scheduler.add(r)
            except ValueError:
                self.scheduler._finish(r, "error")
        for r in aborts:
            self.scheduler.abort(r)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not (self._stop or self._inbox or self._aborts or self.scheduler.has_work):
                    self._cv.wait()
                if self._stop:
                    return
            self._drain()
            try:
                self.scheduler.step()
            except BaseException as e:   # never leave clients hanging
                self.error = e
                for r in list(self.scheduler.running) + list(self.scheduler.waiting):
                    if r.table.blocks:
                        self.scheduler.cache.free(r.table)
                    self.scheduler._finish(r, "error")
                self.scheduler.running.clear()
                self.scheduler.waiting.clear()
