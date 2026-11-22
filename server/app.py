"""OpenAI-compatible completions server on top of the continuous-batching engine.

    .venv\\Scripts\\python.exe -m server.app                       # http://127.0.0.1:8001
    .venv\\Scripts\\python.exe -m server.app --policy static       # static batching (for comparison)

Endpoints: POST /v1/completions (stream via SSE), GET /v1/models, /metrics, /health.
The engine runs in one background thread; each HTTP request registers callbacks that
push tokens into an asyncio.Queue on the server's event loop.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request as HTTPRequest
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.scheduler import LLMEngine, Request, SamplingParams  # noqa: E402
from engine.tokenizer import decode, encode  # noqa: E402

DEFAULT_PORT = 8001


class CompletionBody(BaseModel):
    model: str = "gpt2"
    prompt: str | list[str] | list[int]
    max_tokens: int = Field(16, ge=1)
    temperature: float = Field(1.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    stream: bool = False
    n: int = 1
    stop: str | list[str] | None = None
    ignore_eos: bool = False      # extension (as in vLLM): keep going past <|endoftext|>, for benchmarks


class TextStream:
    """Turns token ids into text deltas without splitting a multi-byte UTF-8 character
    (a GPT-2 token can be half of an emoji; decode() would give U+FFFD until the rest arrives)."""

    def __init__(self):
        self.ids: list[int] = []
        self.sent = 0

    def push(self, tok: int) -> str:
        self.ids.append(tok)
        text = decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta

    def flush(self) -> str:
        text = decode(self.ids)
        delta, self.sent = text[self.sent:], len(text)
        return delta


def create_app(engine: LLMEngine, model_name: str = "gpt2") -> FastAPI:
    app = FastAPI(title="llm-inference-engine")
    sched = engine.scheduler
    n_ctx = sched.n_ctx
    started = time.time()
    http_stats = {"completions": 0, "rejected": 0}

    def tokenize(body: CompletionBody) -> list[int]:
        if body.model != model_name:
            raise HTTPException(404, f"model {body.model!r} not found; this server serves {model_name!r}")
        if body.n != 1:
            raise HTTPException(400, "only n=1 is supported")
        if body.stop is not None:
            raise HTTPException(400, "stop sequences are not supported")
        p = body.prompt
        if isinstance(p, list) and p and isinstance(p[0], int):
            ids = list(p)
        else:
            if isinstance(p, list):
                if len(p) != 1:
                    raise HTTPException(400, "batched prompts are not supported; send one per request")
                p = p[0]
            ids = encode(p)
        if not ids:
            raise HTTPException(400, "empty prompt")
        if len(ids) >= n_ctx:
            raise HTTPException(400, f"prompt is {len(ids)} tokens; context is {n_ctx}")
        if sched.cache.blocks_for(len(ids) + 1) > sched.cache.num_blocks:
            raise HTTPException(400, "prompt does not fit in the KV cache")
        return ids

    def submit(ids, body, loop, queue) -> Request:
        req = Request(ids, SamplingParams(max_tokens=body.max_tokens, temperature=body.temperature,
                                          top_p=body.top_p, stop_at_eos=not body.ignore_eos))
        req.on_token = lambda t: loop.call_soon_threadsafe(queue.put_nowait, ("tok", t))
        req.on_finish = lambda r: loop.call_soon_threadsafe(queue.put_nowait, ("end", r.finish_reason))
        return engine.submit(req)

    def chunk(cid, created, text, finish=None):
        return {"id": cid, "object": "text_completion", "created": created, "model": model_name,
                "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish}]}

    @app.get("/health")
    def health():
        if engine.error is not None:
            return JSONResponse({"status": "error", "detail": repr(engine.error)}, status_code=500)
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": model_name, "object": "model", "created": int(started), "owned_by": "local"}]}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        m = sched.metrics()
        lines = []

        def put(name, value, kind, help_):
            lines.extend([f"# HELP {name} {help_}", f"# TYPE {name} {kind}", f"{name} {value}"])
        put("llm_requests_total", m["requests"], "counter", "Requests accepted by the scheduler")
        put("llm_requests_finished_total", m["finished"], "counter", "Requests that finished normally")
        put("llm_requests_aborted_total", m["aborted"], "counter", "Requests aborted (client went away)")
        put("llm_http_rejected_total", http_stats["rejected"], "counter", "Requests rejected with 4xx")
        put("llm_prompt_tokens_total", m["prompt_tokens"], "counter", "Prompt tokens received")
        put("llm_generated_tokens_total", m["generated_tokens"], "counter", "Tokens generated")
        put("llm_preemptions_total", m["preemptions"], "counter", "Sequences preempted for KV blocks")
        put("llm_decode_steps_total", m["decode_steps"], "counter", "Batched decode steps")
        put("llm_running", m["running"], "gauge", "Sequences currently decoding")
        put("llm_waiting", m["waiting"], "gauge", "Sequences waiting for admission")
        put("llm_kv_blocks_free", m["free_blocks"], "gauge", "Free KV cache blocks")
        put("llm_kv_blocks_total", m["total_blocks"], "gauge", "KV cache blocks")
        put("llm_avg_decode_batch", round(m["avg_decode_batch"], 3), "gauge", "Mean sequences per decode step")
        put("llm_ttft_p50_seconds", round(m["ttft_p50_s"], 6), "gauge", "Median time to first token")
        return "\n".join(lines) + "\n"

    @app.post("/v1/completions")
    async def completions(body: CompletionBody, http: HTTPRequest):
        try:
            ids = tokenize(body)
        except HTTPException:
            http_stats["rejected"] += 1
            raise
        http_stats["completions"] += 1
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        req = submit(ids, body, loop, queue)
        cid, created = f"cmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        if not body.stream:
            try:
                while True:
                    kind, val = await queue.get()
                    if kind == "end":
                        break
            except asyncio.CancelledError:
                engine.abort(req)
                raise
            if val == "error":
                raise HTTPException(500, "engine error")
            out = chunk(cid, created, decode(req.output_ids), val)
            n_out = len(req.output_ids)
            out["usage"] = {"prompt_tokens": len(ids), "completion_tokens": n_out,
                            "total_tokens": len(ids) + n_out}
            return out

        async def events():
            ts, finished = TextStream(), False
            try:
                while True:
                    kind, val = await queue.get()
                    if kind == "tok":
                        delta = ts.push(val)
                        if delta:
                            yield f"data: {json.dumps(chunk(cid, created, delta))}\n\n"
                        continue
                    finished = True
                    if val == "error":
                        yield f"data: {json.dumps({'error': {'message': 'engine error'}})}\n\n"
                    else:
                        yield f"data: {json.dumps(chunk(cid, created, ts.flush(), val))}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            finally:
                if not finished:          # client disconnected mid-stream
                    engine.abort(req)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app


def build_engine(model: str = "gpt2", num_blocks: int = 512, block_size: int = 16,
                 max_batch: int = 16, policy: str = "continuous", device: str | None = None,
                 seed: int = 0) -> LLMEngine:
    from engine.scheduler import Scheduler
    from engine.torch_model import TorchGPT2
    from engine.weights import load_gpt2
    hp, params = load_gpt2(model)
    tm = TorchGPT2(params, hp, device=device)
    sched = Scheduler(tm, tm.new_cache(num_blocks, block_size), max_batch=max_batch,
                      policy=policy, seed=seed)
    return LLMEngine(sched)


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--num-blocks", type=int, default=512, help="KV blocks (16 tokens each by default)")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--policy", choices=["continuous", "static"], default="continuous")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    engine = build_engine(args.model, args.num_blocks, args.block_size, args.max_batch,
                          args.policy, args.device, args.seed).start()
    s = engine.scheduler
    print(f"model {args.model} on {s.model.device} ({s.model.dtype}), policy {args.policy}, "
          f"max_batch {args.max_batch}, KV cache {s.cache.num_blocks} x {s.cache.block_size} tokens "
          f"= {s.cache.nbytes / 2**20:.0f} MB", flush=True)
    try:
        uvicorn.run(create_app(engine, args.model), host=args.host, port=args.port, log_level="warning")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
