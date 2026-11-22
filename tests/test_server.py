"""HTTP API: OpenAI-compatible responses, SSE streaming, and the official OpenAI client."""
import json
import socket
import threading
import time

import pytest
import torch
from fastapi.testclient import TestClient

from engine.numpy_model import generate as np_generate
from engine.scheduler import LLMEngine, Scheduler
from engine.tokenizer import decode, encode
from engine.torch_model import TorchGPT2
from server.app import TextStream, create_app

PROMPT = "Alan Turing theorized that computers would one day become"


@pytest.fixture(scope="module")
def engine(gpt2_small):
    hp, p = gpt2_small
    tm = TorchGPT2(p, hp, device="cpu", dtype=torch.float32)
    eng = LLMEngine(Scheduler(tm, tm.new_cache(128), max_batch=8)).start()
    yield eng
    eng.stop()


@pytest.fixture(scope="module")
def client(engine):
    return TestClient(create_app(engine))


@pytest.fixture(scope="module")
def expected(gpt2_small):
    hp, p = gpt2_small
    return decode(np_generate(encode(PROMPT), p, hp, 12, use_cache=True))


def test_health_and_models(client):
    assert client.get("/health").json() == {"status": "ok"}
    m = client.get("/v1/models").json()
    assert m["object"] == "list" and m["data"][0]["id"] == "gpt2"


def test_completion_greedy(client, expected):
    r = client.post("/v1/completions", json={"model": "gpt2", "prompt": PROMPT,
                                             "max_tokens": 12, "temperature": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion" and body["id"].startswith("cmpl-")
    assert body["choices"][0]["text"] == expected
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": len(encode(PROMPT)), "completion_tokens": 12,
                             "total_tokens": len(encode(PROMPT)) + 12}


def test_prompt_as_token_ids(client, expected):
    r = client.post("/v1/completions", json={"prompt": encode(PROMPT), "max_tokens": 12, "temperature": 0})
    assert r.json()["choices"][0]["text"] == expected


def test_streaming_sse_matches_non_streaming(client, expected):
    chunks = []
    with client.stream("POST", "/v1/completions", json={"prompt": PROMPT, "max_tokens": 12,
                                                         "temperature": 0, "stream": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                chunks.append(line[len("data: "):])
    assert chunks[-1] == "[DONE]"
    objs = [json.loads(c) for c in chunks[:-1]]
    assert len(objs) >= 2                                  # actually streamed, not one blob
    assert "".join(o["choices"][0]["text"] for o in objs) == expected
    assert objs[-1]["choices"][0]["finish_reason"] == "length"
    assert all(o["choices"][0]["finish_reason"] is None for o in objs[:-1])


def test_concurrent_requests_are_batched(client, engine, expected):
    before = engine.scheduler.metrics()
    results = [None] * 6

    def go(i):
        results[i] = client.post("/v1/completions", json={"prompt": PROMPT, "max_tokens": 12,
                                                          "temperature": 0}).json()
    ts = [threading.Thread(target=go, args=(i,)) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert all(r["choices"][0]["text"] == expected for r in results)
    after = engine.scheduler.metrics()
    steps = after["decode_steps"] - before["decode_steps"]
    batch = (after["decode_batch_sum"] - before["decode_batch_sum"]) / steps
    assert batch > 1.5, batch                              # sequences shared decode steps
    assert after["free_blocks"] == after["total_blocks"]


@pytest.mark.parametrize("payload, code", [
    ({"prompt": PROMPT, "temperature": -1}, 422),
    ({"prompt": PROMPT, "top_p": 0}, 422),
    ({"prompt": PROMPT, "max_tokens": 0}, 422),
    ({"prompt": ""}, 400),
    ({"prompt": "hi " * 1100}, 400),
    ({"prompt": PROMPT, "model": "gpt-4"}, 404),
    ({"prompt": PROMPT, "n": 2}, 400),
    ({"prompt": [PROMPT, PROMPT]}, 400),
])
def test_rejects_bad_requests(client, payload, code):
    assert client.post("/v1/completions", json=payload).status_code == code


def test_metrics_prometheus_format(client):
    text = client.get("/metrics").text
    for name in ("llm_requests_total", "llm_generated_tokens_total", "llm_kv_blocks_free",
                 "llm_running", "llm_waiting", "llm_ttft_p50_seconds"):
        assert f"# TYPE {name} " in text
    vals = dict(l.split(" ") for l in text.splitlines() if not l.startswith("#"))
    assert float(vals["llm_generated_tokens_total"]) > 0


def test_text_stream_holds_back_partial_utf8():
    ids = encode("hi 🙂 there")
    ts = TextStream()
    pieces = [ts.push(t) for t in ids]
    assert "�" not in "".join(pieces)
    assert "".join(pieces) + ts.flush() == "hi 🙂 there"


# ------------------------------------------------------------------ real server + OpenAI client

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(engine):
    import uvicorn
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(engine), host="127.0.0.1", port=port,
                                           log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    t.join(5)


def test_openai_client(live_server, expected):
    openai = pytest.importorskip("openai")
    c = openai.OpenAI(base_url=live_server, api_key="not-needed")
    assert [m.id for m in c.models.list()] == ["gpt2"]
    r = c.completions.create(model="gpt2", prompt=PROMPT, max_tokens=12, temperature=0)
    assert r.choices[0].text == expected and r.usage.completion_tokens == 12
    text = "".join(ch.choices[0].text for ch in
                   c.completions.create(model="gpt2", prompt=PROMPT, max_tokens=12,
                                        temperature=0, stream=True))
    assert text == expected


def test_client_disconnect_aborts_request(live_server, engine):
    import httpx
    with httpx.stream("POST", live_server + "/completions",
                      json={"prompt": PROMPT, "max_tokens": 500, "temperature": 0, "stream": True},
                      timeout=30) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                break                                    # read one chunk, then hang up
    for _ in range(200):
        m = engine.scheduler.metrics()
        if m["running"] == 0 and m["waiting"] == 0:
            break
        time.sleep(0.05)
    assert m["running"] == 0 and m["free_blocks"] == m["total_blocks"]
    assert m["aborted"] >= 1
