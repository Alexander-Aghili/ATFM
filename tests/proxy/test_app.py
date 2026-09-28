import asyncio, json, time
import httpx, pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from atfm.bus import InMemoryBus
from atfm.proxy.config import ProxyConfig
from atfm.proxy.app import create_app

def _upstream(delay_s=0.0, fail=False):
    up = FastAPI()
    up.state.seen = []
    @up.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        up.state.seen.append({"body": body, "headers": dict(req.headers)})
        await asyncio.sleep(delay_s)
        if fail:
            return JSONResponse({"error": "boom"}, status_code=500)
        return JSONResponse({"id": "x", "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 7}})
    return up

def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")

def _up_client(up):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=up), base_url="http://up")

def _cfg(**kw):
    return ProxyConfig(upstream_url="http://up", **kw)

def _headers(sid, cls="background", deadline=None):
    h = {"x-atfm-session": sid, "x-atfm-class": cls, "x-atfm-tenant": "t1"}
    if deadline is not None:
        h["x-atfm-deadline"] = str(deadline)
    return h

BODY = {"model": "m", "messages": [{"role": "user", "content": "hello " * 50}], "max_tokens": 32}

async def test_hints_injected_body_otherwise_unchanged():
    up = _upstream(); bus = InMemoryBus()
    app = create_app(_cfg(window=4), upstream_client=_up_client(up), bus=bus)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=BODY, headers=_headers("s1", "interactive"))
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "ok"
    sent = up.state.seen[0]
    assert sent["body"]["messages"] == BODY["messages"] and sent["body"]["max_tokens"] == 32
    assert sent["body"]["nvext"]["agent_hints"] == {"priority": 3, "strict_priority": 1, "osl": 32}
    assert sent["headers"]["x-dynamo-session-id"] == "s1"
    kinds = [e.kind for e in bus.drain()]
    assert kinds == ["session.start", "llm.request", "llm.first_token", "llm.done"]

async def test_window_orders_interactive_slack_first():
    up = _upstream(delay_s=0.3)
    app = create_app(_cfg(window=1, slack_threshold_s=5.0), upstream_client=_up_client(up))
    async with _client(app) as c:
        first = asyncio.create_task(c.post("/v1/chat/completions", json=BODY, headers=_headers("bg0")))
        await asyncio.sleep(0.05)
        bgs = [asyncio.create_task(c.post("/v1/chat/completions", json=BODY, headers=_headers(f"bg{i}"))) for i in (1, 2, 3)]
        await asyncio.sleep(0.05)
        it = asyncio.create_task(c.post("/v1/chat/completions", json=BODY, headers=_headers("int", "interactive", deadline=time.time() + 2)))
        await asyncio.gather(first, *bgs, it)
    order = [s["headers"]["x-atfm-session"] for s in up.state.seen]
    assert order[0] == "bg0" and order[1] == "int"
    assert up.state.seen[1]["body"]["nvext"]["agent_hints"]["strict_priority"] == 2

async def test_directive_hold_capped_and_upstream_error_passthrough():
    up = _upstream(fail=True); bus = InMemoryBus()
    app = create_app(_cfg(window=4, max_hold_s=0.2), upstream_client=_up_client(up), bus=bus)
    async with _client(app) as c:
        r = await c.post("/directives", json={"session_id": "h1", "release_not_before": time.time() + 60, "reason": "gdp"})
        assert r.status_code == 200
        t0 = time.time()
        r = await c.post("/v1/chat/completions", json=BODY, headers=_headers("h1"))
        assert 0.15 <= time.time() - t0 < 2.0
        assert r.status_code == 500
        st = (await c.get("/state")).json()
        assert st["caps"] == 1
    ev = bus.drain()
    req = [e for e in ev if e.kind == "llm.request"][0]
    assert req.held_s >= 0.15
    assert [e for e in ev if e.kind == "llm.done"][0].status == 500

async def test_gate_endpoint_reports_directive():
    app = create_app(_cfg(window=4))
    async with _client(app) as c:
        assert (await c.post("/gate", json={"session_id": "z", "kind": "tool"})).json()["allowed_at"] is None
        await c.post("/directives", json={"session_id": "z", "release_not_before": 5000.0, "reason": "gdp"})
        assert (await c.post("/gate", json={"session_id": "z", "kind": "tool"})).json()["allowed_at"] == 5000.0

async def test_cancel_and_upstream_exception_release_slots():
    class Boom:
        async def post(self, *a, **k):
            raise RuntimeError("boom")
    bus = InMemoryBus()
    app = create_app(_cfg(window=1, max_hold_s=5.0), upstream_client=Boom(), bus=bus)
    async with _client(app) as c:
        # upstream raising a non-HTTP exception must still free the slot
        r = await c.post("/v1/chat/completions", json=BODY, headers=_headers("x1"))
        assert r.status_code == 502
        st = (await c.get("/state")).json()
        assert st["in_flight"] == 0 and st["queued"] == 0
        await _cancel_held(c)
    assert all(e.status == 502 for e in bus.drain() if e.kind == "llm.done")


async def _cancel_held(c):
    # a request cancelled while held must leave nothing behind
    await c.post("/directives", json={"session_id": "held", "release_not_before": time.time() + 3, "reason": "gdp"})
    task = asyncio.create_task(c.post("/v1/chat/completions", json=BODY, headers=_headers("held")))
    await asyncio.sleep(0.1)
    assert (await c.get("/state")).json()["held"] == 1
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    st = (await c.get("/state")).json()
    assert st["queued"] == 0 and st["in_flight"] == 0

async def test_malformed_deadline_is_ignored():
    up = _upstream()
    app = create_app(_cfg(window=2), upstream_client=_up_client(up))
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=BODY, headers={**_headers("m", "interactive"), "x-atfm-deadline": "soon"})
    assert r.status_code == 200 and up.state.seen[0]["body"]["nvext"]["agent_hints"]["strict_priority"] == 1
