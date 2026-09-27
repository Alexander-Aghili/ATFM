"""Proxy hardening (spec 10): hold-queue overflow forwards FCFS with an alarm; directives expire; the proxy
can issue keep-alive touches upstream and account for them."""
import asyncio

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from atfm.proxy.app import create_app
from atfm.proxy.config import ProxyConfig
from atfm.proxy.queue import Entry, HoldQueue


def _entry(sid, t, tier=0, index=0.0):
    return Entry(session_id=sid, t_arrival=t, tier=tier, index=index)


def test_hold_queue_over_max_size_forwards_fcfs_and_counts_alarms():
    clock = {"t": 0.0}
    q = HoldQueue(window=1, clock=lambda: clock["t"], max_hold_s=600.0, max_size=2)
    first = _entry("a", 0.0)
    q.submit(first)                                   # takes the window
    q.submit(_entry("b", 1.0, tier=0, index=1.0))     # pending 1
    q.submit(_entry("c", 2.0, tier=2, index=9.0))     # pending 2 (would be first by tier)
    assert q.alarms == 0
    late = _entry("d", 3.0, tier=0, index=0.0)
    q.submit(late)                                    # pending 3 > max_size: overflow -> FCFS, alarm
    assert q.alarms == 1 and q.overflow is True
    assert first.released.is_set()
    q.complete(first)
    assert q.release_order == ["a", "b"]              # FCFS under overflow, not tier order


def test_expired_directive_is_ignored():
    clock = {"t": 100.0}
    q = HoldQueue(window=4, clock=lambda: clock["t"], max_hold_s=600.0)
    q.set_directive("s", release_not_before=150.0, reason="gdp", expires_at=120.0)
    e = _entry("s", 100.0)
    q.submit(e)
    assert e.not_before == 150.0
    clock["t"] = 130.0
    e2 = _entry("s", 130.0)
    q.submit(e2)
    assert e2.not_before == 0.0 and q.directive_for("s") is None        # expired: dropped on read


def _upstream():
    up = FastAPI()
    up.state.seen = []

    @up.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        up.state.seen.append(body)
        return JSONResponse({"id": "x", "choices": [{"message": {"role": "assistant", "content": "k"}}],
                             "usage": {"prompt_tokens": 500, "completion_tokens": 1}})
    return up


async def test_touch_endpoint_issues_a_minimal_prefix_request_and_accounts():
    up = _upstream()
    app = create_app(ProxyConfig(upstream_url="http://up"), upstream_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=up), base_url="http://up"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
        body = {"model": "m", "messages": [{"role": "user", "content": "prefix " * 100}]}
        r = await c.post("/v1/chat/completions", json=body, headers={"x-atfm-session": "s1", "x-atfm-class": "background", "x-atfm-tenant": "t"})
        assert r.status_code == 200
        r = await c.post("/touch", json={"session_id": "s1"})
        assert r.status_code == 200 and r.json() == {"ok": True, "prompt_tokens": 500}
        sent = up.state.seen[-1]
        assert sent["max_tokens"] == 1 and sent["messages"] == body["messages"] and sent["nvext"]["agent_hints"]["strict_priority"] == 0
        st = (await c.get("/state")).json()
        assert st["touches"] == 1 and st["touch_tokens"] == 500
        r = await c.post("/touch", json={"session_id": "never-seen"})
        assert r.status_code == 200 and r.json()["ok"] is False           # nothing to touch: fail-open
