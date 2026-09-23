"""ATFM harness proxy: an OpenAI-compatible endpoint in front of a Dynamo frontend.

Classifies each call, computes the index and tier, holds it in the global window or per
controller directive, injects nvext.agent_hints, forwards the body otherwise unchanged, and
emits llm.* events. Fail-open everywhere (D10).
"""
from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from atfm.bus import InMemoryBus, JsonlBus
from atfm.schema.events import LlmDone, LlmFirstToken, LlmRequest, SessionStart

from .config import ProxyConfig
from .index import CallMeta, compute_index, estimate_isl, priority_bucket, service_time, tier
from .queue import Entry, HoldQueue


def _meta_from(req: Request, body: dict, cfg: ProxyConfig, now: float, turn_index: int) -> CallMeta:
    h = req.headers
    sid = h.get("x-atfm-session") or f"anon-{uuid.uuid4().hex[:8]}"
    cls = h.get("x-atfm-class", "background")
    cls = cls if cls in ("interactive", "background") else "background"
    dl = h.get("x-atfm-deadline")
    osl = int(body.get("max_tokens") or cfg.default_osl)
    return CallMeta(session_id=sid, cls=cls, tenant=h.get("x-atfm-tenant", "t0"),
                    deadline=float(dl) if dl else None, parent=h.get("x-atfm-parent"), turn_index=turn_index,
                    isl=estimate_isl(body), predicted_osl=osl, t_arrival=now)


def create_app(cfg: ProxyConfig, *, upstream_client: httpx.AsyncClient | None = None, bus=None, predictor=None,
               clock=time.time) -> FastAPI:
    app = FastAPI()
    st = app.state
    st.cfg = cfg
    st.bus = bus if bus is not None else (JsonlBus(cfg.events_path) if cfg.events_path else InMemoryBus())
    st.client = upstream_client or httpx.AsyncClient(base_url=cfg.upstream_url, timeout=httpx.Timeout(600.0))
    st.queue = HoldQueue(cfg.window, clock=clock, max_hold_s=cfg.max_hold_s)
    st.predictor = predictor
    st.pool = ThreadPoolExecutor(max_workers=4)
    st.turns = {}
    st.known = set()
    st.trace = open(cfg.trace_path, "a") if cfg.trace_path else None

    def predict(meta: CallMeta) -> tuple[float, float]:
        e_service = service_time(meta, cfg)
        if st.predictor is None:
            return e_service, 0.0

        def _call():
            return (float(st.predictor.expected_service(meta.session_id, meta.isl, meta.predicted_osl)),
                    float(st.predictor.expected_tool_next(meta.session_id)))

        fut = st.pool.submit(_call)
        try:
            return fut.result(timeout=cfg.board_timeout_s)
        except Exception:
            return e_service, 0.0

    def emit(e) -> None:
        try:
            st.bus.publish(e)
        except Exception:
            pass

    def finish(meta: CallMeta, rid: str, t_arr: float, t_rel: float, t_first, t_last: float, osl: int,
               status: int, first_emitted: bool = False) -> None:
        st.queue.complete()
        if t_first is not None and not first_emitted:
            emit(LlmFirstToken(t=t_first, session_id=meta.session_id, request_id=rid))
        emit(LlmDone(t=t_last, session_id=meta.session_id, request_id=rid, osl=osl, status=status))
        if st.trace is not None:
            row = {"session_id": meta.session_id, "parent_session_id": meta.parent, "class": meta.cls,
                   "tenant": meta.tenant, "turn_index": meta.turn_index, "t_request": t_arr, "t_release": t_rel,
                   "t_first_token": t_first, "t_last_token": t_last, "isl": meta.isl, "osl": osl, "status": status}
            st.trace.write(json.dumps(row) + "\n")
            st.trace.flush()

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/state")
    async def state():
        return st.queue.stats()

    @app.post("/directives")
    async def directives(req: Request):
        d = await req.json()
        st.queue.set_directive(d["session_id"], float(d["release_not_before"]), d.get("reason", ""))
        st.queue.tick()
        return {"ok": True}

    @app.post("/gate")
    async def gate(req: Request):
        d = await req.json()
        return {"allowed_at": st.queue.directive_for(d["session_id"])}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        now = clock()
        turn = st.turns.get(req.headers.get("x-atfm-session", ""), 0)
        meta = _meta_from(req, body, cfg, now, turn)
        st.turns[meta.session_id] = turn + 1
        if meta.session_id not in st.known:
            st.known.add(meta.session_id)
            emit(SessionStart(t=now, session_id=meta.session_id, tenant=meta.tenant, cls=meta.cls,
                              parent_session_id=meta.parent, deadline=meta.deadline))
        e_service, e_tool = predict(meta)
        idx = compute_index(meta, cfg, e_service, e_tool)
        tr = tier(meta, cfg, now, e_service)
        entry = Entry(session_id=meta.session_id, tier=tr, index=idx, t_arrival=now)
        st.queue.submit(entry)
        await entry.released.wait()
        t_rel = clock()
        bucket = priority_bucket(idx, st.queue.tier_indices(tr) + [idx])
        hints = {"priority": bucket, "strict_priority": tr, "osl": meta.predicted_osl}
        body = dict(body)
        body["nvext"] = dict(body.get("nvext") or {})
        body["nvext"]["agent_hints"] = hints
        rid = uuid.uuid4().hex[:16]
        emit(LlmRequest(t=t_rel, session_id=meta.session_id, turn_index=meta.turn_index, request_id=rid,
                        isl=meta.isl, predicted_osl=meta.predicted_osl, hints=hints, held_s=t_rel - now))
        headers = {"content-type": "application/json", "x-dynamo-session-id": meta.session_id}
        headers.update({k: v for k, v in req.headers.items() if k.lower().startswith("x-atfm-")})
        stream = bool(body.get("stream"))
        t_first = None
        try:
            if not stream:
                r = await st.client.post("/v1/chat/completions", json=body, headers=headers)
                t_first = clock()
                osl = 0
                try:
                    osl = int(r.json().get("usage", {}).get("completion_tokens", 0))
                except Exception:
                    pass
                finish(meta, rid, now, t_rel, t_first, clock(), osl, r.status_code)
                return Response(content=r.content, status_code=r.status_code,
                                media_type=r.headers.get("content-type", "application/json"))
            upstream = await st.client.send(
                st.client.build_request("POST", "/v1/chat/completions", json=body, headers=headers), stream=True)

            async def gen():
                nonlocal t_first
                osl = 0
                try:
                    async for chunk in upstream.aiter_bytes():
                        if t_first is None:
                            t_first = clock()
                            emit(LlmFirstToken(t=t_first, session_id=meta.session_id, request_id=rid))
                        osl += chunk.count(b"data:")
                        yield chunk
                finally:
                    await upstream.aclose()
                    finish(meta, rid, now, t_rel, t_first, clock(), osl, upstream.status_code, first_emitted=True)

            return StreamingResponse(gen(), status_code=upstream.status_code,
                                     media_type=upstream.headers.get("content-type", "text/event-stream"))
        except httpx.HTTPError as e:
            finish(meta, rid, now, t_rel, None, clock(), 0, 502)
            return JSONResponse({"error": f"upstream unavailable: {e}"}, status_code=502)

    return app
