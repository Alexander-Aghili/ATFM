"""ATFM harness proxy: an OpenAI-compatible endpoint in front of a Dynamo frontend.

Classifies each call, computes the index and tier, holds it in the global window or per
controller directive, injects nvext.agent_hints, forwards the body otherwise unchanged, and
emits llm.* events. Fail-open everywhere (D10).
"""
from __future__ import annotations

from atfm.control.directives import HoldBatch, HoldUpdate

from collections import OrderedDict

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from atfm.proxy.prediction import PredictionRunner

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from atfm.bus import InMemoryBus, JsonlBus
from atfm.schema.events import LlmDone, LlmFirstToken, LlmRequest, SessionStart

from .config import ProxyConfig
from .index import CallMeta, compute_index, estimate_isl, promote_at, service_time, tier
from .queue import Entry, HoldQueue


def _meta_from(req: Request, body: dict, cfg: ProxyConfig, now: float, turn_index: int) -> CallMeta:
    h = req.headers
    sid = h.get("x-atfm-session") or f"anon-{uuid.uuid4().hex[:8]}"
    cls = h.get("x-atfm-class", "background")
    cls = cls if cls in ("interactive", "background") else "background"
    try:
        deadline = float(h["x-atfm-deadline"]) if h.get("x-atfm-deadline") else None
    except ValueError:  # fail-open: a malformed deadline is no deadline
        deadline = None
    try:
        osl = int(body.get("max_tokens") or cfg.default_osl)
    except (TypeError, ValueError):
        osl = cfg.default_osl
    return CallMeta(session_id=sid, cls=cls, tenant=h.get("x-atfm-tenant", "t0"),
                    deadline=deadline, parent=h.get("x-atfm-parent"), turn_index=turn_index,
                    isl=estimate_isl(body), predicted_osl=osl, t_arrival=now)


def create_app(cfg: ProxyConfig, *, upstream_client: httpx.AsyncClient | None = None, bus=None, predictor=None,
               clock=time.time) -> FastAPI:
    app = FastAPI()
    runtime = ProxyRuntime(app, cfg=cfg, clock=clock, predictor=predictor, upstream_client=upstream_client, bus=bus)
    runtime.register(app)
    app.router.lifespan_context = runtime.lifespan
    return app


class ProxyRuntime:
    def __init__(self, app, cfg, clock, predictor, upstream_client=None, bus=None):
        self.cfg, self.clock, self.st = cfg, clock, app.state
        st = self.st
        self.owns_upstream = upstream_client is None
        st.cfg = cfg
        st.bus = bus if bus is not None else (JsonlBus(cfg.events_path) if cfg.events_path else InMemoryBus())
        st.client = upstream_client or httpx.AsyncClient(base_url=cfg.upstream_url, timeout=httpx.Timeout(600.0))
        st.queue = HoldQueue(cfg.window, clock=clock, max_hold_s=cfg.max_hold_s, max_size=cfg.max_queue_size)
        self._initialize_sessions()
        self._initialize_predictor(predictor)

    def _initialize_sessions(self):
        st = self.st
        st.last_body = OrderedDict()
        st.touches, st.touch_tokens, st.touch_failures = 0, 0, 0
        st.last_index, st.turns, st.known = {}, {}, set()
        st.trace = open(self.cfg.trace_path, "a") if self.cfg.trace_path else None

    def _initialize_predictor(self, predictor):
        self.owns_predictor = predictor is None and self.cfg.board_url is not None
        if self.owns_predictor:
            from atfm.proxy.board_client import BoardClient
            predictor = BoardClient(self.cfg.board_url, timeout_s=self.cfg.board_timeout_s)
        self.st.predictor = predictor
        self.st.prediction_runner = PredictionRunner(predictor, self.cfg.prediction_limit, self.cfg.board_timeout_s)
        self.st.pool = self.st.prediction_runner.pool
        self.st.predictions = self.st.prediction_runner.counts

    @asynccontextmanager
    async def lifespan(self, app):
        try:
            yield
        finally:
            await asyncio.to_thread(self.st.prediction_runner.close)
            if self.owns_predictor:
                self.st.predictor.client.close()
            if self.owns_upstream:
                await self.st.client.aclose()
            if self.st.trace is not None:
                self.st.trace.close()
            if self.st.queue._timer is not None:
                self.st.queue._timer.cancel()

    def register(self, app):
        app.get('/healthz')(self.healthz)
        app.get('/state')(self.state)
        app.post('/directives')(self.directives)
        app.post('/directives/batch')(self.directive_batch)
        app.get('/session/{session_id}/prompt')(self.session_prompt)
        app.post('/touch')(self.touch)
        app.post('/gate')(self.gate)
        app.post('/v1/chat/completions')(self.chat)

    async def predict(self, meta: CallMeta) -> tuple[float, float]:
        return await self.st.prediction_runner.predict(meta, (service_time(meta, self.cfg), 0.0))

    def emit(self, e) -> None:
        try:
            self.st.bus.publish(e)
        except Exception:
            pass

    def finish(self, entry: Entry, meta: CallMeta, rid: str, t_arr: float, t_rel: float, t_first,
               t_last: float, osl: int, status: int, first_emitted: bool = False) -> None:
        if entry.done:
            return
        self.st.queue.complete(entry)
        if t_first is not None and (not first_emitted):
            self.emit(LlmFirstToken(t=t_first, session_id=meta.session_id, request_id=rid))
        self.emit(LlmDone(t=t_last, session_id=meta.session_id, request_id=rid, osl=osl, status=status))
        if self.st.trace is not None:
            row = {'session_id': meta.session_id, 'parent_session_id': meta.parent, 'class': meta.cls,
                   'tenant': meta.tenant, 'turn_index': meta.turn_index, 't_request': t_arr,
                   't_release': t_rel, 't_first_token': t_first, 't_last_token': t_last,
                   't_enqueued': entry.t_enqueued, 't_admitted': entry.t_release, 'prediction_s': entry.prediction_s,
                   'isl': meta.isl, 'osl': osl, 'status': status}
            self.st.trace.write(json.dumps(row) + '\n')
            self.st.trace.flush()

    async def healthz(self):
        return {'ok': True}

    async def state(self):
        return {**self.st.queue.stats(), 'predictions': self.st.prediction_runner.snapshot(), 'touches': self.st.touches,
                'touch_tokens': self.st.touch_tokens, 'touch_failures': self.st.touch_failures}

    def apply_holds(self, holds: list[HoldUpdate]) -> dict:
        now = self.clock()
        applied = 0
        for directive in holds:
            if directive.expires_at is not None and directive.expires_at <= now:
                continue
            self.st.queue.set_directive(directive.session_id, directive.release_not_before, directive.reason, expires_at=directive.expires_at)
            applied += 1
        if applied:
            self.st.queue.tick()
        return {'ok': True, 'applied': applied, 'expired': len(holds) - applied}

    async def directives(self, directive: HoldUpdate):
        self.apply_holds([directive])
        return {'ok': True}

    async def directive_batch(self, batch: HoldBatch):
        return self.apply_holds(batch.holds)

    async def session_prompt(self, session_id: str):
        """The session's last prompt (model and messages), for a controller that needs its token ids."""
        last = self.st.last_body.get(session_id)
        if last is None:
            return JSONResponse({'error': 'unknown session'}, status_code=404)
        return {'session_id': session_id, 'model': last.get('model'), 'messages': last.get('messages', [])}

    async def touch(self, req: Request):
        """Refresh a remembered prefix at the lowest priority; failures are counted."""
        data = await req.json()
        last = self.st.last_body.get(data.get('session_id', ''))
        if last is None:
            return {'ok': False, 'prompt_tokens': 0}
        body = {k: v for k, v in last.items() if k in ('model', 'messages')}
        body.update(max_tokens=1, stream=False)
        body['nvext'] = {'agent_hints': {'priority': 0, 'strict_priority': 0, 'osl': 1}}
        headers = {'content-type': 'application/json', 'x-dynamo-session-id': data['session_id'], 'x-atfm-touch': '1'}
        return await self._send_touch(body, headers)

    async def _send_touch(self, body, headers):
        try:
            response = await self.st.client.post('/v1/chat/completions', json=body, headers=headers)
            tokens = int(response.json().get('usage', {}).get('prompt_tokens', 0)) if response.status_code == 200 else 0
        except Exception:
            return self._touch_failed()
        if response.status_code != 200:
            return self._touch_failed()
        self.st.touches += 1
        self.st.touch_tokens += tokens
        return {'ok': True, 'prompt_tokens': tokens}

    def _touch_failed(self):
        self.st.touch_failures += 1
        return {'ok': False, 'prompt_tokens': 0}

    async def gate(self, req: Request):
        d = await req.json()
        return {'allowed_at': self.st.queue.directive_for(d['session_id'])}

    async def chat(self, req: Request):
        body = await req.json()
        now = self.clock()
        meta = self._remember_call(req, body, now)
        entry = await self._enqueue(meta, now)
        rid = uuid.uuid4().hex[:16]
        try:
            return await self._forward(entry, meta, body, req, now, rid)
        except BaseException:
            # Cancellation before release must remove the waiter; afterward it frees a slot.
            if not entry.released.is_set():
                self.st.queue.cancel(entry)
            else:
                self.finish(entry, meta, rid, now, entry.t_release or now, None, self.clock(), 0, 502)
            raise

    def _remember_call(self, req, body, now):
        st = self.st
        turn = st.turns.get(req.headers.get('x-atfm-session', ''), 0)
        meta = _meta_from(req, body, self.cfg, now, turn)
        st.turns[meta.session_id] = turn + 1
        st.last_body[meta.session_id] = body
        st.last_body.move_to_end(meta.session_id)
        while len(st.last_body) > self.cfg.max_remembered_sessions:
            st.last_body.popitem(last=False)
        if meta.session_id not in st.known:
            st.known.add(meta.session_id)
            self.emit(SessionStart(t=now, session_id=meta.session_id, tenant=meta.tenant, cls=meta.cls,
                                  parent_session_id=meta.parent, deadline=meta.deadline))
        return meta

    async def _enqueue(self, meta, now):
        started = time.perf_counter()
        e_service, e_tool = await self.predict(meta)
        prediction_s = time.perf_counter() - started
        idx = compute_index(meta, self.cfg, e_service, e_tool)
        self.st.last_index[meta.session_id] = idx
        entry = Entry(session_id=meta.session_id, tier=tier(meta, self.cfg, now, e_service), index=idx,
                      t_arrival=now, promote_at=promote_at(meta, self.cfg, e_service),
                      t_enqueued=self.clock(), prediction_s=prediction_s)
        self.st.queue.submit(entry)
        return entry

    async def _forward(self, entry: Entry, meta: CallMeta, body: dict, req: Request, now: float, rid: str):
        await entry.released.wait()
        t_rel = self.clock()
        body = self._prepare_body(entry, meta, body, now, rid, t_rel)
        headers = self._forward_headers(req, meta.session_id)
        call = (entry, meta, rid, now, t_rel)
        try:
            if not body.get('stream'):
                return await self._forward_response(call, body, headers)
            return await self._forward_stream(call, body, headers)
        except Exception as exc:
            self.finish(*call, None, self.clock(), 0, 502)
            return JSONResponse({'error': f'upstream unavailable: {exc}'}, status_code=502)

    def _prepare_body(self, entry, meta, body, now, rid, t_rel):
        bucket = self.st.queue.priority_bucket(entry.tier, entry.index)
        hints = {'priority': bucket, 'strict_priority': entry.tier, 'osl': meta.predicted_osl}
        body = dict(body)
        body['nvext'] = dict(body.get('nvext') or {})
        body['nvext']['agent_hints'] = hints
        self.emit(LlmRequest(t=t_rel, session_id=meta.session_id, turn_index=meta.turn_index, request_id=rid,
                             isl=meta.isl, predicted_osl=meta.predicted_osl, hints=hints, held_s=t_rel - now))
        return body

    def _forward_headers(self, req, session_id):
        headers = {'content-type': 'application/json', 'x-dynamo-session-id': session_id}
        headers.update({k: v for k, v in req.headers.items() if k.lower().startswith('x-atfm-')})
        if req.headers.get('authorization'):
            headers['authorization'] = req.headers['authorization']
        return headers

    async def _forward_response(self, call, body, headers):
        response = await self.st.client.post('/v1/chat/completions', json=body, headers=headers)
        t_first = self.clock()
        osl = 0
        try:
            osl = int(response.json().get('usage', {}).get('completion_tokens', 0))
        except Exception:
            pass
        self.finish(*call, t_first, self.clock(), osl, response.status_code)
        return Response(content=response.content, status_code=response.status_code,
                        media_type=response.headers.get('content-type', 'application/json'))

    async def _forward_stream(self, call, body, headers):
        request = self.st.client.build_request('POST', '/v1/chat/completions', json=body, headers=headers)
        upstream = await self.st.client.send(request, stream=True)
        return StreamingResponse(self._stream_chunks(call, upstream), status_code=upstream.status_code,
                                 media_type=upstream.headers.get('content-type', 'text/event-stream'))

    async def _stream_chunks(self, call, upstream):
        _, meta, rid, _, _ = call
        t_first, osl = None, 0
        try:
            async for chunk in upstream.aiter_bytes():
                if t_first is None:
                    t_first = self.clock()
                    self.emit(LlmFirstToken(t=t_first, session_id=meta.session_id, request_id=rid))
                osl += chunk.count(b'data:')
                yield chunk
        finally:
            await upstream.aclose()
            self.finish(*call, t_first, self.clock(), osl, upstream.status_code, first_emitted=True)
