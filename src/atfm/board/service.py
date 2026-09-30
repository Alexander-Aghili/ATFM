"""Board HTTP service: the proxy's per-request predictions (under a time budget, fail-open), forecast
snapshots for operators, and the controllers' directives (holds, touches, tier log, replica floor)."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from atfm.board.execution import ControlWorker
from atfm.board.publication import Publication, snapshot_json
from atfm.board.serving import BoardReader
from atfm.board.live import LiveBoard, SessionRegistry
from atfm.board.metrics import worker_metrics_from_prometheus
from atfm.board.resumption import resumption_quantiles
from atfm.bus import InMemoryBus
from atfm.control import Deferrable, GdpPlanner, PrefetchPlanner, ReplicaFloor, Residency, TierLogger, TouchController


def residency_from_registry(registry: SessionRegistry, now: float, block_size: int = 16, default_worker: str = "w0") -> dict[str, Residency]:
    """Deploy-side view of who holds KV: every session not currently in a call, with its context in
    blocks and the time of its last completed call, on the worker that served it."""
    out = {}
    for s in registry.states(now):
        if s.phase == "llm_running" or s.t_last_done is None:
            continue
        out[s.session_id] = Residency(worker_id=s.worker_id or default_worker, blocks=s.ctx_tokens // block_size + 1, last_used=s.t_last_done)
    return out


def configure_controllers(app, cfg: dict, fetch=None):
    """Attach configured controllers and return a fail-open metrics scraper."""
    st = app.state
    for key, controller in (('gdp', GdpPlanner), ('touch', TouchController),
                            ('tier', TierLogger), ('replica', ReplicaFloor), ('prefetch', PrefetchPlanner)):
        if key in cfg:
            setattr(st, key, controller(**cfg[key]))
    st.worker_metrics, st.metrics_cfg = {}, cfg.get('metrics', {})
    return MetricsScraper(st, cfg, fetch).scrape


class MetricsScraper:
    def __init__(self, state, cfg, fetch):
        self.st, self.cfg = state, cfg
        self.fetch = fetch if fetch is not None else self._fetch

    def _fetch(self):
        import httpx
        url = self.st.metrics_cfg.get('url')
        return httpx.get(url, timeout=2.0).text if url else ''

    def scrape(self):
        try:
            page = self.fetch()
        except Exception:
            return
        metrics = self.st.metrics_cfg
        events = worker_metrics_from_prometheus(
            page, t=time.time(), default_total_blocks=metrics.get('default_total_blocks'),
            default_worker_id=metrics.get('default_worker_id'))
        if events:
            future = self.st.control_worker.submit(self._update, events)
            if future is not None:
                future.result()

    def _update(self, events):
        st, metrics = self.st, self.st.metrics_cfg
        st.worker_metrics = {event.worker_id: event for event in events}
        free = float(sum(event.kv_blocks_total - event.kv_blocks_used for event in events))
        slot = st.gdp.slot_s if st.gdp is not None else 30.0
        tps = float(metrics.get('prefill_tps_per_worker', self.cfg.get('replica', {}).get('prefill_tps_per_replica', 0.0)))
        st.capacity = {'kv_blocks': free, 'prefill_tokens': tps * slot * len(events)}
        threshold = float(metrics.get('pressure_threshold', 0.9))
        st.frontier = {event.worker_id: 0.0 for event in events
                       if event.kv_blocks_total and event.kv_blocks_used / event.kv_blocks_total >= threshold}
        st.residency = residency_from_registry(st.board.registry, time.time(), int(metrics.get('block_size', 16)),
                                               default_worker=next(iter(st.worker_metrics)))


def create_board_app(board: LiveBoard, *, bus=None, clock=time.time, rng=None, budget_s: float = 0.05,
                     gdp: GdpPlanner | None = None, capacity: dict[str, float] | None = None,
                     touch: TouchController | None = None, tier: TierLogger | None = None,
                     replica: ReplicaFloor | None = None, prediction_max_age_s: float | None = None) -> FastAPI:
    app = FastAPI()
    max_age = prediction_max_age_s if prediction_max_age_s is not None else max(1., 3 * getattr(board, "tick_s", 5.))
    runtime = BoardRuntime(app, board=board, budget_s=budget_s, clock=clock, bus=bus, rng=rng, max_age_s=max_age,
                           gdp=gdp, capacity=capacity, touch=touch, tier=tier, replica=replica)
    runtime.register(app)
    return app


class BoardRuntime:
    def __init__(self, app, board, budget_s, clock, max_age_s, bus=None, rng=None, gdp=None, capacity=None, touch=None, tier=None, replica=None):
        self.board, self.clock, self.budget_s = board, clock, budget_s
        st = app.state
        st.board, st.bus, st.rng = board, bus if bus is not None else InMemoryBus(), rng if rng is not None else np.random.default_rng(0)
        st.control_worker = self.worker = ControlWorker()
        st.reader = self.reader = BoardReader(board, clock(), budget_s, max_age_s)
        st.snapshot = None
        st.directives_cache = None       # (snapshot object, response): computed once per snapshot, served to every poller
        st.gdp, st.capacity, st.touch, st.tier, st.replica = gdp, capacity or {}, touch, tier, replica
        st.prefetch = None
        st.residency, st.frontier = {}, {}          # fed by a metrics scraper when one is attached
        self.st = st
        self.directives_response = (None, None)

    def register(self, app):
        app.get('/healthz')(self.healthz)
        app.post('/tick')(self.tick)
        app.get('/snapshot')(self.snapshot)
        app.post('/predict')(self.predict)
        app.post('/directives')(self.directives)
        app.get('/state')(self.status)
        app.router.lifespan_context = self.lifespan

    def ingest(self) -> int:
        for e in self.st.bus.drain():
            self.board.registry.apply(e)
        return len(self.board.registry.states(self.clock()))

    async def healthz(self):
        return {'ok': True}

    @asynccontextmanager
    async def lifespan(self, app):
        try:
            yield
        finally:
            await asyncio.to_thread(self.worker.close)
            await asyncio.to_thread(self.reader.close)

    async def tick(self):
        return await self.worker.run(self._tick)

    def _tick(self):
        captured_at = self.reader.monotonic()
        with self.worker.timings.measure('ingest'):
            n = self.ingest()
        now = self.clock()
        with self.worker.timings.measure('projection'):
            predictions = self.reader.capture(self.board, now)
        with self.worker.timings.measure('forecast'):
            snapshot = self.board.step(now, self.st.rng)
        with self.worker.timings.measure('snapshot_serialize'):
            encoded = snapshot_json(snapshot)
        version = self.reader.published.read().version + 1
        self.reader.published.publish(Publication(version, captured_at, snapshot, predictions, encoded))
        self.st.snapshot = snapshot
        return {'sessions': n, 't': snapshot.t, 'version': version}

    async def snapshot(self):
        return self.reader.snapshot()

    async def predict(self, req: Request):
        return await self.reader.predict(await req.json())

    async def status(self):
        return dict(**self.reader.status(), control=self.worker.snapshot(), control_stages=self.worker.timings.snapshot())

    async def directives(self):
        return await self.worker.run(self._directives_measured)

    def _directives_measured(self):
        with self.worker.timings.measure('directives'):
            result = self._directives()
        if self.directives_response[0] is not result:
            with self.worker.timings.measure('directives_serialize'):
                self.directives_response = (result, JSONResponse(result))
        return self.directives_response[1]

    def _directives(self):
        """Controllers run once per snapshot (they spend touch credit and reset planner state); every poller
            of the same snapshot gets the cached answer. POST because it is not free of side effects."""
        now = self.clock()
        s = self.st.snapshot
        out = {'t': now, 'snapshot_t': None if s is None else s.t, 'holds': [], 'touches': [], 'tier': [], 'replica': None}
        if s is None:
            return out
        if self.st.directives_cache is not None and self.st.directives_cache[0] is s:
            return self.st.directives_cache[1]
        states = {state.session_id: state for state in self.board.registry.states(now)}
        resumptions = resumption_quantiles(self.board.forecaster.predictor, states.values(), now, 64, self.st.rng, skip_errors=True)
        self._plan_directives(now, s, states, resumptions, out)
        self.st.directives_cache = (s, out)
        return out

    def _plan_directives(self, now, s, states, resumptions, out):
        if self.st.gdp is not None and self.st.capacity:
            self._plan_holds(now, s, states, resumptions, out)
        if self.st.touch is not None:
            out['touches'] = [t.model_dump() for t in self.st.touch.plan(now, resumptions, self.st.residency, self.st.frontier)]
        if self.st.tier is not None:
            out['tier'] = [t.model_dump() for t in self.st.tier.plan(now, resumptions)]
        if self.st.prefetch is not None:
            out['tier'] += [t.model_dump() for t in self.st.prefetch.plan(now, resumptions, states)]
        if self.st.replica is not None:
            out['replica'] = self.st.replica.propose(now, s).model_dump()

    def _plan_holds(self, now, s, states, resumptions, out):
        deferrable = [Deferrable(session_id=sid, tenant=states[sid].tenant, eta_s=q[1],
                                     kv_blocks=int(np.ceil(states[sid].ctx_tokens / 16)), prefill_tokens=states[sid].ctx_tokens)
                      for sid, q in resumptions.items() if states[sid].cls == 'background']
        out['holds'] = [h.model_dump() for h in self.st.gdp.plan(now, s, self.st.capacity, deferrable)]
