"""Board HTTP service: the proxy's per-request predictions (under a time budget, fail-open), forecast
snapshots for operators, and the controllers' directives (holds, touches, tier log, replica floor)."""
from __future__ import annotations

import asyncio
import time

import numpy as np
from fastapi import FastAPI, Request

from atfm.board.forecaster import CLASSES, TARGETS
from atfm.board.live import LiveBoard, SessionRegistry
from atfm.board.metrics import worker_metrics_from_prometheus
from atfm.bus import InMemoryBus
from atfm.control import Deferrable, GdpPlanner, ReplicaFloor, Residency, TierLogger, TouchController


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
    """Attach controllers from a config dict and return a scrape callable that turns a Prometheus page
    (from `fetch()`, default: HTTP GET of cfg["metrics"]["url"]) into capacity, worker metrics and the
    eviction frontier the touch controller needs. Under cache pressure (usage over the threshold) every idle
    context is at risk, so the frontier age is 0; otherwise nothing is."""
    st = app.state
    if "gdp" in cfg:
        st.gdp = GdpPlanner(**cfg["gdp"])
    if "touch" in cfg:
        st.touch = TouchController(**cfg["touch"])
    if "tier" in cfg:
        st.tier = TierLogger(**cfg["tier"])
    if "replica" in cfg:
        st.replica = ReplicaFloor(**cfg["replica"])
    m = cfg.get("metrics", {})
    st.worker_metrics = {}
    st.metrics_cfg = m
    if fetch is None:
        url = m.get("url")

        def fetch():
            import httpx
            return httpx.get(url, timeout=2.0).text if url else ""

    def scrape():
        try:
            page = fetch()
        except Exception:
            return
        evs = worker_metrics_from_prometheus(page, t=time.time(), default_total_blocks=m.get("default_total_blocks"),
                                             default_worker_id=m.get("default_worker_id"))
        if not evs:
            return
        st.worker_metrics = {e.worker_id: e for e in evs}
        free = float(sum(e.kv_blocks_total - e.kv_blocks_used for e in evs))
        slot = st.gdp.slot_s if st.gdp is not None else 30.0
        tps = float(m.get("prefill_tps_per_worker", cfg.get("replica", {}).get("prefill_tps_per_replica", 0.0)))
        st.capacity = {"kv_blocks": free, "prefill_tokens": tps * slot * len(evs)}
        thr = float(m.get("pressure_threshold", 0.9))
        st.frontier = {e.worker_id: 0.0 for e in evs if e.kv_blocks_total and e.kv_blocks_used / e.kv_blocks_total >= thr}
        st.residency = residency_from_registry(st.board.registry, time.time(), int(m.get("block_size", 16)),
                                               default_worker=next(iter(st.worker_metrics)))
    return scrape


def create_board_app(board: LiveBoard, *, bus=None, clock=time.time, rng=None, budget_s: float = 0.05,
                     gdp: GdpPlanner | None = None, capacity: dict[str, float] | None = None,
                     touch: TouchController | None = None, tier: TierLogger | None = None,
                     replica: ReplicaFloor | None = None) -> FastAPI:
    app = FastAPI()
    st = app.state
    st.board, st.bus, st.rng = board, bus if bus is not None else InMemoryBus(), rng if rng is not None else np.random.default_rng(0)
    st.snapshot = None
    st.directives_cache = None       # (snapshot object, response): computed once per snapshot, served to every poller
    st.gdp, st.capacity, st.touch, st.tier, st.replica = gdp, capacity or {}, touch, tier, replica
    st.residency, st.frontier = {}, {}          # fed by a metrics scraper when one is attached

    def ingest() -> int:
        for e in st.bus.drain():
            board.registry.apply(e)
        return len(board.registry.states(clock()))

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.post("/tick")
    async def tick():
        n = ingest()
        st.snapshot = board.step(clock(), st.rng)
        return {"sessions": n, "t": st.snapshot.t}

    @app.get("/snapshot")
    async def snapshot():
        s = st.snapshot
        if s is None:
            return {"t": None, "horizons": [], "q50": {}, "q90": {}}
        return {"t": s.t, "model_id": s.model_id, "horizons": list(s.horizons),
                "q50": {t: {c: s.quantiles(t, c, 0.5).tolist() for c in CLASSES} for t in TARGETS},
                "q90": {t: {c: s.quantiles(t, c, 0.9).tolist() for c in CLASSES} for t in TARGETS},
                "endogenous_fraction": {c: (s.endogenous_fraction[c].tolist() if c in s.endogenous_fraction else []) for c in CLASSES}}

    @app.post("/predict")
    async def predict(req: Request):
        d = await req.json()
        t0 = time.perf_counter()
        sid, isl, osl = d.get("session_id", ""), int(d.get("isl", 0)), int(d.get("osl", 0))

        def _compute():
            return float(board.expected_service(sid, isl, osl)), float(board.expected_tool_next(sid))
        try:                                     # the budget is enforced: past it the proxy's defaults are returned
            e_service, e_tool = await asyncio.wait_for(asyncio.to_thread(_compute), timeout=budget_s)
            over = False
        except (asyncio.TimeoutError, Exception):
            e_service, e_tool, over = 0.0, 0.0, True
        ms = (time.perf_counter() - t0) * 1000.0
        return {"e_service_s": e_service, "e_tool_next_s": e_tool, "elapsed_ms": ms, "over_budget": over or ms > budget_s * 1000.0}

    @app.post("/directives")
    async def directives():
        """Controllers run once per snapshot (they spend touch credit and reset planner state); every poller
        of the same snapshot gets the cached answer. POST because it is not free of side effects."""
        now = clock()
        s = st.snapshot
        out = {"t": now, "snapshot_t": None if s is None else s.t, "holds": [], "touches": [], "tier": [], "replica": None}
        if s is None:
            return out
        if st.directives_cache is not None and st.directives_cache[0] is s:
            return st.directives_cache[1]
        resumptions = {}
        pred = board.forecaster.predictor
        for state in board.registry.states(now):
            try:
                samples = pred.resumption(state, now, 64, st.rng)
                finite = samples[np.isfinite(samples)]
                if len(finite):
                    q = np.quantile(finite, [0.1, 0.5, 0.9])
                    resumptions[state.session_id] = (float(q[0]), float(q[1]), float(q[2]))
            except Exception:
                continue
        if st.gdp is not None and st.capacity:
            deferrable = [Deferrable(session_id=sid, tenant=board.registry._s[sid].tenant, eta_s=q[1],
                                     kv_blocks=int(np.ceil(board.registry._s[sid].ctx_tokens / 16)),
                                     prefill_tokens=int(board.registry._s[sid].ctx_tokens))
                          for sid, q in resumptions.items() if board.registry._s[sid].cls == "background"]
            out["holds"] = [h.model_dump() for h in st.gdp.plan(now, s, st.capacity, deferrable)]
        if st.touch is not None:
            out["touches"] = [t.model_dump() for t in st.touch.plan(now, resumptions, st.residency, st.frontier)]
        if st.tier is not None:
            out["tier"] = [t.model_dump() for t in st.tier.plan(now, resumptions)]
        if st.replica is not None:
            out["replica"] = st.replica.propose(now, s).model_dump()
        st.directives_cache = (s, out)
        return out

    return app
