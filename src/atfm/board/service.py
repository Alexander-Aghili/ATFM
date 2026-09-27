"""Board HTTP service: the proxy's per-request predictions (under a time budget, fail-open), forecast
snapshots for operators, and the controllers' directives (holds, touches, tier log, replica floor)."""
from __future__ import annotations

import time

import numpy as np
from fastapi import FastAPI, Request

from atfm.board.live import LiveBoard
from atfm.bus import InMemoryBus
from atfm.control import Deferrable, GdpPlanner, ReplicaFloor, TierLogger, TouchController
from atfm.board.forecaster import CLASSES, TARGETS


def create_board_app(board: LiveBoard, *, bus=None, clock=time.time, rng=None, budget_s: float = 0.05,
                     gdp: GdpPlanner | None = None, capacity: dict[str, float] | None = None,
                     touch: TouchController | None = None, tier: TierLogger | None = None,
                     replica: ReplicaFloor | None = None) -> FastAPI:
    app = FastAPI()
    st = app.state
    st.board, st.bus, st.rng = board, bus if bus is not None else InMemoryBus(), rng if rng is not None else np.random.default_rng(0)
    st.snapshot = None
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
        try:
            e_tool = float(board.expected_tool_next(d.get("session_id", "")))
            e_service = float(board.expected_service(d.get("session_id", ""), int(d.get("isl", 0)), int(d.get("osl", 0))))
        except Exception:                        # fail-open: defaults the proxy would use anyway
            e_tool, e_service = 0.0, 0.0
        ms = (time.perf_counter() - t0) * 1000.0
        return {"e_service_s": e_service, "e_tool_next_s": e_tool, "elapsed_ms": ms, "over_budget": ms > budget_s * 1000.0}

    @app.get("/directives")
    async def directives():
        now = clock()
        out = {"t": now, "holds": [], "touches": [], "tier": [], "replica": None}
        s = st.snapshot
        if s is None:
            return out
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
        return out

    return app
