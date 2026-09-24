"""Arms 3 and 4: the proxy index with a forecast term, plus a forecast-driven hold rule, on simulated time."""
from __future__ import annotations

from typing import Literal

import numpy as np

from atfm.board.forecaster import ExogenousModel, SessionForecaster
from atfm.board.live import SessionRegistry
from atfm.board.predictors import ProgressPredictor, SurvivalPredictor
from atfm.proxy.config import ProxyConfig
from atfm.proxy.index import compute_index, service_time, tier
from atfm.schema.forecast import ForecastSnapshot
from atfm.schema.trace import TraceTable
from atfm.traces.sidecar import events_to_trace_table

from .core import Simulator
from .engine import EngineConfig
from .policies import NativePolicy, _meta


class GdpLite:
    """Hold a deferrable call while the forecast's upper quantile of interactive demand within one slot
    exceeds the fleet's free KV blocks or free batch slots (spec 6.2, single-slot chance constraint)."""

    def __init__(self, slot_s: float = 30.0, eps: float = 0.1, max_hold_s: float = 600.0):
        self.slot_s, self.eps, self.max_hold_s = slot_s, eps, max_hold_s

    def hold_until(self, now: float, snap: ForecastSnapshot, free_blocks: int, free_slots: int, mean_isl: float,
                   e_service_s: float = 1.0) -> float | None:
        """`free_blocks` is capacity not held by running requests; the slot test compares expected busy
        slots over the slot (calls x E[S] / slot) with the free slots, not a raw call count."""
        h = int(np.argmin(np.abs(np.asarray(snap.horizons) - self.slot_s)))
        q = 1.0 - self.eps
        kv = float(np.quantile(snap.samples["kv_blocks"]["interactive"][h], q))
        calls = float(np.quantile(snap.samples["prefill_tokens"]["interactive"][h], q)) / max(mean_isl, 1.0)
        occupancy = calls * e_service_s / self.slot_s
        if kv > free_blocks or occupancy > free_slots:
            return now + self.slot_s
        return None


class ForecastPolicy:
    def __init__(self, window: int, cfg: ProxyConfig, predictor, train_table: TraceTable | None, horizons: list[float],
                 n: int = 128, hold: bool = True, gdp: GdpLite | None = None):
        self._window, self.cfg, self.predictor = window, cfg, predictor
        self.registry = SessionRegistry()
        exo = ExogenousModel()
        if train_table is not None:
            exo.fit(train_table)
        self.forecaster = SessionForecaster(predictor, exo, horizons, n=n)
        self.hold, self.gdp = hold, gdp or GdpLite()
        self.name = f"forecast_{predictor.name.split('_')[0]}"
        self.snapshot: ForecastSnapshot | None = None
        self.snapshots = 0
        self.last_hold_reason = ""
        self._starts_ptr = 0.0
        self._isl_sum, self._isl_n = 0.0, 0

    def window(self, sim) -> int | None:
        return self._window

    def _ingest(self, sim) -> None:
        for e in sim.events.drain():
            self.registry.apply(e)
            if e.kind == "llm.request":
                self._isl_sum += e.isl
                self._isl_n += 1
        for sid in list(self.registry._s):   # ended sessions must not be forecast as imminent demand
            run = sim.sessions.get(sid)
            if run is not None and run.done:
                self.registry.drop(sid)

    def on_tick(self, sim, now: float) -> None:
        self._ingest(sim)
        starts = [s for s in self.registry.new_starts_since(self._starts_ptr) if s[0] < now]
        self._starts_ptr = now
        self.forecaster.exo.update(now, starts)
        self.snapshot = self.forecaster.forecast(now, self.registry.states(now), sim.rng)
        self.snapshots += 1

    def e_tool_next(self, sim, call) -> float:
        """Expected duration of the tool this session will run next, conditioned on its tool history
        (current tool if running, else the last one); pooled mean when nothing is known."""
        st = self.registry._s.get(call.session.program.session_id)
        tool = None
        if st is not None:
            tool = st.tool_name if st.phase == "tool_running" else (st.tool_history[-1][0] if st.tool_history else None)
        return float(self.predictor.dm.mean(tool))

    def tier_and_index(self, sim, call) -> tuple[int, float]:
        m = _meta(call, sim.now)
        e_s = service_time(m, self.cfg)
        return tier(m, self.cfg, sim.now, e_s), compute_index(m, self.cfg, e_s, self.e_tool_next(sim, call))

    def on_arrival(self, sim, call) -> float | None:
        if not self.hold or call.session.program.cls != "background" or self.snapshot is None:
            return None
        free_blocks = sum(w.capacity_free_blocks() for w in sim.workers)
        free_slots = sum(max(0, w.cfg.max_batch - len(w.running)) for w in sim.workers)
        mean_isl = (self._isl_sum / self._isl_n) if self._isl_n else 3000.0
        e_service = mean_isl / self.cfg.prefill_tps + self.cfg.default_osl / self.cfg.decode_tps
        t = self.gdp.hold_until(sim.now, self.snapshot, free_blocks, free_slots, mean_isl, e_service_s=e_service)
        if t is not None:
            self.last_hold_reason = "forecast_surge"
        return t

    def on_tool_end(self, sim, session, now: float) -> None:
        return None


def fit_predictor_on_programs(kind: Literal["M1", "M2"], programs, engines: list[EngineConfig], rng):
    """Run the training programs under the native arm and fit the predictor on the resulting trace table."""
    sim = Simulator(programs, engines, NativePolicy(), rng=rng)
    sim.run()
    table = events_to_trace_table(sim.events.drain())
    pred = (SurvivalPredictor() if kind == "M1" else ProgressPredictor()).fit(table)
    return pred, table
