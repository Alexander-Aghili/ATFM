"""Arms 3 and 4: the proxy index with a forecast term, plus a forecast-driven hold rule, on simulated time."""
from __future__ import annotations

from typing import Literal

import numpy as np

from atfm.board.forecaster import CLASSES, TARGETS, ExogenousModel, SessionForecaster
from atfm.board.live import SessionRegistry
from atfm.board.predictors import ProgressPredictor, SurvivalPredictor
from atfm.proxy.config import ProxyConfig
from atfm.proxy.index import compute_index, service_time, tier
from atfm.schema.forecast import ForecastSnapshot
from atfm.schema.trace import TraceTable
from atfm.traces.sidecar import events_to_trace_table

from .core import Simulator
from .engine import EngineConfig
from .policies import NativePolicy, OraclePolicy, _meta


class GdpLite:
    """Hold a deferrable call while the forecast's upper quantile of interactive demand within one slot
    exceeds the fleet's free KV blocks or free batch slots (spec 6.2, single-slot chance constraint)."""

    def __init__(self, slot_s: float = 30.0, eps: float = 0.1, max_hold_s: float = 600.0):
        self.slot_s, self.eps, self.max_hold_s = slot_s, eps, max_hold_s

    def hold_until(self, now: float, snap: ForecastSnapshot, free_blocks: int, free_slots: int, mean_isl: float,
                   e_service_s: float = 1.0, freeing_slots: int = 0, freeing_blocks: int = 0) -> float | None:
        """`free_blocks` is capacity not held by running requests; the slot test compares expected busy
        slots over the slot (calls x E[S] / slot) with the slots free now plus those that free up inside
        the slot (`freeing_*`, v2). Without the freeing terms an instantaneous test under load sees no free
        capacity at any moment and holds every deferrable call regardless of the forecast."""
        h = int(np.argmin(np.abs(np.asarray(snap.horizons) - self.slot_s)))
        q = 1.0 - self.eps
        kv = float(np.quantile(snap.samples["kv_blocks"]["interactive"][h], q))
        calls = float(np.quantile(snap.samples["prefill_tokens"]["interactive"][h], q)) / max(mean_isl, 1.0)
        occupancy = calls * e_service_s / self.slot_s
        if kv > free_blocks + freeing_blocks or occupancy > free_slots + freeing_slots:
            return now + self.slot_s
        return None


def freeing_capacity(sim, now: float, slot_s: float, e_service_s: float | None) -> tuple[int, int]:
    """(slots, KV blocks) held by running requests expected to finish within the slot. With `e_service_s`
    the end is estimated as start + E[S] (what a proxy can know); with None the engine's true end is used
    (oracle)."""
    slots, blocks = 0, 0
    for w in sim.workers:
        for req, t_start, t_end in w.running.values():
            end = t_end if e_service_s is None else t_start + e_service_s
            if end <= now + slot_s:
                slots += 1
                blocks += w.resident_blocks(req.session_id)
    return slots, blocks


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
        fs, fb = freeing_capacity(sim, sim.now, self.gdp.slot_s, e_service)
        t = self.gdp.hold_until(sim.now, self.snapshot, free_blocks, free_slots, mean_isl, e_service_s=e_service,
                                freeing_slots=fs, freeing_blocks=fb)
        if t is not None:
            self.last_hold_reason = "forecast_surge"
        return t

    def on_tool_end(self, sim, session, now: float) -> None:
        return None


def next_call_of(sim, kind: str, payload) -> tuple[str, str, int] | None:
    """(session id, class, prompt tokens) of the LLM call a heap event (start / arrive / tool_end) leads to."""
    if kind == "start":
        p = payload
        return p.session_id, p.cls, int(p.turns[0].isl_new)
    if kind == "arrive":
        s = sim.sessions.get(payload)
        if s is None or s.turn >= len(s.program.turns):
            return None
        turn = s.program.turns[s.turn]
        return s.program.session_id, s.program.cls, int(turn.isl_new if turn.reset else s.ctx + turn.isl_new)
    if kind == "tool_end":
        s = sim.sessions.get(payload[0])
        ti = payload[2] + 1
        if s is None or ti >= len(s.program.turns):
            return None
        turn = s.program.turns[ti]
        return s.program.session_id, s.program.cls, int(turn.isl_new if turn.reset else s.ctx + turn.isl_new)
    return None


class OracleRulePolicy(OraclePolicy):
    """Like-for-like upper bound for the forecast arms: the same GdpLite hold rule and the same index,
    fed the *true* first-call demand per horizon (read from the event heap) instead of forecast samples,
    and the true duration of the next tool. Whatever the forecast arms lose against this arm is forecast
    error; whatever this arm loses against `native` on background cost is the rule itself."""

    name = "oracle_rule"

    def __init__(self, window: int, cfg: ProxyConfig, horizons: list[float] | None = None, n: int = 8,
                 gdp: GdpLite | None = None, block_size: int = 16):
        super().__init__(window, cfg, hold=True)
        self.horizons, self.n, self.gdp, self.block_size = list(horizons or [30.0, 120.0, 300.0]), n, gdp or GdpLite(), block_size
        self.snapshot: ForecastSnapshot | None = None
        self.snapshots = 0
        self._isl_sum, self._isl_n = 0.0, 0

    def _next_call(self, sim, kind: str, payload) -> tuple[str, str, int] | None:
        return next_call_of(sim, kind, payload)

    def on_tick(self, sim, now: float) -> None:
        for e in sim.events.drain():
            if e.kind == "llm.request":
                self._isl_sum += e.isl
                self._isl_n += 1
        H = len(self.horizons)
        samples = {tgt: {c: np.zeros((H, self.n)) for c in CLASSES} for tgt in TARGETS}
        seen: set[str] = set()
        for t, _, kind, payload in sorted(sim._heap, key=lambda x: (x[0], x[1])):
            if kind not in ("start", "arrive", "tool_end"):
                continue
            nc = self._next_call(sim, kind, payload)
            if nc is None or nc[0] in seen:
                continue
            seen.add(nc[0])                       # demand truth: first call per session per horizon
            sid, cls, isl = nc
            for i, h in enumerate(self.horizons):
                if t <= now + h:
                    samples["kv_blocks"][cls][i, :] += float(np.ceil(isl / self.block_size))
                    samples["prefill_tokens"][cls][i, :] += float(isl)
        self.snapshot = ForecastSnapshot(t=now, horizons=self.horizons, model_id=self.name, samples=samples)
        self.snapshots += 1

    def on_arrival(self, sim, call) -> float | None:
        if call.session.program.cls != "background" or self.snapshot is None:
            return None
        free_blocks = sum(w.capacity_free_blocks() for w in sim.workers)
        free_slots = sum(max(0, w.cfg.max_batch - len(w.running)) for w in sim.workers)
        mean_isl = (self._isl_sum / self._isl_n) if self._isl_n else 3000.0
        e_service = mean_isl / self.cfg.prefill_tps + self.cfg.default_osl / self.cfg.decode_tps
        fs, fb = freeing_capacity(sim, sim.now, self.gdp.slot_s, None)     # true completion times
        t = self.gdp.hold_until(sim.now, self.snapshot, free_blocks, free_slots, mean_isl, e_service_s=e_service,
                                freeing_slots=fs, freeing_blocks=fb)
        if t is not None:
            self.last_hold_reason = "oracle_rule_surge"
        return t


def fit_predictor_on_programs(kind: Literal["M1", "M2"], programs, engines: list[EngineConfig], rng):
    """Run the training programs under the native arm and fit the predictor on the resulting trace table."""
    sim = Simulator(programs, engines, NativePolicy(), rng=rng)
    sim.run()
    table = events_to_trace_table(sim.events.drain())
    pred = (SurvivalPredictor() if kind == "M1" else ProgressPredictor()).fit(table)
    return pred, table
