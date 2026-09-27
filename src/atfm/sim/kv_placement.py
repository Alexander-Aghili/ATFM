"""KV placement arms: the forecast (or the truth) decides which idle session's KV to evict, and admission is
rules-only. The engine asks the policy for an eviction order whenever it must make room; the policy answers
"the session forecast to return last goes first", with sessions it knows nothing about first of all.

This is the lever the second H2 batch points at: holds under load could not move the interactive tail,
but the recompute paid on resumption is set by which KV survives, which is a per-session return-time
prediction, exactly what the ladder scores."""
from __future__ import annotations

import numpy as np

from atfm.proxy.config import ProxyConfig
from atfm.schema.trace import TraceTable

from .forecast_arm import ForecastPolicy, next_call_of
from .policies import OraclePolicy


def order_victims(candidates: list[str], eta: dict[str, float], blocks: dict[str, int], now: float,
                  size_aware: bool, cls: dict[str, str] | None = None, bg_weight: float = 1.0) -> list[str]:
    """Eviction order, first to go first. Absence = max(0, eta - now); unknown sessions (no eta) go first.
    size_aware scores idle block-seconds (absence x resident blocks) instead of absence alone. bg_weight > 1
    multiplies background absence: the interactive SLO pays for interactive misses, so background contexts
    go first at equal predicted absence (Belady weights every miss equally; the objective does not)."""
    def score(sid: str) -> float:
        e = eta.get(sid, float("inf"))
        if not np.isfinite(e):
            return float("inf")
        absence = max(0.0, e - now)
        s = absence * blocks.get(sid, 1) if size_aware else absence
        if cls is not None and cls.get(sid) == "background":
            s *= bg_weight
        return s
    return sorted(candidates, key=lambda sid: -score(sid))


class _KvOrdering:
    """Shared eviction ordering over a per-session expected return time (absolute seconds). Admission is
    exactly `proxy_rules` (no next-tool index term), so the arms differ from rules by placement alone."""
    _eta: dict[str, float]
    size_aware: bool = False
    bg_weight: float = 1.0
    _now: float = 0.0

    def e_tool_next(self, sim, call) -> float:
        return 0.0

    def kv_victims(self, sim, worker, candidates: list[str]) -> list[str]:
        blocks = {sid: worker.resident_blocks(sid) for sid in candidates}
        cls = {sid: sim.sessions[sid].program.cls for sid in candidates if sid in sim.sessions}
        return order_victims(candidates, self._eta, blocks, sim.now, self.size_aware, cls, self.bg_weight)


class OracleKvPolicy(_KvOrdering, OraclePolicy):
    """Rules-only admission plus eviction by the *true* next-call time read from the event heap."""

    name = "oracle_kv"

    def __init__(self, window: int, cfg: ProxyConfig, size_aware: bool = False, bg_weight: float = 1.0):
        super().__init__(window, cfg, hold=False)
        self._eta = {}
        self.size_aware, self.bg_weight = size_aware, bg_weight
        if size_aware:
            self.name = "oracle_kv_size"
        elif bg_weight != 1.0:
            self.name = "oracle_kv_cw"

    def on_tick(self, sim, now: float) -> None:
        eta: dict[str, float] = {}
        for t, _, kind, payload in sorted(sim._heap, key=lambda x: (x[0], x[1])):
            if kind not in ("start", "arrive", "tool_end"):
                continue
            nc = next_call_of(sim, kind, payload)
            if nc is not None and nc[0] not in eta:
                eta[nc[0]] = float(t)
        for c in sim.proxy_queue:                 # waiting for the window: no heap event, but the call is imminent
            eta[c.session.program.session_id] = max(now, float(c.release_not_before))
        for w in sim.workers:                     # released but not yet scheduled (batch full or no KV room): imminent
            for req in w.queue:
                eta[req.session_id] = now
            for req, _, t_end in w.running.values():   # running now: the next call comes after this one and its tool
                s = sim.sessions.get(req.session_id)
                if s is None:
                    continue
                turn = s.program.turns[min(s.turn, len(s.program.turns) - 1)]
                if s.turn + 1 >= len(s.program.turns) or turn.tool_name is None:
                    continue                                    # last call: the session ends, unknown is right
                eta[req.session_id] = float(t_end + (turn.tool_duration or 0.0) + sim.harness_overhead_s)
        self._eta = eta


class ForecastKvPolicy(_KvOrdering, ForecastPolicy):
    """Rules-only admission plus eviction by the predictor's median time-to-next-call per session."""

    def __init__(self, window: int, cfg: ProxyConfig, predictor, train_table: TraceTable | None, horizons: list[float],
                 n: int = 64, size_aware: bool = False, bg_weight: float = 1.0):
        super().__init__(window, cfg, predictor, train_table, horizons, n=n, hold=False)
        self.size_aware, self.bg_weight = size_aware, bg_weight
        self.name = f"forecast_{predictor.name.split('_')[0]}_kv" + ("_size" if size_aware else "_cw" if bg_weight != 1.0 else "")
        self._eta = {}
        self._n = n

    def on_tick(self, sim, now: float) -> None:
        super().on_tick(sim, now)
        eta: dict[str, float] = {}
        for st in self.registry.states(now):
            samples = self.predictor.resumption(st, now, self._n, sim.rng)
            finite = samples[np.isfinite(samples)]
            eta[st.session_id] = now + float(np.median(finite)) if len(finite) else float("inf")
        self._eta = eta
