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


class _KvOrdering:
    """Shared eviction ordering over a per-session expected return time (absolute seconds)."""
    _eta: dict[str, float]

    def kv_victims(self, sim, worker, candidates: list[str]) -> list[str]:
        eta = self._eta
        return sorted(candidates, key=lambda sid: -eta.get(sid, float("inf")))


class OracleKvPolicy(_KvOrdering, OraclePolicy):
    """Rules-only admission plus eviction by the *true* next-call time read from the event heap."""

    name = "oracle_kv"

    def __init__(self, window: int, cfg: ProxyConfig):
        super().__init__(window, cfg, hold=False)
        self._eta = {}

    def on_tick(self, sim, now: float) -> None:
        eta: dict[str, float] = {}
        for t, _, kind, payload in sorted(sim._heap, key=lambda x: (x[0], x[1])):
            if kind not in ("start", "arrive", "tool_end"):
                continue
            nc = next_call_of(sim, kind, payload)
            if nc is not None and nc[0] not in eta:
                eta[nc[0]] = float(t)
        self._eta = eta


class ForecastKvPolicy(_KvOrdering, ForecastPolicy):
    """Rules-only admission plus eviction by the predictor's median time-to-next-call per session."""

    def __init__(self, window: int, cfg: ProxyConfig, predictor, train_table: TraceTable | None, horizons: list[float],
                 n: int = 64):
        super().__init__(window, cfg, predictor, train_table, horizons, n=n, hold=False)
        self.name = f"forecast_{predictor.name.split('_')[0]}_kv"
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
