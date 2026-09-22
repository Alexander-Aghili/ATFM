from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from atfm.board.state import SessionState
from atfm.schema.trace import TraceTable

from .base import SessionPredictor

POOLED = "__pooled__"


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


class DurationModel:
    """Per-tool empirical duration distributions plus the small per-tool nuisance quantities."""

    def __init__(self, min_conditional: int = 5):
        self.min_conditional = min_conditional
        self.durations: dict[str, np.ndarray] = {}
        self.lognorm: dict[str, tuple[float, float]] = {}
        self._no_return: dict[str, float] = {}
        self._overhead: dict[str, float] = {}
        self._isl_delta: dict[str, np.ndarray] = {}
        self._spawn_rate: dict[str, float] = {}

    def fit(self, train: TraceTable) -> "DurationModel":
        durs, last, gaps, deltas, spawns = (defaultdict(list) for _ in range(5))
        for _, g in train.sessions():
            rows = g.to_dict("records")
            for i, r in enumerate(rows):
                tool = r["tool_name"]
                if tool is None or _isnan(r["t_tool_start"]) or _isnan(r["t_tool_end"]):
                    continue
                d = max(float(r["t_tool_end"] - r["t_tool_start"]), 1e-3)
                is_last = i + 1 >= len(rows)
                for key in (tool, POOLED):
                    durs[key].append(d)
                    last[key].append(1.0 if is_last else 0.0)
                    spawns[key].append(float(r["spawned_children"]))
                if not is_last:
                    nxt = rows[i + 1]
                    gap = max(0.0, float(nxt["t_request"] - r["t_tool_end"]))
                    dl = max(0, int(nxt["isl"]) - (int(r["isl"]) + int(r["osl"])))
                    for key in (tool, POOLED):
                        gaps[key].append(gap)
                        deltas[key].append(dl)
        for tool, ds in durs.items():
            arr = np.sort(np.asarray(ds, float))
            self.durations[tool] = arr
            logs = np.log(arr)
            self.lognorm[tool] = (float(logs.mean()), float(max(logs.std(), 0.1)))
            self._no_return[tool] = float(np.mean(last[tool]))
            self._overhead[tool] = float(np.median(gaps[tool])) if gaps[tool] else 0.0
            self._isl_delta[tool] = np.asarray(deltas[tool] or [0], int)
            self._spawn_rate[tool] = float(np.mean(spawns[tool]))
        if POOLED not in self.durations:
            self.durations[POOLED] = np.array([1.0])
            self.lognorm[POOLED] = (0.0, 0.5)
            self._no_return[POOLED] = 0.0
            self._overhead[POOLED] = 0.0
            self._isl_delta[POOLED] = np.array([0])
            self._spawn_rate[POOLED] = 0.0
        return self

    def _key(self, tool: str | None) -> str:
        return tool if tool in self.durations else POOLED

    def sample(self, tool: str | None, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.durations[self._key(tool)], size=n, replace=True)

    def sample_conditional(self, tool: str | None, elapsed: float, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw durations D given D > elapsed (survival conditioning), with a log-normal tail fallback."""
        k = self._key(tool)
        arr = self.durations[k]
        tail = arr[np.searchsorted(arr, elapsed, side="right"):]
        if len(tail) >= self.min_conditional:
            return rng.choice(tail, size=n, replace=True)
        mu, sigma = self.lognorm[k]
        out = np.empty(n)
        filled = 0
        for _ in range(20):
            cand = rng.lognormal(mu, sigma, size=4 * n)
            cand = cand[cand > elapsed]
            take = min(len(cand), n - filled)
            out[filled:filled + take] = cand[:take]
            filled += take
            if filled >= n:
                break
        if filled < n:
            out[filled:] = elapsed * rng.lognormal(0.0, 0.5, size=n - filled) + elapsed
        return out

    def mean(self, tool: str | None) -> float:
        return float(self.durations[self._key(tool)].mean())

    def no_return_prob(self, tool: str | None) -> float:
        return self._no_return[self._key(tool)]

    def overhead(self, tool: str | None) -> float:
        return self._overhead[self._key(tool)]

    def isl_delta(self, tool: str | None, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self._isl_delta[self._key(tool)], size=n, replace=True)

    def spawn_rate(self, tool: str | None) -> float:
        return self._spawn_rate[self._key(tool)]


class HistoryPredictor(SessionPredictor):
    """B2: per-tool historical durations, elapsed time ignored (Continuum-style TTL from start)."""

    name = "B2_history"

    def __init__(self):
        self.dm = DurationModel()

    def fit(self, train: TraceTable) -> "HistoryPredictor":
        self.dm.fit(train)
        return self

    def _draw_duration(self, s: SessionState, now: float, n: int, rng) -> np.ndarray:
        return self.dm.sample(s.tool_name, n, rng)

    def resumption(self, s: SessionState, now: float, n: int, rng: np.random.Generator) -> np.ndarray:
        if s.phase != "tool_running":
            return np.zeros(n)
        d = self._draw_duration(s, now, n, rng)
        r = np.maximum(d - s.elapsed(now), 0.0) + self.dm.overhead(s.tool_name)
        p = self.dm.no_return_prob(s.tool_name)
        if p > 0:
            r = np.where(rng.random(n) < p, np.inf, r)
        return r

    def next_call_isl(self, s: SessionState, n: int, rng: np.random.Generator) -> np.ndarray:
        return (s.ctx_tokens + self.dm.isl_delta(s.tool_name, n, rng)).astype(int)

    def spawn(self, s: SessionState, horizon: float, n: int, rng: np.random.Generator) -> np.ndarray:
        rate = self.dm.spawn_rate(s.tool_name)
        if rate <= 0:
            return np.zeros(n, int)
        frac = min(1.0, horizon / max(self.dm.mean(s.tool_name), 1e-6))
        return rng.poisson(rate * frac, size=n)


class SurvivalPredictor(HistoryPredictor):
    """M1: B2 plus conditioning on elapsed time (survival)."""

    name = "M1_survival"

    def _draw_duration(self, s: SessionState, now: float, n: int, rng) -> np.ndarray:
        return self.dm.sample_conditional(s.tool_name, s.elapsed(now), n, rng)
