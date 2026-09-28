from __future__ import annotations

from collections import defaultdict

import numpy as np

from atfm.board.sampling import empirical_draw

from atfm.board.state import SessionState
from atfm.schema.trace import TraceTable, is_missing_scalar as _isnan

from .base import SessionPredictor

POOLED = "__pooled__"


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
        self._llm: dict[str, np.ndarray] = {}  # per-class LLM call durations (t_last_token - t_request)
        self._sorted_llm: dict[str | None, np.ndarray] = {}
        self._gaps: dict[str, np.ndarray] = {}  # per-tool pending gap: next t_request - t_tool_end

    def fit(self, train: TraceTable) -> "DurationModel":
        durs, last, gaps, deltas, spawns, llm = (defaultdict(list) for _ in range(6))
        for _, rows in train.session_records():
            for i, r in enumerate(rows):
                if not _isnan(r["t_last_token"]):
                    llm[r["class"]].append(max(float(r["t_last_token"] - r["t_request"]), 1e-3))
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
            self._gaps[tool] = np.sort(np.asarray(gaps[tool] or [0.0], float))
            self._isl_delta[tool] = np.asarray(deltas[tool] or [0], int)
            self._spawn_rate[tool] = float(np.mean(spawns[tool]))
        self._llm = {c: np.asarray(v) for c, v in llm.items()}
        self._sorted_llm.clear()
        if POOLED not in self.durations:
            self.durations[POOLED] = np.array([1.0])
            self.lognorm[POOLED] = (0.0, 0.5)
            self._no_return[POOLED] = 0.0
            self._overhead[POOLED] = 0.0
            self._isl_delta[POOLED] = np.array([0])
            self._spawn_rate[POOLED] = 0.0
            self._gaps[POOLED] = np.array([0.0])
        return self

    @staticmethod
    def _conditional(arr: np.ndarray, elapsed: float, n: int, rng: np.random.Generator,
                     min_conditional: int, lognorm: tuple[float, float] | None = None) -> np.ndarray:
        """Draw from the sorted empirical array conditioned on value > elapsed, with a tail fallback."""
        tail = arr[np.searchsorted(arr, elapsed, side="right"):]
        if len(tail) >= min_conditional:
            return empirical_draw(tail, n, rng)
        if lognorm is None:
            pos = arr[arr > 0]
            logs = np.log(pos) if len(pos) else np.array([0.0])
            lognorm = (float(logs.mean()), float(max(logs.std(), 0.1)))
        mu, sigma = lognorm
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

    def _key(self, tool: str | None) -> str:
        return tool if tool in self.durations else POOLED

    def sample(self, tool: str | None, n: int, rng: np.random.Generator) -> np.ndarray:
        return empirical_draw(self.durations[self._key(tool)], n, rng)

    def sample_conditional(self, tool: str | None, elapsed: float, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw durations D given D > elapsed (survival conditioning), with a log-normal tail fallback."""
        k = self._key(tool)
        return self._conditional(self.durations[k], elapsed, n, rng, self.min_conditional, self.lognorm[k])

    def gap(self, tool: str | None, n: int, rng: np.random.Generator) -> np.ndarray:
        """Pending gap after a tool of this kind ends (harness overhead, or a user who walked away)."""
        return empirical_draw(self._gaps[self._key(tool)], n, rng)

    def gap_conditional(self, tool: str | None, elapsed: float, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._conditional(self._gaps[self._key(tool)], elapsed, n, rng, self.min_conditional)

    def llm_duration_conditional(self, cls: str, elapsed: float, n: int, rng: np.random.Generator) -> np.ndarray:
        key = cls if cls in self._llm and len(self._llm[cls]) else None
        if key not in self._sorted_llm:
            arr = self._llm[key] if key is not None else (
                np.concatenate(list(self._llm.values())) if self._llm else np.array([1.0])
            )
            self._sorted_llm[key] = np.sort(arr)
        return self._conditional(self._sorted_llm[key], elapsed, n, rng, self.min_conditional)

    def llm_duration(self, cls: str, n: int, rng: np.random.Generator) -> np.ndarray:
        arr = self._llm.get(cls)
        if arr is None or len(arr) == 0:
            arr = np.concatenate(list(self._llm.values())) if self._llm else np.array([1.0])
        return empirical_draw(arr, n, rng)

    def mean(self, tool: str | None) -> float:
        return float(self.durations[self._key(tool)].mean())

    def no_return_prob(self, tool: str | None) -> float:
        return self._no_return[self._key(tool)]

    def overhead(self, tool: str | None) -> float:
        return self._overhead[self._key(tool)]

    def isl_delta(self, tool: str | None, n: int, rng: np.random.Generator) -> np.ndarray:
        return empirical_draw(self._isl_delta[self._key(tool)], n, rng)

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

    def _off_tool(self, s: SessionState, now: float, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sessions not running a tool.

        llm_pending: the tool ended and the next request has not arrived; the remaining gap is drawn
        from the per-tool pending-gap distribution conditioned on how long it has been pending.
        llm_running: the remaining decode (conditioned on elapsed) is followed by an unknown tool and
        its pending gap.
        """
        if s.phase == "llm_running":
            decode_left = self.dm.llm_duration_conditional(s.cls, s.elapsed(now), n, rng) - s.elapsed(now)
            return decode_left + self.dm.sample(None, n, rng) + self.dm.gap(None, n, rng)
        if s.phase == "llm_pending":
            return self.dm.gap_conditional(s.tool_name, s.elapsed(now), n, rng) - s.elapsed(now)
        return np.zeros(n)

    def _finish(self, s: SessionState, rem: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """Add the pending gap and the no-return mass to a remaining-tool-time draw."""
        r = rem + self.dm.gap(s.tool_name, n, rng)
        p = self.dm.no_return_prob(s.tool_name)
        if p > 0:
            r = np.where(rng.random(n) < p, np.inf, r)
        return r

    def resumption(self, s: SessionState, now: float, n: int, rng: np.random.Generator) -> np.ndarray:
        if s.phase != "tool_running":
            return self._off_tool(s, now, n, rng)
        d = self._draw_duration(s, now, n, rng)
        return self._finish(s, np.maximum(d - s.elapsed(now), 0.0), n, rng)


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
