from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from atfm.board.state import SessionState
from atfm.schema.trace import TraceTable

from .duration import POOLED, SurvivalPredictor


def rate_posterior(progress: list[dict], t_start: float, now: float, prior_shape: float = 2.0,
                   prior_rate_per_unit: float | None = None) -> tuple[float, float]:
    """Gamma(shape, rate) posterior over work rate (units/s) from (dw, dt) increments."""
    pts = [(t_start, 0.0)] + [(float(e["t"]), float(e["completed"])) for e in progress
                              if e.get("completed") is not None]
    pts.sort()
    dw = np.diff([p[1] for p in pts])
    dt = np.diff([p[0] for p in pts])
    mask = dt > 0
    dw, dt = dw[mask], dt[mask]
    if prior_rate_per_unit is None:
        if len(pts) > 1:
            prior_rate_per_unit = prior_shape * max(now - t_start, 1.0) / max(pts[-1][1], 1.0)
        else:
            prior_rate_per_unit = prior_shape
    shape = prior_shape + float(dw.sum())
    rate = prior_rate_per_unit + float(dt.sum())
    return shape, rate


class ProgressPredictor(SurvivalPredictor):
    """M2: M1 plus a Bayesian rate filter over live progress events."""

    name = "M2_progress"

    def __init__(self):
        super().__init__()
        self._residual: dict[str, np.ndarray] = {}

    def fit(self, train: TraceTable) -> "ProgressPredictor":
        super().fit(train)
        res = defaultdict(list)
        for _, g in train.sessions():
            for r in g.to_dict("records"):
                ev = r["progress_events"] or []
                t_end = r["t_tool_end"]
                if r["tool_name"] is None or not ev or t_end is None or (isinstance(t_end, float) and math.isnan(t_end)):
                    continue
                last = max(ev, key=lambda e: e["t"])
                last_t = float(last["t"])
                resid = max(0.0, float(t_end) - last_t)
                # Residual = end-phase time beyond what extrapolating the observed rate already covers.
                total, done = last.get("total"), last.get("completed")
                t_start = r["t_tool_start"]
                if total and done and done > 0 and not (isinstance(t_start, float) and math.isnan(t_start)):
                    extrapolated = (float(total) - float(done)) * (last_t - float(t_start)) / float(done)
                    resid = max(0.0, resid - extrapolated)
                res[r["tool_name"]].append(resid)
                res[POOLED].append(resid)
        self._residual = {k: np.asarray(v) for k, v in res.items()}
        return self

    def _residual_draw(self, tool: str | None, n: int, rng) -> np.ndarray:
        arr = self._residual.get(tool, self._residual.get(POOLED))
        if arr is None or len(arr) == 0:
            return np.zeros(n)
        return rng.choice(arr, size=n, replace=True)

    def _remaining_from_progress(self, s: SessionState, now: float, n: int, rng) -> np.ndarray | None:
        usable = [e for e in s.progress if e.get("total") not in (None, 0) and e.get("completed") is not None]
        if not usable:
            return None
        latest = max(usable, key=lambda e: e["t"])
        total, done, t_latest = float(latest["total"]), float(latest["completed"]), float(latest["t"])
        t_start = s.t_tool_start if s.t_tool_start is not None else s.t_phase_start
        shape, rate = rate_posterior(usable, t_start, now)
        r = rng.gamma(shape, 1.0 / rate, size=n)
        work_left = max(total - done, 0.0)
        remaining = work_left / np.maximum(r, 1e-9) + self._residual_draw(s.tool_name, n, rng)
        remaining = np.maximum(remaining - (now - t_latest), 0.0)
        for e in s.data:
            if e.get("metric") == "early_stop_prob":
                p = float(e["value"])
                remaining = np.where(rng.random(n) < p, 0.0, remaining)
        return remaining

    def resumption(self, s: SessionState, now: float, n: int, rng: np.random.Generator, *,
                   factors: dict[str, np.ndarray] | None = None) -> np.ndarray:
        if s.phase != "tool_running":
            return self._off_tool(s, now, n, rng)
        rem = self._remaining_from_progress(s, now, n, rng)
        if rem is None:
            d = self._draw_duration(s, now, n, rng)
            rem = np.maximum(d - s.elapsed(now), 0.0)
        rem = self._scale(s, rem, n, rng, factors)
        return self._finish(s, rem, n, rng)

    def _scale(self, s, rem, n, rng, factors):
        return rem
