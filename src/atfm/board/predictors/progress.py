from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from atfm.board.sampling import empirical_draw

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


class ProgressCurve:
    """Per-tool monotone map from reported progress fraction to elapsed-time fraction, learned from
    training phases. cmake percent markers or test counts are rarely linear in time; the curve turns
    "50% reported" into "about 20% of the duration has passed" for such tools. Unknown tools are linear."""

    def __init__(self, bins: int = 20, min_phases: int = 2, sigma_floor: float = 0.05):
        self.bins, self.min_phases, self.sigma_floor = bins, min_phases, sigma_floor
        self.curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # tool -> (progress grid, time fraction)
        self.sigma: dict[str, float] = {}                             # tool -> sd of log(actual / predicted duration)

    @staticmethod
    def composite(tool: str | None, key: str | None) -> str | None:
        """Curve id for a (tool, command signature) pair; None when there is no signature."""
        if tool is None or key is None or (isinstance(key, float) and math.isnan(key)):
            return None
        return f"{tool}|{key}"

    def fit(self, train: TraceTable) -> "ProgressCurve":
        pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
        phases: dict[str, int] = defaultdict(int)
        self._collect_pairs(train, pairs, phases)
        edges = np.linspace(0.0, 1.0, self.bins + 1)
        for tool, pts in pairs.items():
            if phases[tool] < self.min_phases:
                continue
            self._fit_curve(tool, pts, edges)
        return self

    def _fit_curve(self, tool, pts, edges):
        arr = np.asarray(pts)
        pf, tf = np.clip(arr[:, 0], 0, 1), np.clip(arr[:, 1], 0, 1)
        grid, vals = [0.0], [0.0]
        for i in range(self.bins):
            m = (pf >= edges[i]) & (pf < edges[i + 1] if i < self.bins - 1 else pf <= edges[i + 1])
            if m.any():
                grid.append(float(np.median(pf[m]))); vals.append(float(np.median(tf[m])))
        grid.append(1.0); vals.append(1.0)
        vals = np.maximum.accumulate(np.asarray(vals))                # monotone in progress
        g = np.asarray(grid)
        self.curves[tool] = (g, vals)
        self._fit_sigma(tool, pf, tf, g, vals)

    def _fit_sigma(self, tool, pf, tf, g, vals):
        pred_tf = np.interp(pf, g, vals)
        ok = pred_tf > 0.02
        if ok.any():
            # predicted duration = elapsed / tf; actual = dur; ratio spread is the curve's uncertainty
            ratio = (tf[ok] / pred_tf[ok])   # = actual_dur / predicted_dur
            self.sigma[tool] = float(max(np.std(np.log(np.clip(ratio, 1e-3, 1e3))), self.sigma_floor))
        else:
            self.sigma[tool] = 0.5

    def _collect_pairs(self, train, pairs, phases):
        for _, rows in train.session_records():
            for r in rows:
                ev = r["progress_events"] or []
                ts, te = r["t_tool_start"], r["t_tool_end"]
                if r["tool_name"] is None or not ev or ts is None or te is None or (isinstance(te, float) and math.isnan(te)):
                    continue
                dur = float(te) - float(ts)
                if dur <= 0:
                    continue
                usable = self._usable_pairs(ev, ts, dur)
                if not usable:
                    continue
                for k in (r["tool_name"], self.composite(r["tool_name"], r.get("tool_args_hash"))):
                    if k is not None:               # one curve per tool name, one per (tool, signature)
                        phases[k] += 1
                        pairs[k].extend(usable)

    def _usable_pairs(self, ev, ts, dur):
        usable = [(float(e["completed"]) / float(e["total"]), (float(e["t"]) - float(ts)) / dur)
                  for e in ev if e.get("total") not in (None, 0) and e.get("completed") is not None]
        return usable

    def _which(self, tool: str | None, key: str | None) -> str | None:
        c = self.composite(tool, key)
        if c in self.curves:
            return c
        return tool if tool in self.curves else None

    def has(self, tool: str | None, key: str | None = None) -> bool:
        return self._which(tool, key) is not None

    def sigma_for(self, tool: str | None, key: str | None = None, default: float = 0.5) -> float:
        w = self._which(tool, key)
        return self.sigma.get(w, default) if w is not None else default

    def time_fraction(self, tool: str | None, progress_fraction: float, key: str | None = None) -> float:
        """Signature-specific curve when one was learned, else the tool-name curve, else linear."""
        pf = float(np.clip(progress_fraction, 0.0, 1.0))
        w = self._which(tool, key)
        if w is None:
            return pf
        g, v = self.curves[w]
        return float(np.interp(pf, g, v))


class ProgressPredictor(SurvivalPredictor):
    """M2: M1 plus a Bayesian rate filter over live progress events."""

    name = "M2_progress"

    def __init__(self):
        super().__init__()
        self._residual: dict[str, np.ndarray] = {}
        self.curve = ProgressCurve()

    def fit(self, train: TraceTable) -> "ProgressPredictor":
        super().fit(train)
        self.curve.fit(train)
        res = defaultdict(list)
        for _, rows in train.session_records():
            for r in rows:
                ev = r["progress_events"] or []
                t_end = r["t_tool_end"]
                if r["tool_name"] is None or not ev or t_end is None or (isinstance(t_end, float) and math.isnan(t_end)):
                    continue
                resid = self._phase_residual(r, ev, t_end)
                res[r["tool_name"]].append(resid)
                res[POOLED].append(resid)
        self._residual = {k: np.asarray(v) for k, v in res.items()}
        return self

    def _phase_residual(self, r, ev, t_end):
        last = max(ev, key=lambda e: e["t"])
        last_t = float(last["t"])
        resid = max(0.0, float(t_end) - last_t)
        # Residual = end-phase time beyond what extrapolating the observed rate already covers.
        total, done = last.get("total"), last.get("completed")
        t_start = r["t_tool_start"]
        if total and done and done > 0 and not (isinstance(t_start, float) and math.isnan(t_start)):
            extrapolated = (float(total) - float(done)) * (last_t - float(t_start)) / float(done)
            resid = max(0.0, resid - extrapolated)
        return resid

    def _residual_draw(self, tool: str | None, n: int, rng) -> np.ndarray:
        arr = self._residual.get(tool, self._residual.get(POOLED))
        if arr is None or len(arr) == 0:
            return np.zeros(n)
        return empirical_draw(arr, n, rng)

    def _remaining_from_progress(self, s: SessionState, now: float, n: int, rng) -> np.ndarray | None:
        usable = [e for e in s.progress if e.get("total") not in (None, 0) and e.get("completed") is not None]
        if not usable:
            return None
        latest = max(usable, key=lambda e: e["t"])
        total, done, t_latest = float(latest["total"]), float(latest["completed"]), float(latest["t"])
        t_start = s.t_tool_start if s.t_tool_start is not None else s.t_phase_start
        key = getattr(s, "tool_args_hash", None)
        tf = self.curve.time_fraction(s.tool_name, done / total, key) if self.curve.has(s.tool_name, key) else None
        if tf is not None and tf > 0.02:
            # learned non-linear progress: elapsed at the latest report / time fraction = duration estimate
            d_hat = (t_latest - t_start) / tf
            d = d_hat * rng.lognormal(0.0, self.curve.sigma_for(s.tool_name, key), size=n)
            remaining = np.maximum(d - (now - t_start), 0.0) + self._residual_draw(s.tool_name, n, rng)
            return remaining
        return self._rate_remaining(s, now, n, rng, usable, t_start, t_latest, total, done)

    def _rate_remaining(self, s, now, n, rng, usable, t_start, t_latest, total, done):
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
