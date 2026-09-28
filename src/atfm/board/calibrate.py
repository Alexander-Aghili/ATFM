"""Dispersion calibration: widen a forecaster's samples around their mean by a per-horizon factor
fitted on train-block ticks so that the central interval reaches its nominal coverage.

The session models draw sessions independently, so the fleet total is under-dispersed when sessions
share hidden state (same user, same repository, correlated stalls). This is a conformal-style fix:
it changes only the spread, never the mean, and is fitted on data the model was trained on, so the
held-out coverage it produces is an honest number.
"""
from __future__ import annotations

import numpy as np

from atfm.schema.forecast import CLASSES, TARGETS, ForecastSnapshot


def inflate(samples: np.ndarray, k: np.ndarray) -> np.ndarray:
    """samples (H, n), k (H,): mean + k * (sample - mean), clipped at 0."""
    mu = samples.mean(axis=1, keepdims=True)
    return np.clip(mu + k[:, None] * (samples - mu), 0.0, None)


def _coverage_at(sample_sets: list[np.ndarray], truths: list[np.ndarray], k: np.ndarray, level: float) -> np.ndarray:
    hits = []
    for s, y in zip(sample_sets, truths):
        w = inflate(s, k)
        lo = np.quantile(w, (1 - level) / 2, axis=1)
        hi = np.quantile(w, 1 - (1 - level) / 2, axis=1)
        hits.append((y >= lo) & (y <= hi))
    return np.mean(hits, axis=0)


def fit_inflation(forecaster, ticks: list[float], truth_fn, target: float = 0.9, rng=None,
                  target_name: str = "kv_blocks", grid: np.ndarray | None = None, states_fn=None,
                  on_tick=None) -> dict[str, np.ndarray]:
    """Per-class, per-horizon factor k (H,) = smallest grid value whose central `target` interval covers
    the truth at least `target` of the time over `ticks`. A class with no demand anywhere keeps k = 1.
    `on_tick(t)` runs before each forecast (e.g. to feed arrivals to the exogenous model)."""
    rng = np.random.default_rng(0) if rng is None else rng
    grid = np.concatenate([[1.0], np.geomspace(1.05, 20.0, 60)]) if grid is None else grid
    sample_sets = {c: [] for c in CLASSES}
    truths = {c: [] for c in CLASSES}
    for t in ticks:
        if on_tick is not None:
            on_tick(t)
        states = states_fn(t) if states_fn is not None else []
        snap = forecaster.forecast(t, states, rng)
        tr = truth_fn(t)[target_name]
        for c in CLASSES:
            sample_sets[c].append(snap.samples[target_name][c])
            truths[c].append(np.asarray(tr[c], float))
    H = sample_sets[CLASSES[0]][0].shape[0]
    out = {}
    for c in CLASSES:
        k = np.ones(H)
        if not any(np.any(y > 0) for y in truths[c]):
            out[c] = k
            continue
        for h in range(H):
            for g in grid:
                kk = np.ones(H)
                kk[h] = g
                if _coverage_at(sample_sets[c], truths[c], kk, target)[h] >= target:
                    k[h] = g
                    break
            else:
                k[h] = grid[-1]
        out[c] = k
    return out


class CalibratedForecaster:
    """Wraps any forecaster; inflates every target by the fitted per-class, per-horizon factor
    (a single array applies to every class)."""

    def __init__(self, inner, k):
        self.inner = inner
        self.k = {c: np.asarray(k[c], float) for c in k} if isinstance(k, dict) else {c: np.asarray(k, float) for c in CLASSES}
        base = (getattr(inner, "model_id", None) or getattr(getattr(inner, "predictor", None), "name", None)
                or getattr(getattr(inner, "series", None), "name", None) or "forecaster")
        self.model_id = f"{base}+cal"

    def __getattr__(self, name):  # delegate exo, predictor, observe, ...
        return getattr(self.inner, name)

    def forecast(self, t: float, states, rng) -> ForecastSnapshot:
        snap = self.inner.forecast(t, states, rng)
        samples = {tgt: {c: inflate(snap.samples[tgt][c], self.k.get(c, np.ones(len(snap.horizons))))
                         for c in snap.samples[tgt]} for tgt in snap.samples}
        return ForecastSnapshot(t=snap.t, horizons=snap.horizons, model_id=self.model_id, samples=samples,
                                endogenous_fraction=snap.endogenous_fraction)
