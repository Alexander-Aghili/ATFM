from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from atfm.board.predictors.backend import BackendPredictor
from atfm.board.predictors.base import SeriesPredictor, SessionPredictor
from atfm.board.state import SessionState
from atfm.schema.forecast import ForecastSnapshot
from atfm.schema.trace import TraceTable

CLASSES = ("interactive", "background")
TARGETS = ("kv_blocks", "prefill_tokens")


class ExogenousModel:
    """Demand from sessions that do not exist yet: nonhomogeneous Poisson arrivals with empirical first-call size."""

    def __init__(self, window_s: float = 1800.0, block_size: int = 16, min_span_s: float = 60.0):
        self.window_s, self.block_size, self.min_span_s = window_s, block_size, min_span_s
        self.first_isl: dict[str, np.ndarray] = {c: np.array([256]) for c in CLASSES}
        self._starts: dict[str, deque] = {c: deque() for c in CLASSES}
        self._t_first: float | None = None  # first update time: the rate divides by the observed span
        self._now: float = 0.0

    def fit(self, train: TraceTable) -> "ExogenousModel":
        firsts = train.df[train.df["turn_index"] == 0]
        for c in CLASSES:
            arr = firsts.loc[firsts["class"] == c, "isl"].to_numpy(int)
            if len(arr):
                self.first_isl[c] = arr
        return self

    def update(self, t: float, new_session_starts: list[tuple[float, str]]) -> None:
        if self._t_first is None:
            self._t_first = min([t] + [ts for ts, _ in new_session_starts])
        self._now = t
        for ts, c in new_session_starts:
            self._starts[c].append(ts)
        for c in CLASSES:
            while self._starts[c] and self._starts[c][0] < t - self.window_s:
                self._starts[c].popleft()

    def rate(self, cls: str, now: float | None = None) -> float:
        if self._t_first is None:
            return 0.0
        now = self._now if now is None else now
        span = min(self.window_s, max(now - self._t_first, self.min_span_s))
        return len(self._starts[cls]) / span

    def draw(self, cls: str, horizon: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        k = rng.poisson(self.rate(cls) * horizon, size=n)
        kv = np.zeros(n)
        pf = np.zeros(n)
        for i in np.nonzero(k)[0]:
            isl = rng.choice(self.first_isl[cls], size=k[i], replace=True)
            kv[i] = np.ceil(isl / self.block_size).sum()
            pf[i] = isl.sum()
        return kv, pf

    def first_call_isl(self, cls: str, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.first_isl[cls], size=n, replace=True)


def _empty(horizons, n):
    return {t: {c: np.zeros((len(horizons), n)) for c in CLASSES} for t in TARGETS}


class SessionForecaster:
    """Monte Carlo aggregate of per-session resumption draws, children and exogenous arrivals (spec 5.3)."""

    def __init__(self, predictor: SessionPredictor, exo: ExogenousModel, horizons: list[float], n: int = 512,
                 block_size: int = 16):
        self.predictor, self.exo, self.horizons, self.n = predictor, exo, list(horizons), n
        self.block_size = block_size

    def forecast(self, t: float, states: list[SessionState], rng: np.random.Generator) -> ForecastSnapshot:
        H, n = len(self.horizons), self.n
        hz = np.asarray(self.horizons)[:, None]
        samples = _empty(self.horizons, n)
        endo = {c: np.zeros((H, n)) for c in CLASSES}
        factors = None
        if isinstance(self.predictor, BackendPredictor):
            backends = {s.backend_id for s in states if s.backend_id is not None}
            factors = {b: self.predictor.factor.draw(b, n, rng) for b in backends}
        for s in states:
            kw = {"factors": factors} if factors is not None else {}
            R = self.predictor.resumption(s, t, n, rng, **kw)
            isl = self.predictor.next_call_isl(s, n, rng)
            due = R[None, :] <= hz  # (H, n)
            kv = np.ceil(isl / self.block_size)
            samples["kv_blocks"][s.cls] += due * kv[None, :]
            samples["prefill_tokens"][s.cls] += due * isl[None, :]
            endo[s.cls] += due * kv[None, :]
            self._children(s, R, samples, rng)
        for c in CLASSES:
            for k, h in enumerate(self.horizons):
                kv, pf = self.exo.draw(c, h, n, rng)
                samples["kv_blocks"][c][k] += kv
                samples["prefill_tokens"][c][k] += pf
        endo_frac = {}
        for c in CLASSES:
            tot = samples["kv_blocks"][c]
            frac = np.where(tot > 0, endo[c] / np.maximum(tot, 1e-12), 0.0)
            endo_frac[c] = frac.mean(axis=1)
        return ForecastSnapshot(t=t, horizons=self.horizons, model_id=self.predictor.name,
                                samples=samples, endogenous_fraction=endo_frac)

    def _children(self, s: SessionState, R: np.ndarray, samples, rng) -> None:
        """One level of fan-out: children spawned by this session's next turn, each with a first call."""
        hmax = self.horizons[-1]
        k = self.predictor.spawn(s, hmax, len(R), rng)
        if not k.any():
            return
        hz = np.asarray(self.horizons)[:, None]
        for i in np.nonzero(k)[0]:
            if not np.isfinite(R[i]) or R[i] > hmax:
                continue
            arrival = R[i] + rng.uniform(0.0, max(hmax - R[i], 1e-9), size=k[i])
            isl = self.exo.first_call_isl(s.cls, k[i], rng)
            due = arrival[None, :] <= hz
            samples["kv_blocks"][s.cls][:, i] += (due * np.ceil(isl / self.block_size)[None, :]).sum(axis=1)
            samples["prefill_tokens"][s.cls][:, i] += (due * isl[None, :]).sum(axis=1)


class SeriesForecaster:
    """History-only forecaster (B0/B1): one time series per (target, class, horizon).

    A truth observed at time t_obs for horizon h describes the window (t_obs, t_obs + h]; it is
    only usable as history once that window has fully elapsed, i.e. at ticks t >= t_obs + h.
    """

    def __init__(self, series: SeriesPredictor, horizons: list[float], n: int = 512, max_history: int = 500):
        self.series, self.horizons, self.n, self.max_history = series, list(horizons), n, max_history
        self._pending: dict[tuple[str, str, int], deque] = defaultdict(deque)  # (t_obs, value)
        self.history: dict[tuple[str, str, int], deque] = defaultdict(lambda: deque(maxlen=max_history))

    def observe(self, t: float, truth: dict) -> None:
        for tgt in TARGETS:
            for c in CLASSES:
                for k in range(len(self.horizons)):
                    self._pending[(tgt, c, k)].append((t, float(truth[tgt][c][k])))

    def _realize(self, t: float) -> None:
        for key, q in self._pending.items():
            h = self.horizons[key[2]]
            while q and q[0][0] + h <= t:
                self.history[key].append(q.popleft()[1])

    def forecast(self, t: float, states: list[SessionState], rng: np.random.Generator) -> ForecastSnapshot:
        self._realize(t)
        samples = _empty(self.horizons, self.n)
        for tgt in TARGETS:
            for c in CLASSES:
                for k in range(len(self.horizons)):
                    hist = np.asarray(self.history[(tgt, c, k)])
                    samples[tgt][c][k] = self.series.predict(hist, self.n, rng)
        endo = {c: np.zeros(len(self.horizons)) for c in CLASSES}
        return ForecastSnapshot(t=t, horizons=self.horizons, model_id=self.series.name, samples=samples,
                                endogenous_fraction=endo)
