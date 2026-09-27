"""KV eviction, touch, and pin policies composed with forecast or oracle timing.

Admission remains rules-only so placement effects can be measured separately.
See docs/development/core.md for timing and finite-return summary contracts.
"""
from __future__ import annotations

import numpy as np

from atfm.board.resumption import resumption_quantiles
from atfm.proxy.config import ProxyConfig
from atfm.schema.forecast import ResumptionQuantiles
from atfm.schema.trace import TraceTable
from atfm.control import Residency, TouchController

from .forecast_arm import ForecastPolicy, next_call_of
from .policies import OraclePolicy


def oracle_etas(sim, now: float) -> dict[str, float]:
    """True absolute time of every session's next LLM call, from the heap plus the proxy and worker
    queues plus running requests (end of call plus tool)."""
    eta: dict[str, float] = {}
    for t, _, kind, payload in sorted(sim._heap, key=lambda x: (x[0], x[1])):
        if kind not in ("start", "arrive", "tool_end"):
            continue
        nc = next_call_of(sim, kind, payload)
        if nc is not None and nc[0] not in eta:
            eta[nc[0]] = float(t)
    for c in sim.proxy_queue:
        eta[c.session.program.session_id] = max(now, float(c.release_not_before))
    for w in sim.workers:
        for req in w.queue:
            eta[req.session_id] = now
        for req, _, t_end in w.running.values():
            s = sim.sessions.get(req.session_id)
            if s is None:
                continue
            turn = s.program.turns[min(s.turn, len(s.program.turns) - 1)]
            if s.turn + 1 >= len(s.program.turns) or turn.tool_name is None:
                continue
            eta[req.session_id] = float(t_end + (turn.tool_duration or 0.0) + sim.harness_overhead_s)
    return eta


def oracle_resumptions(sim, now: float) -> dict[str, ResumptionQuantiles]:
    """Represent known next-call times as deterministic relative quantiles."""
    return {sid: (t - now, t - now, t - now) for sid, t in oracle_etas(sim, now).items() if np.isfinite(t)}


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

    def __init__(self, window: int, cfg: ProxyConfig, size_aware: bool = False, bg_weight: float = 1.0, fresh: bool = False):
        super().__init__(window, cfg, hold=False)
        self._eta = {}
        self.size_aware, self.bg_weight, self.fresh = size_aware, bg_weight, fresh
        if size_aware:
            self.name = "oracle_kv_size"
        elif bg_weight != 1.0:
            self.name = "oracle_kv_cw"
        elif fresh:
            self.name = "oracle_kv_fresh"

    def kv_victims(self, sim, worker, candidates: list[str]) -> list[str]:
        if self.fresh:                                   # exact times at the eviction instant, not the last tick
            self._eta = oracle_etas(sim, sim.now)
        return super().kv_victims(sim, worker, candidates)

    def on_tick(self, sim, now: float) -> None:
        self._eta = oracle_etas(sim, now)


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


class _TouchMixin:
    """Keep alive imminent sessions whose cached context is near eviction.

    ``prefetch`` also admits evicted contexts; ``retry`` queues busy touches;
    ``yield_to_requests`` skips a touch when its worker has no free slot.
    """
    touch: TouchController
    touched: list[tuple[float, str]]
    prefetch: bool = False
    retry: bool = False
    yield_to_requests: bool = False
    skipped_busy: int = 0

    def e_tool_next(self, sim, call) -> float:
        return 0.0

    def _resumptions(self, sim, now: float) -> dict[str, ResumptionQuantiles]:
        raise NotImplementedError

    def _touch_tick(self, sim, now: float) -> None:
        resumptions = self._resumptions(sim, now)
        residency, frontier = {}, {}
        running_all = set()
        for w in sim.workers:
            running = w._running_sessions()
            running_all |= running
            fa = w.frontier_age(now)
            frontier[w.worker_id] = fa if fa is not None else 0.0
            for sid, blocks in w.resident.items():
                if sid not in running:
                    residency[sid] = Residency(worker_id=w.worker_id, blocks=blocks, last_used=w.last_used.get(sid, 0.0))
        if self.prefetch:
            for sid in resumptions:
                s = sim.sessions.get(sid)
                if sid in residency or sid in running_all or s is None or s.done:
                    continue
                w = s.worker if s.worker is not None else sim.workers[0]
                residency[sid] = Residency(worker_id=w.worker_id, blocks=s.ctx // w.cfg.block_size + 1, last_used=float("-inf"))
        for d in self.touch.plan(now, resumptions, residency, frontier):
            blocks = residency[d.session_id].blocks
            if self.yield_to_requests:
                s = sim.sessions.get(d.session_id)
                w = s.worker if s is not None and s.worker is not None else sim.workers[0]
                if w.cfg.touch_step_s > 0 and len(w.running) >= w.cfg.max_batch:
                    self.skipped_busy += 1
                    continue
            sim.touch(d.session_id, blocks)
            self.touched.append((now, d.session_id))


class OracleTouchPolicy(_TouchMixin, OraclePolicy):
    """Rules-only admission plus touches driven by the true next-call times."""

    name = "oracle_touch"

    def __init__(self, window: int, cfg: ProxyConfig, horizon_s: float = 30.0, age_s: float = 10.0,
                 budget_per_s: float = 1.0, tick_s: float = 5.0, prefetch: bool = False, retry: bool = False,
                 yield_to_requests: bool = False):
        super().__init__(window, cfg, hold=False)
        self.touch = TouchController(horizon_s=horizon_s, age_s=age_s, budget_per_s=budget_per_s, tick_s=tick_s)
        self.touched = []
        self.prefetch, self.retry, self.yield_to_requests = prefetch, retry, yield_to_requests

    def _resumptions(self, sim, now):
        return oracle_resumptions(sim, now)

    def on_tick(self, sim, now: float) -> None:
        self._touch_tick(sim, now)


class ForecastTouchPolicy(_TouchMixin, ForecastPolicy):
    """Rules-only admission plus touches driven by the predictor's resumption quantiles per session."""

    def __init__(self, window: int, cfg: ProxyConfig, predictor, train_table: TraceTable | None, horizons: list[float],
                 n: int = 64, horizon_s: float = 30.0, age_s: float = 10.0, budget_per_s: float = 1.0, tick_s: float = 5.0,
                 prefetch: bool = False, retry: bool = False, yield_to_requests: bool = False):
        super().__init__(window, cfg, predictor, train_table, horizons, n=n, hold=False)
        self.name = f"forecast_{predictor.name.split('_')[0]}_touch"
        self.touch = TouchController(horizon_s=horizon_s, age_s=age_s, budget_per_s=budget_per_s, tick_s=tick_s)
        self.touched = []
        self.prefetch, self.retry, self.yield_to_requests = prefetch, retry, yield_to_requests
        self._n = n

    def _resumptions(self, sim, now):
        return resumption_quantiles(self.predictor, self.registry.states(now), now, self._n, sim.rng)

    def on_tick(self, sim, now: float) -> None:
        ForecastPolicy.on_tick(self, sim, now)
        self._touch_tick(sim, now)


class RandomTouchPolicy(_TouchMixin, OraclePolicy):
    """Ablation for the touch mechanism: the same touch budget spent on uniformly random idle sessions.
    Separates "touching helps" from "touching the right sessions helps"."""

    name = "touch_random"

    def __init__(self, window: int, cfg: ProxyConfig, budget_per_s: float = 1.0, tick_s: float = 5.0, retry: bool = False):
        super().__init__(window, cfg, hold=False)
        self.touch = TouchController(horizon_s=float("inf"), age_s=float("inf"), budget_per_s=budget_per_s, tick_s=tick_s)
        self.touched = []
        self.retry = retry

    def _resumptions(self, sim, now):
        # every idle resident session is "imminent" with a random rank: the controller's sort is then a shuffle
        sids = [sid for w in sim.workers for sid in w.resident if sid not in w._running_sessions()]
        r = sim.rng.random(len(sids))
        return {sid: (float(x), float(x), float(x)) for sid, x in zip(sids, r)}

    def on_tick(self, sim, now: float) -> None:
        self._touch_tick(sim, now)


class _PinMixin:
    """Hard-pin placement, the LMCache form: every tick, pin (exclude from eviction) the resident idle
    sessions forecast to return within `horizon_s`, most imminent first, until `pin_budget_blocks` is
    reached; each pin lasts `horizon_s`. Admission is rules-only."""
    horizon_s: float = 30.0
    pin_budget_blocks: int | None = None
    pinned_log: list
    max_pinned_blocks_seen: int = 0

    def e_tool_next(self, sim, call) -> float:
        return 0.0

    def _resumptions(self, sim, now: float) -> dict[str, ResumptionQuantiles]:
        raise NotImplementedError

    def _pin_tick(self, sim, now: float) -> None:
        res = self._resumptions(sim, now)
        for w in sim.workers:
            w.expire_pins(now)
            running = w._running_sessions()
            cands = sorted(((q[1], sid) for sid, q in res.items() if sid in w.resident and sid not in running and q[1] <= self.horizon_s))
            used = w.pinned_blocks()
            for q50, sid in cands:
                if sid in w.pins:
                    continue
                blocks = w.resident.get(sid, 0)
                if self.pin_budget_blocks is not None and used + blocks > self.pin_budget_blocks:
                    continue
                if w.pin(sid, until=now + self.horizon_s):
                    used += blocks
                    sim.pins += 1
                    self.pinned_log.append((now, sid))
            self.max_pinned_blocks_seen = max(self.max_pinned_blocks_seen, used)


class OraclePinPolicy(_PinMixin, OraclePolicy):
    name = "oracle_pin"

    def __init__(self, window: int, cfg: ProxyConfig, horizon_s: float = 30.0, pin_budget_blocks: int | None = None):
        super().__init__(window, cfg, hold=False)
        self.horizon_s, self.pin_budget_blocks, self.pinned_log = horizon_s, pin_budget_blocks, []

    def _resumptions(self, sim, now):
        return oracle_resumptions(sim, now)

    def on_tick(self, sim, now: float) -> None:
        self._pin_tick(sim, now)


class ForecastPinPolicy(_PinMixin, ForecastPolicy):
    def __init__(self, window: int, cfg: ProxyConfig, predictor, train_table: TraceTable | None, horizons: list[float],
                 n: int = 64, horizon_s: float = 30.0, pin_budget_blocks: int | None = None):
        super().__init__(window, cfg, predictor, train_table, horizons, n=n, hold=False)
        self.name = f"forecast_{predictor.name.split('_')[0]}_pin"
        self.horizon_s, self.pin_budget_blocks, self.pinned_log, self._n = horizon_s, pin_budget_blocks, [], n

    def _resumptions(self, sim, now):
        return resumption_quantiles(self.predictor, self.registry.states(now), now, self._n, sim.rng)

    def on_tick(self, sim, now: float) -> None:
        ForecastPolicy.on_tick(self, sim, now)
        self._pin_tick(sim, now)


class RandomPinPolicy(_PinMixin, OraclePolicy):
    """Ablation: the same pin budget spent on random idle sessions."""

    name = "pin_random"

    def __init__(self, window: int, cfg: ProxyConfig, horizon_s: float = 30.0, pin_budget_blocks: int | None = None):
        super().__init__(window, cfg, hold=False)
        self.horizon_s, self.pin_budget_blocks, self.pinned_log = horizon_s, pin_budget_blocks, []

    def _resumptions(self, sim, now):
        sids = [sid for w in sim.workers for sid in w.resident if sid not in w._running_sessions()]
        r = sim.rng.random(len(sids)) * self.horizon_s
        return {sid: (float(x), float(x), float(x)) for sid, x in zip(sids, r)}

    def on_tick(self, sim, now: float) -> None:
        self._pin_tick(sim, now)
