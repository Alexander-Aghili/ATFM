"""Ground delay program planner (spec 6.2).

Every plan interval, slice the horizon into slots and, for each deferrable session in ration-by-schedule
order (earliest forecast resumption first), find the earliest slot at or after its own in which the
chance constraint P(interactive demand + assigned deferrable demand <= capacity) >= 1 - eps holds on the
forecast samples for every resource. The session's demand is then committed to that slot. A session whose
earliest feasible slot is beyond the hard cap is released at the cap; a tenant's imposed delay is bounded
by `tenant_max_delay`. Everything is evaluated on samples, never on moments (D5)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atfm.schema.forecast import ForecastSnapshot

from .directives import HoldDirective

RESOURCES = ("kv_blocks", "prefill_tokens")


@dataclass
class Deferrable:
    session_id: str
    tenant: str
    eta_s: float            # forecast resumption, seconds from now (q50)
    kv_blocks: int
    prefill_tokens: int
    cost_per_s: float = 1.0
    deadline: float | None = None


class GdpPlanner:
    def __init__(self, slot_s: float = 30.0, horizon_s: float = 900.0, eps: float = 0.1, max_hold_s: float = 600.0):
        self.slot_s, self.horizon_s, self.eps, self.max_hold_s = slot_s, horizon_s, eps, max_hold_s
        self.last_assignment: dict[int, list[str]] = {}
        self.max_imposed_delay: dict[str, float] = {}

    @property
    def n_slots(self) -> int:
        return max(1, int(round(self.horizon_s / self.slot_s)))

    def slot_demand(self, snap: ForecastSnapshot, resource: str) -> list[np.ndarray]:
        """Interactive demand *within* each slot, (n,) samples: differences of the cumulative forecast at
        the horizons closest to consecutive slot ends, clipped at zero."""
        hz = np.asarray(snap.horizons, float)
        cum = snap.samples[resource]["interactive"]
        n = cum.shape[1]
        out, prev = [], np.zeros(n)
        for k in range(self.n_slots):
            h = int(np.argmin(np.abs(hz - (k + 1) * self.slot_s)))
            cur = cum[h]
            out.append(np.maximum(cur - prev, 0.0))
            prev = np.maximum(cur, prev)
        return out

    def _feasible(self, inter: np.ndarray, committed: float, need: float, cap: float) -> bool:
        return float((inter + committed + need <= cap).mean()) >= 1.0 - self.eps

    def plan(self, now: float, snap: ForecastSnapshot, capacity: dict[str, float], deferrable: list[Deferrable],
             tenant_max_delay: dict[str, float] | None = None) -> list[HoldDirective]:
        demand = {r: self.slot_demand(snap, r) for r in RESOURCES}
        committed = {r: np.zeros(self.n_slots) for r in RESOURCES}
        self.last_assignment = {}
        self.max_imposed_delay = {}
        tenant_max_delay = tenant_max_delay or {}
        out = []
        for d in sorted(deferrable, key=lambda x: (x.eta_s, x.session_id)):
            start = max(0, int(d.eta_s // self.slot_s))
            chosen = None
            for k in range(start, self.n_slots):
                if k * self.slot_s > self.max_hold_s:
                    break
                if all(self._feasible(demand[r][k], committed[r][k], getattr(d, r), capacity.get(r, float("inf")))
                       for r in RESOURCES):
                    chosen = k
                    break
            if chosen is None:
                delay, reason = self.max_hold_s, "capped"
            else:
                delay, reason = chosen * self.slot_s, "gdp"
            bound = tenant_max_delay.get(d.tenant)
            if bound is not None and delay > bound:
                delay, reason, chosen = bound, "tenant_cap", None
            if chosen is not None:
                self.last_assignment.setdefault(chosen, []).append(d.session_id)
                for r in RESOURCES:
                    committed[r][chosen] += getattr(d, r)
            self.max_imposed_delay[d.tenant] = max(self.max_imposed_delay.get(d.tenant, 0.0), delay)
            out.append(HoldDirective(session_id=d.session_id, release_not_before=now + delay, reason=reason,
                                     tenant=d.tenant, expires_at=now + self.slot_s))
        return out
