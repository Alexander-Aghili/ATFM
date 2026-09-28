"""Ground delay program planner (spec 6.2).

Every plan interval, slice the horizon into slots and, for each deferrable session in ration-by-schedule
order (earliest forecast resumption first), find the earliest slot at or after its own in which the
chance constraint P(interactive demand + assigned deferrable demand <= capacity) >= 1 - eps holds on the
forecast samples for every resource. The session's demand is then committed to that slot. A session whose
earliest feasible slot is beyond the hard cap is released at the cap; a tenant's imposed delay is bounded
by `tenant_max_delay`. Everything is evaluated on samples, never on moments (D5)."""
from __future__ import annotations

from dataclasses import dataclass
import math

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


class _SlotIndex:
    """Range minima of threshold + commitment, preserving feasibility arithmetic.

    Each resource minimum is a necessary condition only: different resources may
    attain their minima in different slots. Search can still visit every leaf.
    """

    def __init__(self, thresholds: dict[str, np.ndarray], committed=None):
        self.thresholds, self.committed = thresholds, committed
        self.values = None

    def _build(self) -> None:
        thresholds = self.thresholds
        count = len(thresholds[RESOURCES[0]])
        self.size = 1 << (count - 1).bit_length()
        self.values = np.full((len(RESOURCES), 2 * self.size), np.inf)
        for i, resource in enumerate(RESOURCES):
            self.values[i, self.size:self.size + count] = (thresholds[resource] if self.committed is None
                                                        else thresholds[resource] + self.committed[resource])
        width = self.size
        while width > 1:
            self.values[:, width // 2:width] = np.minimum(
                self.values[:, width:2 * width:2], self.values[:, width + 1:2 * width:2])
            width //= 2

    def update(self, slot: int, thresholds, committed) -> None:
        if self.values is None:
            return
        node = self.size + slot
        for i, resource in enumerate(RESOURCES):
            self.values[i, node] = thresholds[resource][slot] + committed[resource][slot]
        node //= 2
        while node:
            self.values[:, node] = np.minimum(self.values[:, 2 * node], self.values[:, 2 * node + 1])
            node //= 2

    def first(self, start: int, stop: int, need: tuple, capacity: tuple) -> int | None:
        if self.values is None:
            self._build()
        stack = [(1, 0, self.size)]
        while stack:
            node, left, right = stack.pop()
            if right <= start or left >= stop:
                continue
            if any(not (self.values[i, node] + amount <= capacity[i]) for i, amount in enumerate(need)):
                continue
            if right - left == 1:
                return left
            middle = (left + right) // 2
            stack.append((2 * node + 1, middle, right))
            stack.append((2 * node, left, middle))
        return None


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
        order = np.argsort(hz)
        hz = np.concatenate([[0.0], hz[order]])
        cum = np.vstack([np.zeros((1, snap.samples[resource]["interactive"].shape[1])), snap.samples[resource]["interactive"][order]])
        ends = (np.arange(self.n_slots) + 1) * self.slot_s
        # each sample path interpolated at the slot ends (constant beyond the last horizon), then differenced
        at = np.stack([np.interp(ends, hz, cum[:, j]) for j in range(cum.shape[1])], axis=1)   # (S, n)
        out, prev = [], np.zeros(cum.shape[1])
        for k in range(self.n_slots):
            out.append(np.maximum(at[k] - prev, 0.0))
            prev = np.maximum(at[k], prev)
        return out

    def _feasible(self, inter: np.ndarray, committed: float, need: float, cap: float) -> bool:
        return float((inter + committed + need <= cap).mean()) >= 1.0 - self.eps

    def _slot_thresholds(self, demand: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray] | None:
        """Exact empirical chance thresholds; no interpolated quantile approximation.

        The required count uses the same floating-point division as the original
        Boolean mean. Comparing the selected sample with the original addition
        order also avoids changing decisions at a capacity rounding boundary.
        Unusual/nonfinite inputs retain the general sample-scan implementation.
        """
        if not 0 <= self.eps < 1:
            return None
        thresholds = {}
        for resource, rows in demand.items():
            samples = np.asarray(rows)
            n = samples.shape[1]
            if n == 0 or not np.isfinite(samples).all():
                return None
            required = int(np.searchsorted(np.arange(n + 1) / n, 1.0 - self.eps))
            thresholds[resource] = np.partition(samples, required - 1, axis=1)[:, required - 1].copy()
        return thresholds

    def _first_slot(self, d: Deferrable, start: int, demand, thresholds, committed, capacity, index=None) -> int | None:
        if thresholds is None:
            for k in range(start, self.n_slots):
                if (k - start) * self.slot_s > self.max_hold_s:
                    break
                if all(self._feasible(demand[r][k], committed[r][k], getattr(d, r), capacity.get(r, float("inf")))
                       for r in RESOURCES):
                    return k
            return None
        if start >= self.n_slots or self.max_hold_s < 0:
            return None
        if all(thresholds[r][start] + committed[r][start] + getattr(d, r) <= capacity.get(r, float("inf"))
               for r in RESOURCES):
            return start
        low, high = start, self.n_slots
        while low < high:
            middle = (low + high) // 2
            if (middle - start) * self.slot_s > self.max_hold_s:
                high = middle
            else:
                low = middle + 1
        stop = low
        if index is not None:
            return index.first(start + 1, stop, (d.kv_blocks, d.prefill_tokens),
                               tuple(capacity.get(r, float("inf")) for r in RESOURCES))
        slots = np.arange(start + 1, stop)
        eligible = np.ones(len(slots), dtype=bool)
        for r in RESOURCES:
            eligible &= thresholds[r][slots] + committed[r][slots] + getattr(d, r) <= capacity.get(r, float("inf"))
        feasible = np.flatnonzero(eligible)
        return int(slots[feasible[0]]) if len(feasible) else None

    def plan(self, now: float, snap: ForecastSnapshot, capacity: dict[str, float], deferrable: list[Deferrable],
             tenant_max_delay: dict[str, float] | None = None) -> list[HoldDirective]:
        demand = {r: self.slot_demand(snap, r) for r in RESOURCES}
        thresholds = self._slot_thresholds(demand)
        committed = {r: np.zeros(self.n_slots) for r in RESOURCES}
        index = (_SlotIndex(thresholds, committed) if thresholds is not None and self.n_slots >= 64
                 and self.max_hold_s >= 64 * self.slot_s else None)
        self.last_assignment = {}
        self.max_imposed_delay = {}
        tenant_max_delay = tenant_max_delay or {}
        out = []
        for d in sorted(deferrable, key=lambda x: (x.eta_s, x.session_id)):
            start = max(0, int(d.eta_s // self.slot_s))          # the session's own resumption slot
            if index is not None and not (math.isfinite(d.kv_blocks) and math.isfinite(d.prefill_tokens)):
                index = None
            chosen = self._first_slot(d, start, demand, thresholds, committed, capacity, index)
            if chosen is None:
                imposed, reason, release = self.max_hold_s, "capped", now + d.eta_s + self.max_hold_s
            else:
                imposed, reason, release = (chosen - start) * self.slot_s, "gdp", now + chosen * self.slot_s
            bound = tenant_max_delay.get(d.tenant)
            if bound is not None and imposed > bound:
                imposed, reason = bound, "tenant_cap"
                chosen = min(start + int(bound // self.slot_s), self.n_slots - 1)
                release = now + chosen * self.slot_s
            if chosen is not None:                                  # committed where it is actually released
                self.last_assignment.setdefault(chosen, []).append(d.session_id)
                for r in RESOURCES:
                    committed[r][chosen] += getattr(d, r)
                if index is not None:
                    index.update(chosen, thresholds, committed)
            self.max_imposed_delay[d.tenant] = max(self.max_imposed_delay.get(d.tenant, 0.0), imposed)
            out.append(HoldDirective(session_id=d.session_id, release_not_before=release, reason=reason,
                                     tenant=d.tenant, expires_at=now + self.slot_s))
        return out
