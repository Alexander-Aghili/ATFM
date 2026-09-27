"""Placement controllers.

TouchController is the deployable form of KV placement (L2 design note): no engine exposes pin or evict
for one session, but every prefix cache is an LRU, so refreshing a session's prefix with a minimal request
("touch") keeps it resident. Touch the sessions forecast to return soon whose blocks are near the
eviction frontier, most imminent first, within a touch budget. TierLogger is the pre-staging controller
of spec 6.3, which in v1 only logs what it would do."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .directives import TierDirective, TouchDirective

Quantiles = tuple[float, float, float]   # (q10, q50, q90) seconds until the session's next call


@dataclass
class Residency:
    worker_id: str
    blocks: int
    last_used: float


class TouchController:
    def __init__(self, horizon_s: float = 30.0, age_s: float = 10.0, budget_per_s: float = 1.0, tick_s: float = 5.0):
        self.horizon_s, self.age_s, self.budget_per_s, self.tick_s = horizon_s, age_s, budget_per_s, tick_s
        self._credit = 0.0
        self.issued = 0

    def plan(self, now: float, resumptions: dict[str, Quantiles], residency: dict[str, Residency],
             evict_frontier_age: dict[str, float]) -> list[TouchDirective]:
        """`evict_frontier_age[worker]` is the age (seconds since last use) of the blocks the worker will
        evict next; a session within `age_s` of it is at risk. Budget accrues per tick and is spent in
        whole touches."""
        self._credit += self.budget_per_s * self.tick_s
        allowed = int(math.floor(self._credit + 1e-9))
        if allowed <= 0:
            return []
        cands = []
        for sid, r in residency.items():
            q = resumptions.get(sid)
            frontier = evict_frontier_age.get(r.worker_id)
            if q is None or frontier is None or q[1] > self.horizon_s:
                continue
            if now - r.last_used >= frontier - self.age_s:
                cands.append((q[1], sid, r.worker_id))
        cands.sort()
        out = [TouchDirective(session_id=sid, worker_id=w, eta_q50=q50, expires_at=now + self.horizon_s)
               for q50, sid, w in cands[:allowed]]
        self._credit -= len(out)
        self.issued += len(out)
        return out


class TierLogger:
    """Per session, the deepest tier whose lead time is under the resumption q10 (spec 6.3). Logged only."""

    def __init__(self, tier_lead_s: dict[str, float]):
        self.tiers = sorted(tier_lead_s.items(), key=lambda kv: kv[1])   # fastest first
        self.log: list[tuple[float, list[TierDirective]]] = []

    def plan(self, now: float, resumptions: dict[str, Quantiles]) -> list[TierDirective]:
        out = []
        fastest = self.tiers[0][0]
        for sid, (q10, q50, q90) in resumptions.items():
            tier = fastest
            for name, lead in self.tiers:
                if lead < q10:
                    tier = name
            action = "pin" if tier == fastest else "demote"
            out.append(TierDirective(session_id=sid, action=action, tier=tier, eta_q10=q10, eta_q90=q90,
                                     expires_at=now + max(q50, 1.0)))
        self.log.append((now, out))
        return out
