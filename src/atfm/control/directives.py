"""Controller outputs (spec 3.5, 10). Every directive carries an expiry: on restart nothing is enforced
until a fresh snapshot produces new ones, and a stale directive is ignored by its consumer."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class _Directive(BaseModel):
    expires_at: float

    def expired(self, now: float) -> bool:
        return now >= self.expires_at


class HoldDirective(_Directive):
    """Do not forward this session's next call before `release_not_before` (capped by the proxy)."""
    kind: Literal["hold"] = "hold"
    session_id: str
    release_not_before: float
    reason: str = "gdp"
    tenant: str | None = None


class TierDirective(_Directive):
    """Placement the pre-staging controller would issue (logged only in v1, spec 6.3)."""
    kind: Literal["tier"] = "tier"
    session_id: str
    action: Literal["pin", "demote", "promote", "prefetch"]
    tier: str
    eta_q10: float
    eta_q90: float


class TouchDirective(_Directive):
    """Keep-alive touch: refresh this session's prefix on `worker_id` so it stays most-recently used."""
    kind: Literal["touch"] = "touch"
    session_id: str
    worker_id: str | None = None
    eta_q50: float | None = None


class ReplicaDirective(_Directive):
    """Planner PROPOSE with AT_LEAST semantics (spec 6.4)."""
    kind: Literal["replica"] = "replica"
    replicas_at_least: int
    horizon_s: float
    demand_q90_blocks: float = 0.0
