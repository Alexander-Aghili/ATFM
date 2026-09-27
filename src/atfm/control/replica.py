"""Replica floor (spec 6.4): the replica count needed so that q90 forecast demand at the scale-out lead
time fits capacity, proposed with AT_LEAST semantics. VirtualConnector stands in for the Dynamo Planner
PROPOSE plugin until it is wired to a real connector."""
from __future__ import annotations

import math

import numpy as np

from atfm.schema.forecast import ForecastSnapshot

from .directives import ReplicaDirective


class ReplicaFloor:
    def __init__(self, lead_time_s: float, blocks_per_replica: float, prefill_tps_per_replica: float, min_replicas: int = 1):
        if lead_time_s <= 0:
            raise ValueError("lead_time_s must be positive")
        if blocks_per_replica <= 0:
            raise ValueError("blocks_per_replica must be positive")
        self.lead_time_s, self.blocks_per_replica = lead_time_s, blocks_per_replica
        self.prefill_tps_per_replica, self.min_replicas = prefill_tps_per_replica, min_replicas

    def propose(self, now: float, snap: ForecastSnapshot) -> ReplicaDirective:
        h = int(np.argmin(np.abs(np.asarray(snap.horizons, float) - self.lead_time_s)))
        kv_q90 = float(np.quantile(snap.total("kv_blocks")[h], 0.9))
        pf_q90 = float(np.quantile(snap.total("prefill_tokens")[h], 0.9))
        need_kv = math.ceil(kv_q90 / self.blocks_per_replica) if self.blocks_per_replica > 0 else 0
        need_pf = math.ceil(pf_q90 / (self.prefill_tps_per_replica * self.lead_time_s)) if self.prefill_tps_per_replica > 0 else 0
        return ReplicaDirective(replicas_at_least=max(self.min_replicas, need_kv, need_pf), horizon_s=self.lead_time_s,
                                demand_q90_blocks=kv_q90, expires_at=now + self.lead_time_s)


class VirtualConnector:
    """Records proposals and applies AT_LEAST: never scales down on a proposal."""

    def __init__(self, current_replicas: int = 1):
        self.current_replicas = current_replicas
        self.history: list[dict] = []

    def propose_at_least(self, d: ReplicaDirective) -> int:
        self.history.append(d.model_dump())
        self.current_replicas = max(self.current_replicas, d.replicas_at_least)
        return self.current_replicas
