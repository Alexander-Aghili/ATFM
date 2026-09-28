from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LoadConfig(BaseModel):
    """Finite, open-loop session arrivals with sequential agent turns per session."""

    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    sessions: int = Field(default=64, ge=1, le=100000)
    turns: int = Field(default=3, ge=1, le=100)
    arrival_window_s: float = Field(default=2.0, gt=0)
    pattern: Literal['staggered', 'burst'] = 'staggered'
    tool_mean_s: float = Field(default=.15, gt=0)
    burst_period_s: float = Field(default=.5, gt=0)
    interactive_fraction: float = Field(default=.25, ge=0, le=1)
    worker_slots: int = Field(default=8, ge=1)
    worker_service_s: float = Field(default=.05, gt=0)
    worker_fail_every: int = Field(default=0, ge=0)
    proxy_window: int = Field(default=16, ge=1)
    upstream_keepalive_connections: int | None = Field(default=None, ge=0, strict=True)
    prediction_limit: int = Field(default=4, ge=1)
    prediction_max_age_s: float | None = Field(default=None, gt=0)
    prediction_budget_s: float = Field(default=.05, gt=0)
    client_keepalive_connections: int = Field(default=0, ge=0)
    request_timeout_s: float = Field(default=20, gt=0)
    drain_timeout_s: float = Field(default=30, gt=0)
    profile_proxy: bool = False
    control_enabled: bool = True
    control_interval_s: float = Field(default=.5, gt=0)
    control_timeout_s: float = Field(default=2, gt=0)
    monitor_interval_s: float = Field(default=.2, gt=0)
    draws: int = Field(default=128, ge=1)
    slots: int = Field(default=10, ge=1)
    slot_s: float = Field(default=.25, gt=0)
    max_hold_s: float = Field(default=.5, ge=0)
    capacity_kv_blocks: float = Field(default=1024, gt=0)
    capacity_prefill_tokens: float = Field(default=16000, gt=0)
    seed: int = Field(default=7, ge=0)

    @model_validator(mode='after')
    def bounded_run(self):
        if self.sessions * self.turns > 1_000_000:
            raise ValueError('a run is limited to one million requests')
        return self
