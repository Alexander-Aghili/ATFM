from __future__ import annotations

from pydantic import BaseModel, Field


class ProxyConfig(BaseModel):
    upstream_url: str
    window: int = 8
    upstream_keepalive_connections: int | None = Field(default=None, ge=0, strict=True)
    w_interactive: float = 10.0
    w_background: float = 1.0
    beta: float = 0.0
    slack_threshold_s: float = 5.0
    prefill_tps: float = 20000.0
    decode_tps: float = 60.0
    default_osl: int = 256
    max_hold_s: float = 600.0
    board_timeout_s: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    prediction_limit: int = Field(default=4, ge=1, strict=True)
    board_url: str | None = None           # board service for per-request predictions (None: in-process predictor or defaults)
    max_queue_size: int | None = None      # hold-queue size beyond which the proxy forwards FCFS and alarms (spec 10)
    max_remembered_sessions: int = 10000   # bound on per-session state kept for keep-alive touches (LRU)
    events_path: str | None = None
    trace_path: str | None = None
