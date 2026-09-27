from __future__ import annotations

from pydantic import BaseModel


class ProxyConfig(BaseModel):
    upstream_url: str
    window: int = 8
    w_interactive: float = 10.0
    w_background: float = 1.0
    beta: float = 0.0
    slack_threshold_s: float = 5.0
    prefill_tps: float = 20000.0
    decode_tps: float = 60.0
    default_osl: int = 256
    max_hold_s: float = 600.0
    board_timeout_s: float = 0.05
    max_queue_size: int | None = None      # hold-queue size beyond which the proxy forwards FCFS and alarms (spec 10)
    events_path: str | None = None
    trace_path: str | None = None
