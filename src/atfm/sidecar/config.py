"""Shared sidecar settings and bounded launch-gate waiting."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .gate import gate_allowed_at

DEFAULT_BACKENDS = {"pytest": "ci", "build": "ci", "install": "pkg", "clone": "git"}


@dataclass
class SidecarConfig:
    session_id: str
    tenant: str = "t0"
    cls: str = "background"
    bus: object = None
    gate_url: str | None = None
    deferrable: bool | None = None
    backend_map: dict[str, str] | None = None
    max_gate_wait_s: float = 600.0
    clock: Callable[[], float] = time.time
    turn_index: int = 0

    def __post_init__(self):
        if self.deferrable is None:
            self.deferrable = self.cls == "background"
        if self.backend_map is None:
            self.backend_map = dict(DEFAULT_BACKENDS)
        if self.bus is None:
            from atfm.bus import InMemoryBus
            self.bus = InMemoryBus()

    def backend_for(self, tool_name: str) -> str:
        return self.backend_map.get(tool_name, "local")

    def wait_for_gate(self) -> None:
        if self.deferrable:
            allowed = gate_allowed_at(self.gate_url, self.session_id, "tool")
            if allowed is not None:
                wait = min(max(0.0, allowed - self.clock()), self.max_gate_wait_s)
                if wait > 0:
                    time.sleep(wait)
