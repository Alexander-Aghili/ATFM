from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Phase = Literal["tool_running", "llm_pending", "llm_running", "ended"]


@dataclass
class SessionState:
    session_id: str
    cls: str
    tenant: str
    parent_session_id: str | None
    phase: Phase
    turn_index: int
    tool_name: str | None = None
    backend_id: str | None = None
    t_tool_start: float | None = None
    progress: list[dict] = field(default_factory=list)
    data: list[dict] = field(default_factory=list)
    ctx_tokens: int = 0
    tool_history: list[tuple[str, float]] = field(default_factory=list)
    t_phase_start: float = 0.0

    def elapsed(self, now: float) -> float:
        if self.phase == "tool_running" and self.t_tool_start is not None:
            return max(0.0, now - self.t_tool_start)
        return max(0.0, now - self.t_phase_start)
