"""Admission arms for the simulator. Each arm decides whether a call waits in the proxy queue, its tier
and index, and whether it is held; the engine below and the sessions above are identical for every arm."""
from __future__ import annotations

from typing import Protocol


class Policy(Protocol):
    name: str

    def window(self, sim) -> int | None: ...

    def tier_and_index(self, sim, call) -> tuple[int, float]: ...

    def on_arrival(self, sim, call) -> float | None: ...

    def on_tick(self, sim, now: float) -> None: ...

    def on_tool_end(self, sim, session, now: float) -> None: ...


class NativePolicy:
    """Arm 1: no proxy queue; requests go straight to the engine, which may order by class priority."""

    name = "native"

    def __init__(self, priority_by_class: bool = True):
        self.priority_by_class = priority_by_class
        self.last_hold_reason = ""

    def window(self, sim) -> int | None:
        return None

    def tier_and_index(self, sim, call) -> tuple[int, float]:
        return (1 if (self.priority_by_class and call.session.program.cls == "interactive") else 0), 0.0

    def on_arrival(self, sim, call) -> float | None:
        return None

    def on_tick(self, sim, now: float) -> None:
        return None

    def on_tool_end(self, sim, session, now: float) -> None:
        return None
