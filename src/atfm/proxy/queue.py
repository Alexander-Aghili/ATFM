from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class Entry:
    session_id: str
    tier: int
    index: float
    t_arrival: float
    not_before: float = 0.0
    promote_at: float | None = None   # when a waiting interactive call's slack runs out -> tier 2
    released: asyncio.Event = field(default_factory=asyncio.Event)
    t_release: float | None = None
    done: bool = False


class HoldQueue:
    """Global admission window with tier-then-index ordering and per-session hold directives.

    Ordering sticks at the layer that holds the backlog (spec 4.2): at most `window` requests are in
    flight; everything else waits here, ordered by (tier, index, arrival). A directive sets a
    not-before time for a session, capped at t_arrival + max_hold_s (D10).
    """

    def __init__(self, window: int, clock=time.time, max_hold_s: float = 600.0):
        self.window, self.clock, self.max_hold_s = window, clock, max_hold_s
        self.in_flight = 0
        self.pending: list[Entry] = []
        self.directives: dict[str, tuple[float, str]] = {}
        self.caps = 0
        self._timer: asyncio.TimerHandle | None = None

    def set_directive(self, session_id: str, release_not_before: float, reason: str) -> None:
        self.directives[session_id] = (release_not_before, reason)

    def directive_for(self, session_id: str) -> float | None:
        d = self.directives.get(session_id)
        return None if d is None else d[0]

    def submit(self, e: Entry) -> None:
        d = self.directive_for(e.session_id)
        if d is not None and d > e.t_arrival:
            cap = e.t_arrival + self.max_hold_s
            if d > cap:
                self.caps += 1
            e.not_before = min(d, cap)
        self.pending.append(e)
        self.tick()

    def complete(self, e: Entry | None = None) -> None:
        """Free the slot held by `e`; idempotent per entry (a second call is a no-op)."""
        if e is not None:
            if e.done or not e.released.is_set():
                return
            e.done = True
        self.in_flight = max(0, self.in_flight - 1)
        self.tick()

    def cancel(self, e: Entry) -> None:
        """Drop a still-waiting entry (client went away); a released one is freed via complete()."""
        if e in self.pending:
            self.pending.remove(e)
            self.tick()
        elif e.released.is_set():
            self.complete(e)

    def tick(self) -> None:
        now = self.clock()
        for e in self.pending:
            if e.tier == 1 and e.promote_at is not None and e.promote_at <= now:
                e.tier = 2
        while self.in_flight < self.window:
            eligible = [e for e in self.pending if e.not_before <= now]
            if not eligible:
                break
            best = max(eligible, key=lambda e: (e.tier, e.index, -e.t_arrival))
            self.pending.remove(best)
            self.in_flight += 1
            best.t_release = now
            best.released.set()
        self._arm_timer(now)

    def _arm_timer(self, now: float) -> None:
        future = [e.not_before for e in self.pending if e.not_before > now]
        if not future:
            return
        delay = max(0.0, min(future) - now)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._timer is not None:
            self._timer.cancel()
        self._timer = loop.call_later(delay + 1e-3, self.tick)

    def stats(self) -> dict:
        now = self.clock()
        return {"queued": len(self.pending), "in_flight": self.in_flight,
                "held": sum(1 for e in self.pending if e.not_before > now), "caps": self.caps}

    def tier_indices(self, tier: int) -> list[float]:
        return [e.index for e in self.pending if e.tier == tier]
