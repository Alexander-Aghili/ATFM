from __future__ import annotations

import asyncio
import heapq
from collections import deque
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

    def __init__(self, window: int, clock=time.time, max_hold_s: float = 600.0, max_size: int | None = None,
                 release_order_max: int = 10000):
        self.window, self.clock, self.max_hold_s, self.max_size = window, clock, max_hold_s, max_size
        self.in_flight = 0
        self.pending: list[Entry] = []
        self.directives: dict[str, tuple[float, str, float | None]] = {}   # sid -> (release, reason, expires_at)
        self.caps = 0
        self.alarms = 0                 # overflow episodes (spec 10): queue over max_size forwards FCFS
        self.overflow = False
        self.release_order = deque(maxlen=release_order_max)
        self._timer: asyncio.TimerHandle | None = None

    def set_directive(self, session_id: str, release_not_before: float, reason: str, expires_at: float | None = None) -> None:
        self.directives[session_id] = (release_not_before, reason, expires_at)

    def directive_for(self, session_id: str) -> float | None:
        d = self.directives.get(session_id)
        if d is None:
            return None
        if d[2] is not None and self.clock() >= d[2]:
            del self.directives[session_id]     # expired: dropped on read (spec 10)
            return None
        return d[0]

    def submit(self, e: Entry) -> None:
        d = self.directive_for(e.session_id)
        if d is not None and d > e.t_arrival:
            cap = e.t_arrival + self.max_hold_s
            if d > cap:
                self.caps += 1
            e.not_before = min(d, cap)
        self.pending.append(e)
        if self.max_size is not None and len(self.pending) > self.max_size:
            if not self.overflow:
                self.alarms += 1
            self.overflow = True
            for p in self.pending:            # forward everything FCFS: holds are dropped under overflow
                p.not_before = 0.0
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
        available = max(0, self.window - self.in_flight)
        if available:
            eligible = [(i, e) for i, e in enumerate(self.pending) if e.not_before <= now]
            if self.overflow:
                selected = heapq.nsmallest(available, eligible, key=lambda item: item[1].t_arrival)
            else:
                selected = heapq.nlargest(available, eligible,
                                         key=lambda item: (item[1].tier, item[1].index, -item[1].t_arrival))
            released_indices = {i for i, _ in selected}
            self.pending = [e for i, e in enumerate(self.pending) if i not in released_indices]
            for _, best in selected:
                self.in_flight += 1
                best.t_release = now
                best.released.set()
                self.release_order.append(best.session_id)
        if self.overflow and (self.max_size is None or len(self.pending) <= self.max_size // 2):
            self.overflow = False
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
                "held": sum(1 for e in self.pending if e.not_before > now), "caps": self.caps,
                "alarms": self.alarms, "overflow": self.overflow}

    def tier_indices(self, tier: int) -> list[float]:
        return [e.index for e in self.pending if e.tier == tier]
