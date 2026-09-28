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
    t_enqueued: float | None = None
    prediction_s: float = 0.0


@dataclass
class _Waiting:
    entry: Entry
    ready: bool = False
    version: int = 0


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
        self._entries: dict[int, _Waiting] = {}
        self._entry_ids: dict[int, int] = {}
        self._sequence = 0
        self._ready: list[tuple] = []
        self._arrival: list[tuple] = []
        self._delayed: list[tuple] = []
        self._promotions: list[tuple] = []
        self.directives: dict[str, tuple[float, str, float | None]] = {}   # sid -> (release, reason, expires_at)
        self.caps = 0
        self.alarms = 0                 # overflow episodes (spec 10): queue over max_size forwards FCFS
        self.overflow = False
        self.release_order = deque(maxlen=release_order_max)
        self._timer: asyncio.TimerHandle | None = None

    @property
    def pending(self) -> list[Entry]:
        """Insertion-ordered diagnostic snapshot; scheduling fields are queue-owned."""
        return [record.entry for record in self._entries.values()]

    @pending.setter
    def pending(self, entries: list[Entry]) -> None:
        """Load an isolated benchmark backlog without releasing entries."""
        self._entries.clear()
        self._entry_ids.clear()
        for entry in entries:
            self._sequence += 1
            self._entries[self._sequence] = _Waiting(entry)
            self._entry_ids[id(entry)] = self._sequence
        self._rebuild(self.clock())

    @property
    def queued(self) -> int:
        return len(self._entries)

    def _ready_key(self, key: int, record: _Waiting) -> tuple:
        e = record.entry
        return (-e.tier, -e.index, e.t_arrival, key, record.version)

    def _make_ready(self, key: int, record: _Waiting) -> None:
        record.ready = True
        heapq.heappush(self._ready, self._ready_key(key, record))
        heapq.heappush(self._arrival, (record.entry.t_arrival, key))

    def _index(self, key: int, record: _Waiting, now: float) -> None:
        e = record.entry
        if e.not_before <= now:
            self._make_ready(key, record)
        else:
            heapq.heappush(self._delayed, (e.not_before, key))
        if e.tier == 1 and e.promote_at is not None:
            heapq.heappush(self._promotions, (e.promote_at, key))

    def _rebuild(self, now: float) -> None:
        self._ready, self._arrival, self._delayed, self._promotions = [], [], [], []
        for key, record in self._entries.items():
            e = record.entry
            record.ready = e.not_before <= now
            if record.ready:
                self._ready.append(self._ready_key(key, record))
                self._arrival.append((e.t_arrival, key))
            else:
                self._delayed.append((e.not_before, key))
            if e.tier == 1 and e.promote_at is not None:
                self._promotions.append((e.promote_at, key))
        for heap in (self._ready, self._arrival, self._delayed, self._promotions):
            heapq.heapify(heap)

    def _advance(self, now: float) -> None:
        while self._promotions and self._promotions[0][0] <= now:
            _, key = heapq.heappop(self._promotions)
            record = self._entries.get(key)
            if record is not None and record.entry.tier == 1:
                record.entry.tier = 2
                record.version += 1
                if record.ready:
                    heapq.heappush(self._ready, self._ready_key(key, record))
        while self._delayed and self._delayed[0][0] <= now:
            _, key = heapq.heappop(self._delayed)
            record = self._entries.get(key)
            if record is not None and not record.ready:
                self._make_ready(key, record)

    def _pop_ready(self) -> Entry | None:
        heap = self._arrival if self.overflow else self._ready
        while heap:
            item = heapq.heappop(heap)
            key = item[1] if self.overflow else item[3]
            record = self._entries.get(key)
            if record is None or not record.ready:
                continue
            if not self.overflow and item[4] != record.version:
                continue
            del self._entries[key]
            del self._entry_ids[id(record.entry)]
            return record.entry
        return None

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
        if id(e) in self._entry_ids or e.released.is_set() or e.done:
            raise ValueError("an entry can only be submitted once")
        self._sequence += 1
        self._entries[self._sequence] = record = _Waiting(e)
        self._entry_ids[id(e)] = self._sequence
        self._index(self._sequence, record, self.clock())
        if self.max_size is not None and self.queued > self.max_size:
            self._release_overflow()
        self.tick()

    def _release_overflow(self):
        if not self.overflow:
            self.alarms += 1
            self.overflow = True
            for record in self._entries.values():
                record.entry.not_before = 0.0
            self._rebuild(self.clock())
        else:
            while self._delayed:
                _, key = heapq.heappop(self._delayed)
                record = self._entries.get(key)
                if record is not None and not record.ready:
                    record.entry.not_before = 0.0
                    self._make_ready(key, record)

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
        key = self._entry_ids.pop(id(e), None)
        if key is not None:
            del self._entries[key]
            self.tick()
        elif e.released.is_set():
            self.complete(e)

    def tick(self) -> None:
        now = self.clock()
        self._advance(now)
        while self.in_flight < self.window:
            best = self._pop_ready()
            if best is None:
                break
            self.in_flight += 1
            best.t_release = now
            best.released.set()
            self.release_order.append(best.session_id)
        if self.overflow and (self.max_size is None or self.queued <= self.max_size // 2):
            self.overflow = False
        if sum(map(len, (self._ready, self._arrival, self._delayed, self._promotions))) > 8 * self.queued + 64:
            self._rebuild(now)
        self._arm_timer(now)

    def _arm_timer(self, now: float) -> None:
        for heap in (self._delayed, self._promotions):
            while heap and heap[0][1] not in self._entries:
                heapq.heappop(heap)
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        deadlines = [heap[0][0] for heap in (self._delayed, self._promotions) if heap]
        if not deadlines:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        delay = max(0.0, min(deadlines) - now)
        self._timer = loop.call_later(delay + 1e-3, self.tick)

    def stats(self) -> dict:
        now = self.clock()
        return {"queued": self.queued, "in_flight": self.in_flight,
                "held": sum(1 for e in (r.entry for r in self._entries.values()) if e.not_before > now), "caps": self.caps,
                "alarms": self.alarms, "overflow": self.overflow}

    def tier_indices(self, tier: int) -> list[float]:
        return [e.index for e in (r.entry for r in self._entries.values()) if e.tier == tier]
