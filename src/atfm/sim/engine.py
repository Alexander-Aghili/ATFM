"""Worker engine model: KV cache with LRU eviction and prefix reuse, batch limit, prefill/decode timing."""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal


@dataclass
class EngineConfig:
    kv_blocks: int
    max_batch: int
    prefill_tps: float
    decode_tps: float
    block_size: int = 16
    priority: bool = False
    touch_step_s: float = 0.0      # batch-slot time a keep-alive touch occupies (one decode step); 0 = free (old behaviour)


@dataclass
class Request:
    request_id: str
    session_id: str
    cls: str
    isl_total: int
    isl_new: int
    osl: int
    tier: int = 0
    index: float = 0.0
    t_queued: float = 0.0


class Worker:
    def __init__(self, worker_id: str, cfg: EngineConfig):
        self.worker_id, self.cfg = worker_id, cfg
        self.resident: OrderedDict[str, int] = OrderedDict()  # session -> blocks, LRU order (oldest first)
        self.running: dict[str, tuple[Request, float, float]] = {}
        self.queue: list[Request] = []
        self.last_evictions: list[tuple[str, str]] = []  # (admitted request_id, evicted session) from the last schedule()
        # optional placement policy: given the evictable idle sessions, return them in eviction order (first first);
        # None keeps LRU. This is where a forecast can act on KV residency instead of admission.
        self.victim_policy = None
        self.last_used: dict[str, float] = {}       # session -> last admission, completion or touch time
        self.touch_slot_s = 0.0                     # slot-seconds spent on touches
        self.pins: dict[str, float] = {}            # session -> pin expiry; pinned sessions are never evicted

    def used_blocks(self) -> int:
        return sum(self.resident.values())

    def free_blocks(self) -> int:
        """Blocks not resident at all (cache emptiness; low whenever the cache is warm)."""
        return self.cfg.kv_blocks - self.used_blocks()

    def capacity_free_blocks(self) -> int:
        """Blocks not held by running requests: idle resident blocks are evictable capacity."""
        running = self._running_sessions()
        return self.cfg.kv_blocks - sum(b for sid, b in self.resident.items() if sid in running)

    def resident_blocks(self, session_id: str) -> int:
        return self.resident.get(session_id, 0)

    def submit(self, req: Request, now: float) -> None:
        needed = math.ceil((req.isl_total + req.osl) / self.cfg.block_size)
        if needed > self.cfg.kv_blocks:
            raise ValueError(f"request {req.request_id} needs {needed} blocks but worker {self.worker_id} has {self.cfg.kv_blocks}: "
                             f"it could never be admitted (raise kv_blocks or shrink the programs' contexts)")
        req.t_queued = now
        self.queue.append(req)

    def _running_sessions(self) -> set[str]:
        return {r.session_id for r, _, _ in self.running.values()}

    def pin(self, session_id: str, until: float) -> bool:
        """Hard pin (what an LMCache pin executes): the session's blocks are excluded from eviction until
        `until`. Returns False when the session is not resident."""
        if session_id not in self.resident:
            return False
        self.pins[session_id] = max(until, self.pins.get(session_id, 0.0))
        return True

    def expire_pins(self, now: float) -> int:
        gone = [s for s, t in self.pins.items() if t <= now or s not in self.resident]
        for s in gone:
            del self.pins[s]
        return len(gone)

    def pinned_blocks(self) -> int:
        return sum(self.resident.get(s, 0) for s in self.pins)

    def _make_room(self, needed: int, keep: str) -> list[str] | None:
        """Evict idle LRU sessions (never `keep`, a running session or a pinned one) until `needed` blocks
        are free. Returns the evicted session ids, or None when the room cannot be made."""
        victims: list[str] = []
        running = self._running_sessions()
        protected = running | set(self.pins)
        candidates = [sid for sid in self.resident if sid not in protected and sid != keep]
        order = iter(candidates if self.victim_policy is None else list(self.victim_policy(candidates)))
        free = self.free_blocks()
        while free < needed:
            victim = next((sid for sid in order if sid in self.resident), None)
            if victim is None:
                return None
            free += self.resident.pop(victim)
            victims.append(victim)
        return victims

    def schedule(self, now: float) -> list[tuple]:
        if len(self.running) >= self.cfg.max_batch:
            self.last_evictions = []
            return []
        bs = self._prepare_schedule()
        return self._schedule_queue(now, bs)

    def _schedule_queue(self, now, bs):
        admitted, remaining = [], []
        blocked = False
        for req in self.queue:
            if blocked or len(self.running) >= self.cfg.max_batch:
                remaining.append(req)
                continue
            needed_total, prefix_hit, snapshot, victims = self._reserve(req, bs)
            if victims is None:
                self.resident = snapshot
                remaining.append(req)
                if self.cfg.priority:
                    blocked = True  # no head-of-line skipping under priority: lower-priority calls must not overtake
                continue
            admission = self._admit(req, needed_total, prefix_hit, victims, now)
            admitted.append(admission)
        self.queue = remaining
        return admitted

    def _prepare_schedule(self):
        self.last_evictions = []
        bs = self.cfg.block_size
        if self.cfg.priority:
            self.queue.sort(key=lambda r: (-r.tier, -r.index, r.t_queued))
        return bs

    def _admit(self, req, needed_total, prefix_hit, victims, now):
        self.resident[req.session_id] = needed_total
        self.resident.move_to_end(req.session_id)
        self.last_used[req.session_id] = now
        self.last_evictions.extend((req.request_id, v) for v in victims)
        recomputed = req.isl_total - prefix_hit
        t_first = now + recomputed / self.cfg.prefill_tps
        t_end = t_first + req.osl / self.cfg.decode_tps
        self.running[req.request_id] = (req, now, t_end)
        return (req, now, t_first, t_end, prefix_hit, recomputed, len(victims))

    def _reserve(self, req, bs):
        have = self.resident.get(req.session_id, 0)
        needed_total = math.ceil((req.isl_total + req.osl) / bs)
        extra = max(0, needed_total - have)
        prefix_hit = min(have * bs, max(0, req.isl_total - req.isl_new)) if have else 0
        snapshot = OrderedDict(self.resident)
        victims = self._make_room(extra, keep=req.session_id) if extra > 0 else []
        return needed_total, prefix_hit, snapshot, victims

    def complete(self, request_id: str, now: float) -> None:
        req, _, _ = self.running.pop(request_id)
        if req.session_id in self.resident:
            self.resident.move_to_end(req.session_id)
            self.last_used[req.session_id] = now

    def frontier_age(self, now: float) -> float | None:
        """Age of the idle resident session that would be evicted next (the LRU head), or None."""
        running = self._running_sessions()
        for sid in self.resident:
            if sid not in running:
                return now - self.last_used.get(sid, 0.0)
        return None

    def _occupy_touch_slot(self, session_id: str, now: float) -> None:
        """A touch is a real request for one step: it holds a batch slot until `expire_touches` frees it."""
        if self.cfg.touch_step_s <= 0:
            return
        rid = f"touch:{session_id}:{now:.3f}"
        req = Request(request_id=rid, session_id=session_id, cls="touch", isl_total=0, isl_new=0, osl=0, tier=0, index=0.0, t_queued=now)
        self.running[rid] = (req, now, now + self.cfg.touch_step_s)
        self.touch_slot_s += self.cfg.touch_step_s

    def busy_until(self, prefix: str) -> float | None:
        ends = [t_end for rid, (_, _, t_end) in self.running.items() if rid.startswith(prefix)]
        return max(ends) if ends else None

    def expire_touches(self, now: float) -> int:
        done = [rid for rid, (_, _, t_end) in self.running.items() if rid.startswith("touch:") and t_end <= now]
        for rid in done:
            del self.running[rid]
        return len(done)

    def touch(self, session_id: str, blocks: int, now: float) -> tuple[str, list[str]]:
        """Keep-alive touch. Resident: refresh recency ("hit"). Not resident: speculative prefill, which makes
        room like an admission ("miss", victims). No room or no free slot: ("fail", [])."""
        if self.cfg.touch_step_s > 0 and len(self.running) >= self.cfg.max_batch:
            return "fail", []
        if session_id in self.resident:
            self.resident.move_to_end(session_id)
            self.last_used[session_id] = now
            self._occupy_touch_slot(session_id, now)
            return "hit", []
        snapshot = OrderedDict(self.resident)
        victims = self._make_room(blocks, keep=session_id)
        if victims is None:
            self.resident = snapshot
            return "fail", []
        self.resident[session_id] = blocks
        self.resident.move_to_end(session_id)
        self.last_used[session_id] = now
        self._occupy_touch_slot(session_id, now)
        return "miss", victims

    def evict_session(self, session_id: str) -> int:
        return self.resident.pop(session_id, 0)


class Router:
    def __init__(self, workers: list[Worker], mode: Literal["affinity", "round_robin"] = "affinity"):
        self.workers, self.mode, self._rr = workers, mode, 0

    def pick(self, req: Request) -> Worker:
        if self.mode == "affinity":
            for w in self.workers:
                if req.session_id in w.resident:
                    return w
            return max(self.workers, key=lambda w: (w.free_blocks(), -len(w.queue)))
        w = self.workers[self._rr % len(self.workers)]
        self._rr += 1
        return w
