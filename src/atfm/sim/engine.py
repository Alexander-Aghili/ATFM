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
        req.t_queued = now
        self.queue.append(req)

    def _running_sessions(self) -> set[str]:
        return {r.session_id for r, _, _ in self.running.values()}

    def _make_room(self, needed: int, keep: str) -> list[str] | None:
        """Evict idle LRU sessions (never `keep` or a running session) until `needed` blocks are free.
        Returns the evicted session ids, or None when the room cannot be made."""
        victims: list[str] = []
        running = self._running_sessions()
        order = None
        if self.victim_policy is not None:
            candidates = [s for s in self.resident if s not in running and s != keep]
            order = iter(list(self.victim_policy(candidates)))
        while self.free_blocks() < needed:
            if order is not None:
                victim = next((s for s in order if s in self.resident), None)
            else:
                victim = next((s for s in self.resident if s not in running and s != keep), None)
            if victim is None:
                return None
            del self.resident[victim]
            victims.append(victim)
        return victims

    def schedule(self, now: float) -> list[tuple]:
        bs = self.cfg.block_size
        if self.cfg.priority:
            self.queue.sort(key=lambda r: (-r.tier, -r.index, r.t_queued))
        admitted, remaining = [], []
        self.last_evictions = []
        blocked = False
        for req in self.queue:
            if blocked or len(self.running) >= self.cfg.max_batch:
                remaining.append(req)
                continue
            have = self.resident.get(req.session_id, 0)
            needed_total = math.ceil((req.isl_total + req.osl) / bs)
            extra = max(0, needed_total - have)
            prefix_hit = min(have * bs, max(0, req.isl_total - req.isl_new)) if have else 0
            snapshot = OrderedDict(self.resident)
            victims = self._make_room(extra, keep=req.session_id) if extra > 0 else []
            if victims is None:
                self.resident = snapshot
                remaining.append(req)
                if self.cfg.priority:
                    blocked = True  # no head-of-line skipping under priority: lower-priority calls must not overtake
                continue
            self.resident[req.session_id] = needed_total
            self.resident.move_to_end(req.session_id)
            self.last_used[req.session_id] = now
            self.last_evictions.extend((req.request_id, v) for v in victims)
            recomputed = req.isl_total - prefix_hit
            t_first = now + recomputed / self.cfg.prefill_tps
            t_end = t_first + req.osl / self.cfg.decode_tps
            self.running[req.request_id] = (req, now, t_end)
            admitted.append((req, now, t_first, t_end, prefix_hit, recomputed, len(victims)))
        self.queue = remaining
        return admitted

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

    def touch(self, session_id: str, blocks: int, now: float) -> tuple[str, list[str]]:
        """Keep-alive touch. Resident: refresh recency ("hit"). Not resident: speculative prefill, which makes
        room like an admission ("miss", victims). No room: ("fail", [])."""
        if session_id in self.resident:
            self.resident.move_to_end(session_id)
            self.last_used[session_id] = now
            return "hit", []
        snapshot = OrderedDict(self.resident)
        victims = self._make_room(blocks, keep=session_id)
        if victims is None:
            self.resident = snapshot
            return "fail", []
        self.resident[session_id] = blocks
        self.resident.move_to_end(session_id)
        self.last_used[session_id] = now
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
