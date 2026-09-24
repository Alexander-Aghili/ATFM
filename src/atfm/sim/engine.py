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

    def used_blocks(self) -> int:
        return sum(self.resident.values())

    def free_blocks(self) -> int:
        return self.cfg.kv_blocks - self.used_blocks()

    def resident_blocks(self, session_id: str) -> int:
        return self.resident.get(session_id, 0)

    def submit(self, req: Request, now: float) -> None:
        req.t_queued = now
        self.queue.append(req)

    def _running_sessions(self) -> set[str]:
        return {r.session_id for r, _, _ in self.running.values()}

    def _make_room(self, needed: int, keep: str) -> int:
        """Evict idle LRU sessions (never `keep` or a running session) until `needed` blocks are free.
        Returns evictions made, or -1 when the room cannot be made."""
        evictions = 0
        running = self._running_sessions()
        while self.free_blocks() < needed:
            victim = next((s for s in self.resident if s not in running and s != keep), None)
            if victim is None:
                return -1
            del self.resident[victim]
            evictions += 1
        return evictions

    def schedule(self, now: float) -> list[tuple]:
        bs = self.cfg.block_size
        if self.cfg.priority:
            self.queue.sort(key=lambda r: (-r.tier, -r.index, r.t_queued))
        admitted, remaining = [], []
        for req in self.queue:
            if len(self.running) >= self.cfg.max_batch:
                remaining.append(req)
                continue
            have = self.resident.get(req.session_id, 0)
            needed_total = math.ceil((req.isl_total + req.osl) / bs)
            extra = max(0, needed_total - have)
            prefix_hit = min(have * bs, max(0, req.isl_total - req.isl_new)) if have else 0
            snapshot = OrderedDict(self.resident)
            ev = self._make_room(extra, keep=req.session_id) if extra > 0 else 0
            if ev < 0:
                self.resident = snapshot
                remaining.append(req)
                continue
            self.resident[req.session_id] = needed_total
            self.resident.move_to_end(req.session_id)
            recomputed = req.isl_total - prefix_hit
            t_first = now + recomputed / self.cfg.prefill_tps
            t_end = t_first + req.osl / self.cfg.decode_tps
            self.running[req.request_id] = (req, now, t_end)
            admitted.append((req, now, t_first, t_end, prefix_hit, recomputed, ev))
        self.queue = remaining
        return admitted

    def complete(self, request_id: str, now: float) -> None:
        req, _, _ = self.running.pop(request_id)
        if req.session_id in self.resident:
            self.resident.move_to_end(req.session_id)

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
