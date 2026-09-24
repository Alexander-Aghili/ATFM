"""Closed-loop discrete-event simulator of agent sessions on a worker pool behind an admission arm."""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from atfm.bus import InMemoryBus
from atfm.schema.events import LlmDone, LlmFirstToken, LlmRequest, SessionStart, SpawnRequest, ToolEnd, ToolProgress, ToolStart

from .engine import EngineConfig, Request, Router, Worker
from .programs import Program


@dataclass
class SessionRun:
    program: Program
    turn: int = 0
    ctx: int = 0
    worker: Worker | None = None
    t_start: float = 0.0
    t_end: float | None = None
    deadline: float | None = None
    held_kv_block_s: float = 0.0
    evictions_caused: int = 0
    done: bool = False


@dataclass
class PendingCall:
    session: SessionRun
    turn_index: int
    t_arrival: float
    isl_total: int
    isl_new: int
    osl: int
    tier: int = 0
    index: float = 0.0
    release_not_before: float = 0.0
    hold_reason: str = ""
    t_release: float | None = None
    request_id: str = ""
    sched: tuple | None = None


class Simulator:
    def __init__(self, programs: list[Program], engines: list[EngineConfig], policy, *, harness_overhead_s: float = 0.2,
                 tick_s: float = 5.0, max_hold_s: float = 600.0, slo_ttft_s: float = 2.0, rng=None,
                 router_mode: str = "affinity", bus=None, block_size: int = 16):
        self.programs = programs
        self.workers = [Worker(f"w{i}", e) for i, e in enumerate(engines)]
        self.router = Router(self.workers, router_mode)
        self.policy = policy
        self.harness_overhead_s, self.tick_s, self.max_hold_s, self.slo_ttft_s = harness_overhead_s, tick_s, max_hold_s, slo_ttft_s
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.events = bus if bus is not None else InMemoryBus()
        self.block_size = block_size
        self.now = 0.0
        self._heap: list = []
        self._seq = itertools.count()
        self.sessions: dict[str, SessionRun] = {}
        self.proxy_queue: list[PendingCall] = []
        self.in_flight = 0
        self.caps = 0
        self.rows: list[dict] = []
        self.session_log: list[dict] = []
        self._pending: dict[str, PendingCall] = {}

    # ---- helpers
    def _push(self, t: float, kind: str, payload=None) -> None:
        heapq.heappush(self._heap, (t, next(self._seq), kind, payload))

    def _emit(self, e) -> None:
        self.events.publish(e)

    def _windowed(self) -> bool:
        return self.policy.window(self) is not None

    # ---- lifecycle
    def _start_session(self, p: Program, t: float) -> SessionRun:
        s = SessionRun(program=p, t_start=t, deadline=None if p.deadline_s is None else t + p.deadline_s)
        self.sessions[p.session_id] = s
        self._emit(SessionStart(t=t, session_id=p.session_id, tenant=p.tenant, cls=p.cls, parent_session_id=p.parent,
                                deadline=s.deadline))
        return s

    def _arrive(self, s: SessionRun, t: float) -> None:
        turn = s.program.turns[s.turn]
        call = PendingCall(session=s, turn_index=s.turn, t_arrival=t, isl_total=s.ctx + turn.isl_new,
                           isl_new=turn.isl_new, osl=turn.osl)
        call.tier, call.index = self.policy.tier_and_index(self, call)
        hold = self.policy.on_arrival(self, call)
        if hold is not None and hold > t:
            cap = t + self.max_hold_s
            if hold > cap:
                self.caps += 1
            call.release_not_before = min(hold, cap)
            call.hold_reason = getattr(self.policy, "last_hold_reason", "hold")
        if not self._windowed():
            self._release(call, t)
            return
        self.proxy_queue.append(call)
        self._drain_proxy(t)
        if call in self.proxy_queue and call.release_not_before > t:
            self._push(call.release_not_before, "release", None)

    def _drain_proxy(self, t: float) -> None:
        window = self.policy.window(self)
        if window is None:
            return
        while self.in_flight < window:
            eligible = [c for c in self.proxy_queue if c.release_not_before <= t]
            if not eligible:
                break
            best = max(eligible, key=lambda c: (c.tier, c.index, -c.t_arrival))
            self.proxy_queue.remove(best)
            self.in_flight += 1
            self._release(best, t)

    def _release(self, call: PendingCall, t: float) -> None:
        call.t_release = t
        s = call.session
        held = t - call.t_arrival
        if held > 0 and s.worker is not None:
            s.held_kv_block_s += s.worker.resident_blocks(s.program.session_id) * held
        rid = f"{s.program.session_id}:{call.turn_index}"
        call.request_id = rid
        self._pending[rid] = call
        req = Request(request_id=rid, session_id=s.program.session_id, cls=s.program.cls, isl_total=call.isl_total,
                      isl_new=call.isl_new, osl=call.osl, tier=call.tier, index=call.index, t_queued=t)
        w = self.router.pick(req)
        s.worker = w
        w.submit(req, t)
        self._emit(LlmRequest(t=t, session_id=s.program.session_id, turn_index=call.turn_index, request_id=rid,
                              isl=call.isl_total, predicted_osl=call.osl, hints={"strict_priority": call.tier}, held_s=held))
        self._schedule_worker(w, t)

    def _schedule_worker(self, w: Worker, t: float) -> None:
        for req, ts, tf, te, hit, recomputed, ev in w.schedule(t):
            call = self._pending[req.request_id]
            call.sched = (ts, tf, te, hit, recomputed, ev, w.worker_id)
            if ev and call.t_release is not None and call.t_release > call.t_arrival:
                call.session.evictions_caused += ev
            self._push(tf, "first_token", req.request_id)
            self._push(te, "worker_done", req.request_id)

    def _first_token(self, rid: str, t: float) -> None:
        call = self._pending[rid]
        self._emit(LlmFirstToken(t=t, session_id=call.session.program.session_id, request_id=rid))

    def _worker_done(self, rid: str, t: float) -> None:
        call = self._pending.pop(rid)
        s = call.session
        ts, tf, te, hit, recomputed, ev, wid = call.sched
        s.worker.complete(rid, t)
        if self._windowed():
            self.in_flight = max(0, self.in_flight - 1)
        s.ctx = call.isl_total + call.osl
        turn = s.program.turns[call.turn_index]
        self._emit(LlmDone(t=t, session_id=s.program.session_id, request_id=rid, osl=call.osl, worker_id=wid,
                           prefix_hit_tokens=hit))
        self.rows.append({
            "session_id": s.program.session_id, "class": s.program.cls, "tenant": s.program.tenant,
            "turn_index": call.turn_index, "t_arrival": call.t_arrival, "t_release": call.t_release,
            "t_queued_worker": call.t_release, "t_start": ts, "t_first_token": tf, "t_end": te, "worker_id": wid,
            "isl": call.isl_total, "osl": call.osl, "prefix_hit_tokens": hit, "recomputed_tokens": recomputed,
            "held_s": call.t_release - call.t_arrival, "hold_reason": call.hold_reason,
            "queue_proxy_s": call.t_release - call.t_arrival, "queue_worker_s": ts - call.t_release,
            "hold_kv_block_s": s.held_kv_block_s, "evictions_caused": s.evictions_caused, "deadline": s.deadline,
            "deadline_missed": bool(s.deadline is not None and t > s.deadline),
            "tool_name": turn.tool_name, "tool_duration": turn.tool_duration,
        })
        self._schedule_worker(s.worker, t)
        if self._windowed():
            self._drain_proxy(t)
        if turn.tool_name is None or turn.tool_duration is None:
            self._end_session(s, t)
            return
        call_id = f"{rid}:tool"
        self._emit(ToolStart(t=t, session_id=s.program.session_id, turn_index=call.turn_index, call_id=call_id,
                             tool_name=turn.tool_name, backend_id=turn.backend_id))
        for off, done, total in turn.progress:
            if off < turn.tool_duration:
                self._push(t + off, "tool_progress", (s.program.session_id, call_id, done, total))
        self._push(t + turn.tool_duration, "tool_end", (s.program.session_id, call_id, call.turn_index))

    def _tool_end(self, sid: str, call_id: str, turn_index: int, t: float) -> None:
        s = self.sessions[sid]
        self._emit(ToolEnd(t=t, session_id=sid, call_id=call_id, exit_status=0))
        for idx, child in s.program.spawn_at_turn:
            if idx == turn_index:
                cs = self._start_session(child, t)
                self._emit(SpawnRequest(t=t, parent_session_id=sid, child_session_id=child.session_id))
                self._arrive(cs, t)
        self.policy.on_tool_end(self, s, t)
        s.turn += 1
        self._push(t + self.harness_overhead_s, "arrive", sid)

    def _end_session(self, s: SessionRun, t: float) -> None:
        s.done, s.t_end = True, t
        if s.worker is not None:
            s.worker.evict_session(s.program.session_id)
            self._schedule_worker(s.worker, t)
        self.session_log.append({"session_id": s.program.session_id, "class": s.program.cls, "tenant": s.program.tenant,
                                 "t_start": s.t_start, "t_end": t, "deadline": s.deadline,
                                 "missed": bool(s.deadline is not None and t > s.deadline), "turns": s.turn + 1})

    # ---- main loop
    def run(self, until: float | None = None, max_idle_ticks: int = 100_000) -> pd.DataFrame:
        """Run to completion. `max_idle_ticks` consecutive ticks with no other event is treated as a stall
        (a policy holding every remaining call forever) and raises instead of spinning."""
        for p in self.programs:
            self._push(p.t_arrival, "start", p)
        self._push(0.0, "tick", None)
        idle_ticks = 0
        while self._heap:
            t, _, kind, payload = heapq.heappop(self._heap)
            if until is not None and t > until:
                break
            self.now = t
            if kind == "start":
                self._arrive(self._start_session(payload, t), t)
            elif kind == "arrive":
                self._arrive(self.sessions[payload], t)
            elif kind == "release":
                self._drain_proxy(t)
            elif kind == "first_token":
                self._first_token(payload, t)
            elif kind == "worker_done":
                self._worker_done(payload, t)
            elif kind == "tool_progress":
                sid, call_id, done, total = payload
                self._emit(ToolProgress(t=t, session_id=sid, call_id=call_id, completed=done, total=total, phase="run"))
            elif kind == "tool_end":
                sid, call_id, turn_index = payload
                self._tool_end(sid, call_id, turn_index, t)
            elif kind == "tick":
                self.policy.on_tick(self, t)
                self._drain_proxy(t)
                others = [k for _, _, k, _ in self._heap if k != "tick"]
                pending_work = any(not s.done for s in self.sessions.values()) or "start" in others
                idle_ticks = idle_ticks + 1 if not others else 0
                if idle_ticks > max_idle_ticks:
                    raise RuntimeError(f"simulation stalled: {len(self.proxy_queue)} calls waiting with no other events at t={t:.0f}")
                if pending_work:
                    self._push(t + self.tick_s, "tick", None)
        return pd.DataFrame(self.rows)
