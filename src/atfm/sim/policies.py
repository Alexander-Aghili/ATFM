"""Admission arms for the simulator. Each arm decides whether a call waits in the proxy queue, its tier
and index, and whether it is held; the engine below and the sessions above are identical for every arm."""
from __future__ import annotations

from typing import Protocol

from atfm.proxy.config import ProxyConfig
from atfm.proxy.index import CallMeta, compute_index, service_time, tier


class Policy(Protocol):
    name: str

    def window(self, sim) -> int | None: ...

    def tier_and_index(self, sim, call) -> tuple[int, float]: ...

    def on_arrival(self, sim, call) -> float | None: ...

    def on_tick(self, sim, now: float) -> None: ...

    def on_tool_end(self, sim, session, now: float) -> None: ...


class _PolicyHooks:
    """Optional simulation callbacks default to no action."""

    def on_arrival(self, sim, call) -> float | None:
        return None

    def on_tick(self, sim, now: float) -> None:
        return None

    def on_tool_end(self, sim, session, now: float) -> None:
        return None


class NativePolicy(_PolicyHooks):
    """Arm 1: no proxy queue; requests go straight to the engine, which may order by class priority."""

    name = "native"

    def __init__(self, priority_by_class: bool = True):
        self.priority_by_class = priority_by_class
        self.last_hold_reason = ""

    def window(self, sim) -> int | None:
        return None

    def tier_and_index(self, sim, call) -> tuple[int, float]:
        return (1 if (self.priority_by_class and call.session.program.cls == "interactive") else 0), 0.0


def _meta(call, now: float) -> CallMeta:
    p = call.session.program
    return CallMeta(session_id=p.session_id, cls=p.cls, tenant=p.tenant, deadline=call.session.deadline, parent=p.parent,
                    turn_index=call.turn_index, isl=call.isl_total, predicted_osl=call.osl, t_arrival=now)


class ProxyRulesPolicy(_PolicyHooks):
    """Arm 2: global window, class and deadline tiers, service-time index, no forecast, no holds."""

    name = "proxy_rules"

    def __init__(self, window: int, cfg: ProxyConfig):
        self._window, self.cfg = window, cfg
        self.last_hold_reason = ""

    def window(self, sim) -> int | None:
        return self._window

    def e_tool_next(self, sim, call) -> float:
        return 0.0

    def tier_and_index(self, sim, call) -> tuple[int, float]:
        m = _meta(call, sim.now)
        e_s = service_time(m, self.cfg)
        return tier(m, self.cfg, sim.now, e_s), compute_index(m, self.cfg, e_s, self.e_tool_next(sim, call))


def _event_session_cls(sim, kind: str, payload) -> str | None:
    """Class of the session an upcoming heap event belongs to (start: Program; arrive: id; tool_end: tuple)."""
    if kind == "start":
        return payload.cls
    if kind == "arrive":
        s = sim.sessions.get(payload)
        return s.program.cls if s else None
    if kind == "tool_end":
        s = sim.sessions.get(payload[0])
        return s.program.cls if s else None
    return None


class OraclePolicy(ProxyRulesPolicy):
    """Arm 5: knows the true duration of the tool each turn will launch, and (optionally) the true number
    of interactive calls arriving in the next slot. Upper bound for the forecast arms."""

    name = "oracle"

    def __init__(self, window: int, cfg: ProxyConfig, hold: bool = False, slot_s: float = 30.0):
        super().__init__(window, cfg)
        self.hold, self.slot_s = hold, slot_s

    def e_tool_next(self, sim, call) -> float:
        turn = call.session.program.turns[call.turn_index]
        return float(turn.tool_duration or 0.0)

    def on_arrival(self, sim, call) -> float | None:
        if not self.hold or call.session.program.cls != "background":
            return None
        now = sim.now
        coming = sum(1 for t, _, k, payload in sim._heap
                     if k in ("arrive", "start", "tool_end") and t <= now + self.slot_s
                     and _event_session_cls(sim, k, payload) == "interactive")
        slots = sum(max(0, w.cfg.max_batch - len(w.running)) for w in sim.workers)
        m = _meta(call, now)
        occupancy = coming * service_time(m, self.cfg) / self.slot_s   # expected busy slots over the slot
        if occupancy > slots:
            self.last_hold_reason = "oracle_surge"
            return now + self.slot_s
        return None


class WorkingSetPolicy(_PolicyHooks):
    """Arm 6 (ThunderAgent-style): pause background programs at tool boundaries while the resident working
    set is over budget; resume smallest-context-first once it falls under the low watermark. Reactive, no forecast."""

    name = "working_set"

    def __init__(self, budget_blocks: int, low_watermark: float = 0.8, window: int | None = None):
        self.budget, self.low, self._window = budget_blocks, low_watermark, window or 10**6
        self.paused = False
        self.last_hold_reason = ""

    def window(self, sim) -> int | None:
        return self._window

    def _working_set(self, sim) -> int:
        return sum(w.used_blocks() for w in sim.workers)

    def _update_pause(self, sim) -> None:
        ws = self._working_set(sim)
        if self.paused and ws < self.low * self.budget:
            self.paused = False
        elif not self.paused and ws >= self.budget:
            self.paused = True

    def tier_and_index(self, sim, call) -> tuple[int, float]:
        if call.session.program.cls == "interactive":
            return 1, 0.0
        return 0, -float(call.isl_total)

    @staticmethod
    def _fleet_idle(sim) -> bool:
        return all(len(w.running) == 0 for w in sim.workers)

    def on_arrival(self, sim, call) -> float | None:
        if call.session.program.cls == "interactive":
            return None
        self._update_pause(sim)
        if self.paused and not self._fleet_idle(sim):
            self.last_hold_reason = "working_set"
            # a paused program is offloaded: its KV leaves the working set (recompute is the cost of pausing)
            sim.evict_session_kv(call.session.program.session_id)
            return sim.now + sim.tick_s
        return None

    def on_tick(self, sim, now: float) -> None:
        self._update_pause(sim)
        waiting = [c for c in sim.proxy_queue if c.session.program.cls != "interactive"]
        if self.paused and not self._fleet_idle(sim):
            for c in waiting:
                c.release_not_before = max(c.release_not_before, now + sim.tick_s)
            return
        # not paused, or paused with an idle fleet (the paused sessions themselves hold the KV):
        # let the smallest-context call through so progress is always possible
        if self.paused and waiting:
            smallest = min(waiting, key=lambda c: c.isl_total)
            smallest.release_not_before = min(smallest.release_not_before, now)
            for c in waiting:
                if c is not smallest:
                    c.release_not_before = max(c.release_not_before, now + sim.tick_s)
        else:
            for c in waiting:
                c.release_not_before = min(c.release_not_before, now)
