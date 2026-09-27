"""Live demand board: rebuilds session states from bus events and reuses the L0 forecaster."""
from __future__ import annotations

import numpy as np

from atfm.board.forecaster import SessionForecaster
from atfm.board.state import SessionState
from atfm.schema.events import Event
from atfm.schema.forecast import ForecastSnapshot


class SessionRegistry:
    """Mutable session state reconstructed from events, with inactivity expiry.

    Reads return the owned state objects, not copies. ``states(now)`` applies
    expiry; ``get`` and ``session_ids`` inspect without advancing time.
    """

    def __init__(self, expire_s: float = 7200.0):
        self.expire_s = expire_s
        self._s: dict[str, SessionState] = {}
        self._last: dict[str, float] = {}
        self._starts: list[tuple[float, str]] = []

    def _get(self, sid: str, t: float) -> SessionState:
        s = self._s.get(sid)
        if s is None:
            s = SessionState(session_id=sid, cls="background", tenant="unknown", parent_session_id=None,
                             phase="llm_pending", turn_index=0, t_phase_start=t)
            self._s[sid] = s
        self._last[sid] = max(self._last.get(sid, t), t)
        return s

    def apply(self, e: Event) -> None:
        k = e.kind
        if k == "session.start":
            s = self._get(e.session_id, e.t)
            s.cls, s.tenant, s.parent_session_id = e.cls, e.tenant, e.parent_session_id
            self._starts.append((e.t, e.cls))
        elif k == "llm.request":
            s = self._get(e.session_id, e.t)
            s.phase, s.t_phase_start, s.turn_index = "llm_running", e.t, e.turn_index
            s.ctx_tokens = int(e.isl) + int(e.predicted_osl or 0)
        elif k == "llm.done":
            s = self._get(e.session_id, e.t)
            s.phase, s.t_phase_start = "llm_pending", e.t
            if getattr(e, "worker_id", None):
                s.worker_id = e.worker_id
            s.t_last_done = e.t
        elif k == "tool.start":
            s = self._get(e.session_id, e.t)
            s.phase, s.tool_name, s.backend_id = "tool_running", e.tool_name, e.backend_id
            s.tool_args_hash = getattr(e, "args_hash", None)
            s.t_tool_start, s.t_phase_start = e.t, e.t
            s.progress, s.data = [], []
        elif k in ("tool.progress", "tool.data"):
            s = self._get(e.session_id, e.t)
            if s.phase != "tool_running":  # out-of-order or unannounced tool: attach to a placeholder
                s.phase, s.tool_name, s.backend_id = "tool_running", "unknown", "local"
                s.t_tool_start, s.t_phase_start = e.t, e.t
                s.progress, s.data = [], []
            if k == "tool.progress":
                s.progress.append({"t": e.t, "completed": e.completed, "total": e.total, "phase": e.phase})
            else:
                s.data.append({"t": e.t, "metric": e.metric, "value": e.value})
        elif k == "tool.end":
            s = self._get(e.session_id, e.t)
            if s.phase == "tool_running" and s.t_tool_start is not None:
                s.tool_history.append((s.tool_name or "unknown", max(0.0, e.t - s.t_tool_start)))
            s.phase, s.t_phase_start = "llm_pending", e.t
        elif k == "spawn.request":
            self._get(e.child_session_id, e.t).parent_session_id = e.parent_session_id

    def get(self, session_id: str) -> SessionState | None:
        """Look up a session without creating it or refreshing its expiry."""
        return self._s.get(session_id)

    def session_ids(self) -> tuple[str, ...]:
        """Snapshot the identifiers so callers can drop sessions while iterating."""
        return tuple(self._s)

    def drop(self, session_id: str) -> None:
        self._s.pop(session_id, None)
        self._last.pop(session_id, None)

    def states(self, now: float) -> list[SessionState]:
        for sid in [s for s, t in self._last.items() if now - t > self.expire_s]:
            self.drop(sid)
        return list(self._s.values())

    def new_starts_since(self, t: float) -> list[tuple[float, str]]:
        return [x for x in self._starts if x[0] >= t]


class LiveBoard:
    """Advance a session forecaster on wall-clock or simulated time.

    Each step feeds starts in [previous tick, now) to the exogenous model.
    Ticks must be nondecreasing and share the event timestamp time base.
    """

    def __init__(self, registry: SessionRegistry, forecaster: SessionForecaster, tick_s: float = 5.0,
                 prefill_tps: float = 20000.0, decode_tps: float = 60.0):
        self.registry, self.forecaster, self.tick_s = registry, forecaster, tick_s
        self.prefill_tps, self.decode_tps = prefill_tps, decode_tps
        self._starts_ptr = 0.0

    def step(self, now: float, rng: np.random.Generator) -> ForecastSnapshot:
        starts = [s for s in self.registry.new_starts_since(self._starts_ptr) if s[0] < now]
        self._starts_ptr = now
        self.forecaster.exo.update(now, starts)
        return self.forecaster.forecast(now, self.registry.states(now), rng)

    def expected_tool_next(self, session_id: str) -> float:
        dm = getattr(self.forecaster.predictor, "dm", None)
        if dm is None:
            return 0.0
        s = self.registry.get(session_id)
        tool = None
        if s is not None:
            tool = s.tool_name if s.phase == "tool_running" else (s.tool_history[-1][0] if s.tool_history else None)
        return dm.mean(tool)

    def expected_service(self, session_id: str, isl: int, osl: int) -> float:
        return isl / self.prefill_tps + osl / self.decode_tps
