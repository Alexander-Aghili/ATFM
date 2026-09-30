"""Live demand board: rebuilds session states from bus events and reuses the L0 forecaster."""
from __future__ import annotations

import numpy as np

from atfm.board.forecaster import SessionForecaster
from atfm.board.state import SessionState
from atfm.schema.events import Event
from atfm.schema.forecast import ForecastSnapshot
from atfm.traces.agentx import GAP


class SessionRegistry:
    """Mutable session state reconstructed from events, with inactivity expiry.

    Reads return the owned state objects, not copies. ``states(now)`` applies
    expiry; ``get`` and ``session_ids`` inspect without advancing time.
    """

    def __init__(self, expire_s: float = 7200.0, gap_after_done: bool = False):
        """``gap_after_done`` treats the time after each call as the corpus's ``__gap__`` tool phase, for
        sessions observed only at the proxy (no tool events), matching how AgentX traces label gaps."""
        self.expire_s, self.gap_after_done = expire_s, gap_after_done
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
        if e.kind == 'spawn.request':
            self._get(e.child_session_id, e.t).parent_session_id = e.parent_session_id
            return
        handler = _EVENT_HANDLERS.get(e.kind)
        if handler is not None:
            handler(self, self._get(e.session_id, e.t), e)

    def _session_start(self, s, e):
        s.cls, s.tenant, s.parent_session_id = e.cls, e.tenant, e.parent_session_id
        self._starts.append((e.t, e.cls))

    def _llm_request(self, s, e):
        if self.gap_after_done and s.phase == 'tool_running' and s.tool_name == GAP and s.t_tool_start is not None:
            s.tool_history.append((GAP, max(0.0, e.t - s.t_tool_start)))
        s.phase, s.t_phase_start, s.turn_index = 'llm_running', e.t, e.turn_index
        s.ctx_tokens = int(e.isl) + int(e.predicted_osl or 0)

    def _llm_done(self, s, e):
        s.phase, s.t_phase_start = 'llm_pending', e.t
        if getattr(e, 'worker_id', None):
            s.worker_id = e.worker_id
        s.t_last_done = e.t
        if self.gap_after_done:
            s.phase, s.tool_name, s.backend_id, s.t_tool_start = 'tool_running', GAP, 'local', e.t
            s.progress, s.data = [], []

    def _tool_start(self, s, e):
        s.phase, s.tool_name, s.backend_id = 'tool_running', e.tool_name, e.backend_id
        s.tool_args_hash = getattr(e, 'args_hash', None)
        s.t_tool_start, s.t_phase_start = e.t, e.t
        s.progress, s.data = [], []

    def _tool_update(self, s, e):
        if s.phase != 'tool_running':
            # Out-of-order progress belongs to a placeholder until its start arrives.
            s.phase, s.tool_name, s.backend_id = 'tool_running', 'unknown', 'local'
            s.t_tool_start, s.t_phase_start = e.t, e.t
            s.progress, s.data = [], []
        if e.kind == 'tool.progress':
            s.progress.append({'t': e.t, 'completed': e.completed, 'total': e.total, 'phase': e.phase})
        else:
            s.data.append({'t': e.t, 'metric': e.metric, 'value': e.value})

    def _tool_end(self, s, e):
        if s.phase == 'tool_running' and s.t_tool_start is not None:
            s.tool_history.append((s.tool_name or 'unknown', max(0.0, e.t - s.t_tool_start)))
        s.phase, s.t_phase_start = 'llm_pending', e.t

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

    def consume_starts(self, start: float, end: float) -> list[tuple[float, str]]:
        """Consume [start, end), discard late starts, retain future timestamps.

        The live board is the single consumer. Arrival order need not equal event
        time order; events older than its previous tick remain excluded.
        """
        ready = [x for x in self._starts if start <= x[0] < end]
        self._starts = [x for x in self._starts if x[0] >= end]
        return ready

    def new_starts_since(self, t: float) -> list[tuple[float, str]]:
        return [x for x in self._starts if x[0] >= t]


_EVENT_HANDLERS = {
    'session.start': SessionRegistry._session_start,
    'llm.request': SessionRegistry._llm_request,
    'llm.done': SessionRegistry._llm_done,
    'tool.start': SessionRegistry._tool_start,
    'tool.progress': SessionRegistry._tool_update,
    'tool.data': SessionRegistry._tool_update,
    'tool.end': SessionRegistry._tool_end,
}


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
        if now < self._starts_ptr:
            raise ValueError("board ticks must be nondecreasing")
        starts = self.registry.consume_starts(self._starts_ptr, now)
        self._starts_ptr = now
        self.forecaster.exo.update(now, starts)
        return self.forecaster.forecast(now, self.registry.states(now), rng)

    def expected_tool_next(self, session_id: str) -> float:
        dm = getattr(self.forecaster.predictor, "dm", None)
        if dm is None:
            return 0.0
        s = self.registry.get(session_id)
        return dm.mean(self.next_tool(s))

    @staticmethod
    def next_tool(state: SessionState | None) -> str | None:
        if state is None:
            return None
        if state.phase == "tool_running":
            return state.tool_name
        return state.tool_history[-1][0] if state.tool_history else None

    def expected_service(self, session_id: str, isl: int, osl: int) -> float:
        return isl / self.prefill_tps + osl / self.decode_tps
