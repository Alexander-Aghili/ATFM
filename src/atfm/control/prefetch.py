"""Forecast-driven warming: start an L2 -> CPU prefetch early enough to finish before a session returns.

Lead time is a calibrated transfer model, overhead + context bytes / warm rate (stage C measured ~144 KiB per
token and ~1.4 GB/s on H100 and A100 hosts). A warm starts when the earliest likely return (q10) falls inside
lead + one control interval + margin, and is skipped when even a late return (q90) would beat it: a late warm
competes with the request's own load and was slower than recompute on fast GPUs. `trigger='q50'` waits until the
likely return is near (a context warmed right after its call is usually still resident). One warm per session turn,
or another after `rewarm_after_s`;
warmed-but-unused bytes count against a budget until the session returns or the directive expires.
"""
from __future__ import annotations

from atfm.board.state import SessionState
from atfm.control.directives import TierDirective
from atfm.schema.forecast import ResumptionQuantiles as Quantiles


TRIGGERS = {'q10': 0, 'q50': 1}   # which resumption quantile must fall inside lead + window


class PrefetchPlanner:
    def __init__(self, bytes_per_token: float, warm_bytes_per_s: float, overhead_s: float = 0.2,
                 interval_s: float = 5.0, margin_s: float = 1.0, budget_bytes: float = 16 * 2**30,
                 max_per_plan: int = 8, min_tokens: int = 16, trigger: str = 'q10', rewarm_after_s: float | None = None):
        if bytes_per_token <= 0 or warm_bytes_per_s <= 0 or budget_bytes < 0 or max_per_plan < 1:
            raise ValueError("prefetch calibration and budget must be positive")
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {sorted(TRIGGERS)}")
        self.trigger, self.rewarm_after_s = TRIGGERS[trigger], rewarm_after_s
        self.bytes_per_token, self.rate, self.overhead_s = bytes_per_token, warm_bytes_per_s, overhead_s
        self.window_s = interval_s + margin_s
        self.budget_bytes, self.max_per_plan, self.min_tokens = budget_bytes, max_per_plan, min_tokens
        self.in_flight: dict[str, tuple[float, float, int]] = {}   # sid -> (bytes, expires_at, turn)
        self.warmed: dict[str, tuple[int, float]] = {}             # sid -> (turn, time) of the last warm
        self.log: list[tuple[float, list[TierDirective]]] = []

    def lead_s(self, tokens: int) -> float:
        return self.overhead_s + tokens * self.bytes_per_token / self.rate

    def settle(self, session_id: str) -> None:
        self.in_flight.pop(session_id, None)

    def _release(self, now: float, states: dict[str, SessionState]) -> None:
        for sid, (_, expires_at, turn) in list(self.in_flight.items()):
            s = states.get(sid)
            returned = s is not None and (s.phase == 'llm_running' or s.turn_index != turn)
            if now >= expires_at or returned:
                self.settle(sid)

    def _recently_warmed(self, now: float, s: SessionState) -> bool:
        turn, t = self.warmed.get(s.session_id, (None, 0.0))
        return turn == s.turn_index and (self.rewarm_after_s is None or now - t < self.rewarm_after_s)

    def _candidate(self, now: float, s: SessionState | None, q: Quantiles) -> bool:
        if s is None or s.phase == 'llm_running' or s.ctx_tokens < self.min_tokens:
            return False
        if s.session_id in self.in_flight or self._recently_warmed(now, s):
            return False
        lead = self.lead_s(s.ctx_tokens)
        return q[self.trigger] <= lead + self.window_s and q[2] >= lead

    def plan(self, now: float, resumptions: dict[str, Quantiles], states: dict[str, SessionState]) -> list[TierDirective]:
        self._release(now, states)
        order = sorted((q[0], sid) for sid, q in resumptions.items() if self._candidate(now, states.get(sid), q))
        used = sum(b for b, _, _ in self.in_flight.values())
        out = []
        for _, sid in order:
            directive, size = self._directive(now, states[sid], resumptions[sid])
            if len(out) == self.max_per_plan or used + size > self.budget_bytes:
                continue
            used += size
            self.in_flight[sid] = (size, directive.expires_at, states[sid].turn_index)
            self.warmed[sid] = (states[sid].turn_index, now)
            out.append(directive)
        self.log.append((now, out))
        return out

    def _directive(self, now: float, s: SessionState, q: Quantiles) -> tuple[TierDirective, float]:
        size = s.ctx_tokens * self.bytes_per_token
        expires = now + max(q[2], self.lead_s(s.ctx_tokens)) + self.window_s
        return TierDirective(session_id=s.session_id, action='prefetch', tier='cpu', eta_q10=q[0], eta_q90=q[2],
                             expires_at=expires), size
