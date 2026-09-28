from __future__ import annotations

import math

import numpy as np

from atfm.schema.trace import TraceTable, is_missing_scalar as _isnan

from .state import SessionState


def _last_llm_time(row: dict) -> float:
    for k in ("t_last_token", "t_first_token", "t_request"):
        if not _isnan(row[k]):
            return float(row[k])
    return float(row["t_request"])


class _Session:
    __slots__ = ("sid", "cls", "tenant", "parent", "rows", "t_start", "t_end")

    def __init__(self, sid: str, rows: list[dict]):
        self.sid = sid
        self.cls = rows[0]["class"]
        self.tenant = rows[0]["tenant"]
        parent = rows[0]["parent_session_id"]
        self.parent = None if _isnan(parent) else parent
        self.rows = rows
        self.t_start = float(self.rows[0]["t_request"])
        last = self.rows[-1]
        if last["tool_name"] is None or _isnan(last["t_tool_end"]):
            self.t_end = _last_llm_time(last)
        else:
            self.t_end = float(last["t_tool_end"])


class FleetReplayer:
    """Reconstructs in-flight session states and ground-truth demand from a trace table."""

    def __init__(self, table: TraceTable):
        self.table = table
        self.sessions = [_Session(sid, rows) for sid, rows in table.session_records()]
        df = table.df
        t_req = df["t_request"].to_numpy(float)
        order = np.argsort(t_req, kind="stable")
        self._t_req_sorted = t_req[order]
        self._isl_sorted = df["isl"].to_numpy(int)[order]
        self._cls_sorted = df["class"].to_numpy()[order]
        self._sid_sorted = df["session_id"].to_numpy()[order]
        self._start = {s.sid: s.t_start for s in self.sessions}
        self._end = {s.sid: s.t_end for s in self.sessions}

    def states_at(self, t: float) -> list[SessionState]:
        out = []
        for s in self.sessions:
            if t < s.t_start or t >= s.t_end:
                continue
            st = self._state(s, t)
            if st is not None:
                out.append(st)
        return out

    def _state(self, s: _Session, t: float) -> SessionState | None:
        history: list[tuple[str, float]] = []
        for i, r in enumerate(s.rows):
            if t < r["t_request"]:
                break
            t_llm_end, base = self._row_context(s, r)
            if t < t_llm_end:
                return SessionState(phase="llm_running", t_phase_start=float(r["t_request"]),
                                    tool_history=list(history), **base)
            has_tool = r["tool_name"] is not None and not _isnan(r["t_tool_start"])
            if not has_tool:
                return None
            ts, te = float(r["t_tool_start"]), float(r["t_tool_end"])
            if t < te:
                return self._tool_state(r, t, ts, history, base)
            history.append((r["tool_name"], max(0.0, te - ts)))
            pending = self._pending_state(s, i, r, t, te, history, base)
            if pending is not None:
                return pending
        return None

    def _row_context(self, s, r):
        ctx = int(r["isl"]) + int(r["osl"])
        t_llm_end = _last_llm_time(r)
        base = dict(session_id=s.sid, cls=s.cls, tenant=s.tenant, parent_session_id=s.parent,
                    turn_index=int(r["turn_index"]), ctx_tokens=ctx)
        return t_llm_end, base

    def _pending_state(self, s, i, r, t, te, history, base):
        nxt = s.rows[i + 1] if i + 1 < len(s.rows) else None
        if nxt is None or t < nxt["t_request"]:
            return SessionState(phase="llm_pending", tool_name=r["tool_name"], backend_id=r["backend_id"],
                                t_phase_start=te, tool_history=list(history), **base)
        return None

    def _tool_state(self, r, t, ts, history, base):
        prog = [e for e in (r["progress_events"] or []) if e["t"] <= t]
        data = [e for e in (r["data_events"] or []) if e["t"] <= t]
        return SessionState(phase="tool_running", tool_name=r["tool_name"], backend_id=r["backend_id"],
                            t_tool_start=min(ts, t), progress=prog, data=data, t_phase_start=ts,
                            tool_history=list(history), **base)

    def demand_truth(self, t: float, horizons: list[float], block_size: int = 16) -> dict:
        """KV blocks and prefill tokens required by sessions that (re)start an LLM call within each horizon.

        Each session counts once per horizon, with its first call after t: this is the
        "KV required on resumption" quantity from the plan, not the number of hops.
        """
        H = len(horizons)
        classes = ("interactive", "background")
        kv = {c: np.zeros(H) for c in classes}
        pf = {c: np.zeros(H) for c in classes}
        endo = {c: np.zeros(H) for c in classes}
        self._accumulate_demand(t, horizons, block_size, kv, pf, endo)
        return {"kv_blocks": kv, "prefill_tokens": pf, "endogenous_kv_blocks": endo}

    def _accumulate_demand(self, t, horizons, block_size, kv, pf, endo):
        lo = np.searchsorted(self._t_req_sorted, t, side="right")
        hi = np.searchsorted(self._t_req_sorted, t + max(horizons), side="right")
        seen: set = set()
        for j in range(lo, hi):
            sid = self._sid_sorted[j]
            if sid in seen:
                continue
            seen.add(sid)
            tr, isl, c = self._t_req_sorted[j], self._isl_sorted[j], self._cls_sorted[j]
            self._add_demand(sid, tr, isl, c, t, horizons, block_size, kv, pf, endo)

    def _add_demand(self, sid, tr, isl, c, t, horizons, block_size, kv, pf, endo):
        blocks = math.ceil(isl / block_size)
        active = self._start[sid] <= t < self._end[sid]
        for k, h in enumerate(horizons):
            if tr <= t + h:
                kv[c][k] += blocks
                pf[c][k] += isl
                if active:
                    endo[c][k] += blocks
