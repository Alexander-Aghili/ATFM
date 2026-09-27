"""Adapter from bus events (proxy + sidecar) to the canonical trace table."""
from __future__ import annotations

from collections import defaultdict

from atfm.schema.events import Event
from atfm.schema.trace import TraceRow, TraceTable


def events_to_trace_table(events: list[Event]) -> TraceTable:
    by_sess: dict[str, list] = defaultdict(list)
    meta: dict[str, dict] = {}
    for e in sorted(events, key=lambda e: e.t):
        if e.kind == "session.start":
            meta[e.session_id] = {"class": e.cls, "tenant": e.tenant, "parent": e.parent_session_id}
        elif e.kind == "spawn.request":
            meta.setdefault(e.child_session_id, {"class": "background", "tenant": "unknown", "parent": None})
            meta[e.child_session_id]["parent"] = e.parent_session_id
        elif e.kind != "worker.metrics":
            by_sess[e.session_id].append(e)
    rows: list[TraceRow] = []
    for sid, evs in by_sess.items():
        m = meta.get(sid, {"class": "background", "tenant": "unknown", "parent": None})
        calls = [e for e in evs if e.kind == "llm.request"]
        tools = _tools(evs)
        first_call_t = calls[0].t if calls else float("inf")
        # Tools that ran before the session's first LLM call (setup steps) or in a tool-only session get
        # synthetic one-token rows with negative turn indices so their phases are still measured.
        pre = [t for t in tools if t["t_start"] < first_call_t]
        for k, tl in enumerate(pre):
            rows.append(_row(sid, m, -(k + 1), tl["t_start"] - 0.01, tl["t_start"], tl["t_start"], 1, 0, tl))
        for i, c in enumerate(calls):
            first = next((e.t for e in evs if e.kind == "llm.first_token" and e.request_id == c.request_id), None)
            done = next((e for e in evs if e.kind == "llm.done" and e.request_id == c.request_id), None)
            t_last = done.t if done else first
            nxt = calls[i + 1].t if i + 1 < len(calls) else float("inf")
            between = [t for t in tools if c.t <= t["t_start"] < nxt]
            rows.append(_row(sid, m, c.turn_index, c.t, first, t_last, c.isl, done.osl if done else 0,
                             between[0] if between else None))
            for tl in between[1:]:  # extra tools in the same turn: one synthetic row each
                rows.append(_row(sid, m, c.turn_index, tl["t_start"] - 0.01, tl["t_start"], tl["t_start"], 1, 0, tl))
    return TraceTable.from_rows(rows)


def _tools(evs: list) -> list[dict]:
    out: dict[str, dict] = {}
    for e in evs:
        if e.kind == "tool.start":
            out[e.call_id] = {"t_start": e.t, "t_end": None, "name": e.tool_name, "backend": e.backend_id,
                              "hash": getattr(e, "args_hash", None), "progress": [], "data": [], "exit": None}
        elif getattr(e, "call_id", None) in out:
            tl = out[e.call_id]
            if e.kind == "tool.progress":
                tl["progress"].append({"t": e.t, "completed": e.completed, "total": e.total, "phase": e.phase})
            elif e.kind == "tool.data":
                tl["data"].append({"t": e.t, "metric": e.metric, "value": e.value})
            elif e.kind == "tool.end":
                tl["t_end"], tl["exit"] = e.t, e.exit_status
    return sorted(out.values(), key=lambda t: t["t_start"])


def _row(sid, m, turn, t_req, t_first, t_last, isl, osl, tl) -> TraceRow:
    kw = dict(session_id=sid, parent_session_id=m["parent"], cls=m["class"], tenant=m["tenant"], turn_index=turn,
              t_request=t_req, t_first_token=t_first, t_last_token=t_last, isl=int(isl), osl=int(osl), source="sidecar")
    if tl is not None:
        kw.update(tool_name=tl["name"], tool_args_hash=tl.get("hash"), backend_id=tl["backend"], t_tool_start=tl["t_start"],
                  t_tool_end=tl["t_end"],  # None when the tool never finished
                  tool_exit_status=tl["exit"], progress_events=tl["progress"], data_events=tl["data"])
    return TraceRow(**kw)
