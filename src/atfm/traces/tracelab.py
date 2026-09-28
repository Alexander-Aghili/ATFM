from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from atfm.schema.trace import TraceRow, TraceTable

THINK = "__think__"


def _ts(s: str | None) -> float | None:
    if s is None:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _round_to_row(rec: dict, turn_index: int, next_t_request: float | None) -> TraceRow | None:
    events = rec.get("timing_events") or []
    if not events:
        return None
    t_request = _ts(events[0]["timestamp"])
    assistant = [_ts(e["timestamp"]) for e in events if str(e.get("source", "")).startswith("assistant.")]
    t_first = min(assistant) if assistant else None
    t_last = max(assistant) if assistant else None
    tool_name, t_start, t_end, exit_status, backend, spawned = _round_tool(rec, t_request, t_last, next_t_request)
    return TraceRow(
        session_id=rec["session_id"], parent_session_id=None, cls="interactive",
        tenant=rec.get("user") or "unknown", turn_index=turn_index,
        t_request=t_request, t_first_token=t_first, t_last_token=t_last,
        isl=int(rec.get("input_tokens_total") or 0), osl=int(rec.get("output_tokens") or 0),
        prefix_hit_tokens=int(rec.get("prefix_tokens") or 0),
        tool_name=tool_name, t_tool_start=t_start, t_tool_end=t_end, tool_exit_status=exit_status,
        backend_id=backend, spawned_children=spawned, source="tracelab",
    )


def _round_tool(rec, t_request, t_last, next_t_request):
    tools = rec.get("tools") or []
    tool_name = t_start = t_end = exit_status = backend = None
    spawned = 0
    if tools:
        tool_name, t_start, t_end, exit_status, backend, spawned = _recorded_tool(tools, t_last)
    elif next_t_request is not None:
        tool_name = THINK
        t_start = t_last if t_last is not None else t_request
        t_end = max(next_t_request, t_start)
        backend = "human"
    return tool_name, t_start, t_end, exit_status, backend, spawned


def _recorded_tool(tools, t_last):
    crit = max(tools, key=lambda t: t.get("tool_wall_latency_ms") or 0)
    tool_name = crit["tool_name"]
    starts = [_ts(t["emitted_at"]) for t in tools if t.get("emitted_at")]
    ends = _tool_ends(tools)
    t_start = min(starts) if starts else t_last
    t_end = max(ends) if ends else None
    if t_start is not None and t_end is not None and t_end < t_start:
        t_end = t_start
    exit_status = 1 if any(t.get("is_error") for t in tools) else 0
    backend = "local"
    # Subagent calls (Agent/Task) run inside the parent's session in TraceLab; no separate child
    # sessions exist, so spawned_children stays 0 (otherwise the forecaster adds phantom children).
    spawned = 0
    return tool_name, t_start, t_end, exit_status, backend, spawned


def _tool_ends(tools):
    ends = []
    for t in tools:
        e = _ts(t.get("result_at"))
        if e is None and t.get("emitted_at") and t.get("tool_wall_latency_ms") is not None:
            e = _ts(t["emitted_at"]) + t["tool_wall_latency_ms"] / 1000.0
        if e is not None:
            ends.append(e)
    return ends


def rounds_to_rows(rounds: list[dict]) -> list[TraceRow]:
    rounds = sorted(rounds, key=lambda r: r["round_index"])
    rounds = [r for r in rounds if r.get("timing_events")]
    rows: list[TraceRow] = []
    for i, rec in enumerate(rounds):
        nxt = _ts(rounds[i + 1]["timing_events"][0]["timestamp"]) if i + 1 < len(rounds) else None
        row = _round_to_row(rec, len(rows), nxt)
        if row is not None:
            rows.append(row)
    return rows


def load_tracelab(path: str | Path, limit_sessions: int | None = None) -> TraceTable:
    by_session: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["session_id"]
            if limit_sessions is not None and sid not in by_session and len(by_session) >= limit_sessions:
                continue
            by_session[sid].append(rec)
    rows: list[TraceRow] = []
    for recs in by_session.values():
        rows.extend(rounds_to_rows(recs))
    return TraceTable.from_rows(rows)
