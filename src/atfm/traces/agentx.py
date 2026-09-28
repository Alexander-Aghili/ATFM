"""Adapter for the SemiAnalysis AgentX Claude Code corpus (semianalysisai/cc-traces-weka-062126).

One trace = one main-agent session. Main-agent requests (types 's' streaming, 'n' non-streaming)
become LLM calls; the gap after each response until the next request is the tool phase, labelled
`__gap__` because the corpus does not say whether the agent ran a tool or the user thought. A
sub-agent group that starts inside a gap makes that gap an `Agent` phase with `spawned_children`
set, and the group's inner requests become a child session `<id>/<agent_id>` with `parent_session_id`.
Prefix reuse is the number of leading 64-token hash blocks shared with the previous call.
"""
from __future__ import annotations

import json
from pathlib import Path

from atfm.schema.trace import TraceRow, TraceTable

GAP = "__gap__"


def _shared_prefix_tokens(prev: list | None, cur: list, block_size: int) -> int:
    if not prev or not cur:
        return 0
    n = 0
    for a, b in zip(prev, cur):
        if a != b:
            break
        n += 1
    return n * block_size


def _calls_to_rows(session_id: str, parent: str | None, calls: list[dict], groups: list[dict], block_size: int,
                   cls: str, tenant: str) -> list[TraceRow]:
    rows: list[TraceRow] = []
    prev_hash: list | None = None
    for i, r in enumerate(calls):
        t_req = float(r["t"])
        t_last = t_req + float(r.get("api_time") or 0.0)
        t_first = t_req + float(r["ttft"]) if r.get("ttft") is not None else None
        nxt = calls[i + 1] if i + 1 < len(calls) else None
        kw = dict(session_id=session_id, parent_session_id=parent, cls=cls, tenant=tenant, turn_index=i,
                  t_request=t_req, t_first_token=t_first, t_last_token=t_last, isl=int(r["in"]), osl=int(r["out"]),
                  prefix_hit_tokens=_shared_prefix_tokens(prev_hash, r.get("hash_ids") or [], block_size),
                  source="agentx")
        _call_gap(nxt, t_last, groups, kw)
        rows.append(TraceRow(**kw))
        prev_hash = r.get("hash_ids") or []
    return rows


def _call_gap(nxt, t_last, groups, kw):
    if nxt is not None:
        t_end = max(float(nxt["t"]), t_last)  # overlapping requests give a zero-length phase
        spawned = [g for g in groups if t_last <= float(g["t"]) < t_end]
        kw.update(tool_name="Agent" if spawned else GAP, backend_id="unknown", t_tool_start=t_last,
                  t_tool_end=t_end, tool_exit_status=0, spawned_children=len(spawned))


def trace_to_rows(trace: dict, cls: str = "interactive", tenant: str | None = None) -> list[TraceRow]:
    sid = trace["id"]
    tenant = tenant or f"agentx-{sid[:4]}"
    block_size = int(trace.get("block_size") or 64)
    reqs = trace.get("requests") or []
    main = [r for r in reqs if r.get("type") in ("s", "n")]
    groups = [r for r in reqs if r.get("type") == "subagent"]
    rows = _calls_to_rows(sid, None, main, groups, block_size, cls, tenant)
    for g in groups:
        child = f"{sid}/{g.get('agent_id') or 'subagent'}"
        inner = [r for r in (g.get("requests") or []) if r.get("type") in ("s", "n")]
        rows.extend(_calls_to_rows(child, sid, inner, [], block_size, cls, tenant))
    return rows


def load_agentx(path: str | Path, limit: int | None = None, cls: str = "interactive") -> TraceTable:
    rows: list[TraceRow] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.extend(trace_to_rows(json.loads(line), cls=cls))
    return TraceTable.from_rows(rows)
