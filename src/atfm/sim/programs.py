"""Closed-loop session programs: what a session will do, without when (the simulator decides when)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from atfm.schema.trace import TraceTable, is_missing_scalar as _nan
from atfm.traces.synthetic import ClassSpec, ToolSpec, WorkloadSpec, _factor


@dataclass
class Turn:
    isl_new: int
    osl: int
    tool_name: str | None
    tool_duration: float | None
    backend_id: str = "local"
    progress: list[tuple[float, float, float | None]] = field(default_factory=list)
    think: bool = False
    reset: bool = False   # the context was replaced (compaction): full prefill of isl_new tokens


@dataclass
class Program:
    session_id: str
    cls: str
    tenant: str
    t_arrival: float
    turns: list[Turn]
    deadline_s: float | None = None
    parent: str | None = None
    spawn_at_turn: list[tuple[int, "Program"]] = field(default_factory=list)


def _progress_schedule(ts: ToolSpec, d: float) -> list[tuple[float, float, float | None]]:
    if ts.signal == "strong":
        out = []
        k = 1
        while k * ts.progress_every_s < d:
            out.append((k * ts.progress_every_s, 100.0 * k * ts.progress_every_s / d, 100.0))
            k += 1
        return out or [(0.5 * d, 50.0, 100.0)]
    if ts.signal == "weak":
        return [(0.9 * d, 90.0, 100.0)]
    return []


class _Gen:
    def __init__(self, spec: WorkloadSpec, rng: np.random.Generator):
        self.spec, self.rng, self.n = spec, rng, 0

    def program(self, cs: ClassSpec, t_arrival: float, parent: str | None = None, max_turns: int | None = None,
                t_abs_hint: float = 0.0) -> Program:
        rng, spec = self.rng, self.spec
        self.n += 1
        sid = f"{cs.cls[:2]}{self.n}"
        n_turns = 1 + rng.poisson(max(cs.turns_mean - 1, 0.0))
        if max_turns is not None:
            n_turns = min(n_turns, max_turns)
        w = np.array([t.weight for t in cs.tools], float)
        w /= w.sum()
        turns: list[Turn] = []
        spawns: list[tuple[int, Program]] = []
        for i in range(n_turns):
            isl_new = cs.isl0 if i == 0 else cs.isl_growth
            osl = 1 + rng.poisson(cs.osl_mean)
            if i == n_turns - 1:
                turns.append(Turn(isl_new, osl, None, None))
                break
            if cs.cls == "interactive" and cs.think_log_mu is not None and rng.random() < 0.5:
                d = float(rng.lognormal(cs.think_log_mu, cs.think_log_sigma or 0.5))
                turns.append(Turn(isl_new, osl, "__think__", d, "human", [], think=True))
                continue
            ts = cs.tools[rng.choice(len(cs.tools), p=w)]
            d = float(rng.lognormal(ts.log_mu, ts.log_sigma)) * _factor(spec, ts.backend_id, t_abs_hint)
            turns.append(Turn(isl_new, osl, ts.name, d, ts.backend_id, _progress_schedule(ts, d)))
            if rng.random() < ts.spawn_prob:
                spawns.append((i, self.program(cs, 0.0, parent=sid, max_turns=int(rng.integers(1, 4)), t_abs_hint=t_abs_hint)))
        return Program(session_id=sid, cls=cs.cls, tenant=f"t{self.n % spec.tenants}", t_arrival=t_arrival, turns=turns,
                       deadline_s=cs.deadline_s, parent=parent, spawn_at_turn=spawns)


def programs_from_spec(spec: WorkloadSpec, rng: np.random.Generator) -> list[Program]:
    gen = _Gen(spec, rng)
    out: list[Program] = []
    for cs in spec.classes:
        n = rng.poisson(cs.rate_per_hour * spec.duration_s / 3600.0)
        for t0 in np.sort(rng.uniform(0.0, spec.duration_s, size=n)):
            out.append(gen.program(cs, float(t0), t_abs_hint=float(t0)))
    out.sort(key=lambda p: p.t_arrival)
    return out


def programs_from_table(table: TraceTable, rate_per_hour: float | None, duration_s: float,
                        rng: np.random.Generator) -> list[Program]:
    progs: dict[str, Program] = {}
    children: dict[str, list[Program]] = {}
    t0 = float(table.df["t_request"].min()) if len(table.df) else 0.0   # rebase: replayed time starts at zero
    for sid, rows in table.session_records():
        turns: list[Turn] = []
        prev_ctx = 0
        for i, r in enumerate(rows):
            reset = bool(turns) and int(r["isl"]) < prev_ctx
            isl_new = int(r["isl"]) if (not turns or reset) else int(r["isl"]) - prev_ctx
            prev_ctx = int(r["isl"]) + int(r["osl"])
            tool = r["tool_name"] if not _nan(r["tool_name"]) else None
            if tool is None or _nan(r["t_tool_start"]) or _nan(r["t_tool_end"]):
                if i + 1 < len(rows):
                    # no recorded tool phase but the session called again: the gap to the next request is the
                    # think/tool time (a tool-less non-final turn would otherwise end the session in the simulator)
                    gap = max(0.0, float(rows[i + 1]["t_request"]) - float(r["t_last_token"]))
                    turns.append(Turn(isl_new, int(r["osl"]), "__gap__", gap, "local", [], think=True, reset=reset))
                else:
                    turns.append(Turn(isl_new, int(r["osl"]), None, None, reset=reset))
                continue
            d = max(0.0, float(r["t_tool_end"]) - float(r["t_tool_start"]))
            think = tool in ("__think__", "__gap__")
            prog = [(float(e["t"]) - float(r["t_tool_start"]), float(e["completed"]),
                     None if e.get("total") is None else float(e["total"])) for e in (r["progress_events"] or [])]
            backend = r["backend_id"] if not _nan(r["backend_id"]) else "local"
            turns.append(Turn(isl_new, int(r["osl"]), tool, d, backend, prog, think=think, reset=reset))
        parent = rows[0]["parent_session_id"] if not _nan(rows[0]["parent_session_id"]) else None
        p = Program(session_id=sid, cls=rows[0]["class"], tenant=rows[0]["tenant"], t_arrival=float(rows[0]["t_request"]) - t0,
                    turns=turns, parent=parent)
        progs[sid] = p
        if parent is not None:
            children.setdefault(parent, []).append(p)
    roots = [p for p in progs.values() if p.parent is None]
    for root in roots:
        for child in children.get(root.session_id, []):
            idx, t = 0, root.t_arrival
            for i, tr in enumerate(root.turns):
                if child.t_arrival >= t:
                    idx = i
                t += tr.tool_duration or 0.0
            root.spawn_at_turn.append((idx, child))
    if rate_per_hour:
        n = rng.poisson(rate_per_hour * duration_s / 3600.0)
        starts = np.sort(rng.uniform(0.0, duration_s, size=n))
        picks = rng.choice(len(roots), size=n, replace=True)
        return [_clone(roots[i], f"#{k}", float(s)) for k, (s, i) in enumerate(zip(starts, picks))]
    return sorted(roots, key=lambda p: p.t_arrival)


def _clone(p: Program, suffix: str, t_arrival: float) -> Program:
    """Copy a program (and its children, recursively) under a new id suffix so overlay copies never collide."""
    kids = [(idx, _clone(c, suffix, 0.0)) for idx, c in p.spawn_at_turn]
    parent = None if p.parent is None else f"{p.parent}{suffix}"
    return Program(session_id=f"{p.session_id}{suffix}", cls=p.cls, tenant=p.tenant, t_arrival=t_arrival, turns=p.turns,
                   deadline_s=p.deadline_s, parent=parent, spawn_at_turn=kids)
