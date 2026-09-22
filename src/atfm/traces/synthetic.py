from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from atfm.schema.trace import ProgressEvent, TraceRow, TraceTable


class ToolSpec(BaseModel):
    name: str
    weight: float
    log_mu: float
    log_sigma: float
    signal: Literal["strong", "weak", "none"] = "none"
    backend_id: str = "local"
    spawn_prob: float = 0.0
    progress_every_s: float = 10.0


class ClassSpec(BaseModel):
    cls: Literal["interactive", "background"]
    rate_per_hour: float
    turns_mean: float
    isl0: int
    isl_growth: int
    osl_mean: float
    tools: list[ToolSpec]
    think_log_mu: float | None = None
    think_log_sigma: float | None = None
    deadline_s: float | None = None


class Perturbation(BaseModel):
    backend_id: str
    t_start: float
    t_end: float
    factor: float


class WorkloadSpec(BaseModel):
    duration_s: float
    classes: list[ClassSpec]
    perturbations: list[Perturbation] = Field(default_factory=list)
    tenants: int = 4
    llm_prefill_tps: float = 20000.0
    llm_decode_tps: float = 60.0
    seed: int = 0


def _factor(spec: WorkloadSpec, backend: str, t: float) -> float:
    f = 1.0
    for p in spec.perturbations:
        if p.backend_id == backend and p.t_start <= t < p.t_end:
            f *= p.factor
    return f


class _Gen:
    def __init__(self, spec: WorkloadSpec):
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)
        self.rows: list[TraceRow] = []
        self.counter = 0

    def session(self, cs: ClassSpec, t0: float, parent: str | None = None, max_turns: int | None = None) -> None:
        rng, spec = self.rng, self.spec
        self.counter += 1
        sid = f"{cs.cls[:2]}{self.counter}"
        tenant = f"t{self.counter % spec.tenants}"
        n_turns = 1 + rng.poisson(max(cs.turns_mean - 1, 0.0))
        if max_turns is not None:
            n_turns = min(n_turns, max_turns)
        weights = np.array([t.weight for t in cs.tools], float)
        weights /= weights.sum()
        t = t0
        prev_osl = 0
        for turn in range(n_turns):
            isl = cs.isl0 + cs.isl_growth * turn + prev_osl
            osl = 1 + rng.poisson(cs.osl_mean)
            t_first = t + isl / spec.llm_prefill_tps
            t_last = t_first + osl / spec.llm_decode_tps
            row = dict(session_id=sid, parent_session_id=parent, cls=cs.cls, tenant=tenant, turn_index=turn,
                       t_request=t, t_first_token=t_first, t_last_token=t_last, isl=int(isl), osl=int(osl),
                       source="synthetic")
            if turn == n_turns - 1:
                self.rows.append(TraceRow(**row))
                break
            thinks = cs.cls == "interactive" and cs.think_log_mu is not None and rng.random() < 0.5
            if thinks:
                d = rng.lognormal(cs.think_log_mu, cs.think_log_sigma or 0.5)
                row.update(tool_name="__think__", backend_id="human", t_tool_start=t_last, t_tool_end=t_last + d,
                           tool_exit_status=0)
                self.rows.append(TraceRow(**row))
                t = t_last + d
                prev_osl = osl
                continue
            ts = cs.tools[rng.choice(len(cs.tools), p=weights)]
            base = rng.lognormal(ts.log_mu, ts.log_sigma)
            f = _factor(spec, ts.backend_id, t_last)
            d = base * f
            events: list[ProgressEvent] = []
            if ts.signal == "strong":
                k = 1
                while k * ts.progress_every_s < d:
                    events.append(ProgressEvent(t=t_last + k * ts.progress_every_s,
                                                completed=100.0 * k * ts.progress_every_s / d, total=100.0, phase="run"))
                    k += 1
                if not events:
                    events.append(ProgressEvent(t=t_last + d * 0.5, completed=50.0, total=100.0, phase="run"))
            elif ts.signal == "weak":
                events.append(ProgressEvent(t=t_last + 0.9 * d, completed=90.0, total=100.0, phase="end"))
            spawned = 1 if rng.random() < ts.spawn_prob else 0
            row.update(tool_name=ts.name, backend_id=ts.backend_id, t_tool_start=t_last, t_tool_end=t_last + d,
                       tool_exit_status=0, progress_events=events, perturbation_flag=f != 1.0,
                       spawned_children=spawned)
            self.rows.append(TraceRow(**row))
            if spawned:
                self.session(cs, t_last + d, parent=sid, max_turns=int(rng.integers(1, 4)))
            t = t_last + d + 0.2
            prev_osl = osl

    def run(self) -> TraceTable:
        for cs in self.spec.classes:
            n = self.rng.poisson(cs.rate_per_hour * self.spec.duration_s / 3600.0)
            for t0 in np.sort(self.rng.uniform(0.0, self.spec.duration_s, size=n)):
                self.session(cs, float(t0))
        return TraceTable.from_rows(self.rows)


def generate(spec: WorkloadSpec) -> TraceTable:
    return _Gen(spec).run()


def tail_share(table: TraceTable, threshold_s: float = 60.0) -> tuple[float, float]:
    """(share of tool calls over threshold, share of tool time over threshold), excluding think time."""
    df = table.df
    tools = df[df["tool_name"].notna() & (df["tool_name"] != "__think__")]
    d = (tools["t_tool_end"] - tools["t_tool_start"]).to_numpy(float)
    if len(d) == 0:
        return 0.0, 0.0
    long = d > threshold_s
    return float(long.mean()), float(d[long].sum() / d.sum())
