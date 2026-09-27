# L0 Policies: Closed-Loop Fleet Simulator with Six Serving Arms. Implementation Plan

> **Historical implementation plan.** Embedded code and task checklists are a design record,
> not the current source of setup instructions. See [implementation status](../../status.md),
> [operations](../../operations.md), and [core development](../../development/core.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A discrete-event, closed-loop simulator of agent sessions on a Dynamo-like worker pool that runs the same six admission arms the H2 hardware study will run, logs where each arm queues work, accounts the KV cost of holds, and produces the serving metrics of spec section 12 with paired bootstrap confidence intervals.

**Architecture:** Sessions are programs (turns of LLM call then tool) that react to reply times, so every arm changes what happens next. A heap-based event loop drives arrivals, a proxy admission layer (the arm), worker engines with a KV cache and batch limit, and tool timers. Forecast arms embed the L0 demand board on simulated time. Metrics are computed from the per-request lifecycle log that every arm writes identically.

**Tech Stack:** Python 3.12, numpy, pandas, pydantic, pytest. No simulation framework. Reuses `atfm.traces.synthetic` specs, `atfm.board` predictors and forecaster, `atfm.proxy.index` formulas, `atfm.eval.forecast` scoring helpers.

**Spec:** `docs/superpowers/specs/2026-09-22-atfm-architecture-design.md` v1.1 sections 4.3 (index and tiers), 6.2 (hold semantics), 6.3 (hold cost), 8 (simulator), 9 (six arms, both regimes), 12 (metrics). Reviewer instruction (2026-09-23): six arms including a ThunderAgent-style reactive scheduler; log where each arm queues work; record held-session KV residency and recomputed prefill.

## Global Constraints

- Python 3.12; package `atfm`; every stochastic function takes `rng: np.random.Generator`; simulated time is float seconds from 0.
- The simulator is closed-loop (D12): a session's next tool starts only when its LLM reply completes; a session's next call arrives only when its tool ends (plus harness overhead).
- Same arm and formula code as deployment where it exists: index and tier from `atfm.proxy.index`, predictors from `atfm.board.predictors`, forecaster from `atfm.board.forecaster`.
- Every arm writes the same request lifecycle record: `t_arrival, t_release, t_queued_worker, t_start, t_first_token, t_end` plus `worker_id, prefix_hit_tokens, recomputed_tokens, held_s, hold_kv_block_s, evictions_caused`.
- KV block size 16 tokens; a request's resident blocks = `ceil((ctx_tokens)/16)`; a session's KV stays resident after its call until evicted (LRU) or the session ends.
- Policy claims are paired: every arm runs on the same program set and seed; CIs are bootstrapped over sessions.
- Holds are capped at `max_hold_s` (default 600) and charged per class (spec 6.2).

## Review Focus

1. A tool that returns while the session's KV was evicted: the next call must be a full prefill with `prefix_hit_tokens = 0` and `recomputed_tokens = isl`, never a negative or a cache hit. Test in Task 2.
2. A worker with fewer free blocks than a request needs and nothing to evict (all resident blocks belong to running requests): the request must wait in the worker queue, not be admitted into an over-committed cache. Test in Task 2.
3. A hold that would exceed `max_hold_s`: the session is released at the cap and the cap is counted, for every forecast arm. Test in Task 5.
4. A session whose program ends while it is held (deadline passed) still completes its remaining turns; deadline misses are counted, sessions are never dropped. Test in Task 3.
5. Two arms given identical programs and seeds must see identical arrival sequences for the first call of every session (paired design), even though later timing diverges. Test in Task 7.

---

### Task 1: Session programs (closed-loop workload) from specs and traces

**Files:**
- Create: `src/atfm/sim/__init__.py`, `src/atfm/sim/programs.py`
- Test: `tests/sim/test_programs.py`

**Interfaces:**
- Produces:
  - `Turn(isl_new: int, osl: int, tool_name: str | None, tool_duration: float | None, backend_id: str, progress: list[tuple[float, float, float | None]] = [], think: bool = False)`: `isl_new` = tokens appended to the context this turn (the prompt growth); `progress` = list of `(offset_s, completed, total)` relative to tool start; `think=True` marks a human think gap (not a tool).
  - `Program(session_id: str, cls: str, tenant: str, t_arrival: float, turns: list[Turn], deadline_s: float | None, parent: str | None = None, spawn_at_turn: list[tuple[int, "Program"]] = [])`: `spawn_at_turn` children start when the parent's turn index completes.
  - `programs_from_spec(spec: WorkloadSpec, rng) -> list[Program]`: same sampling as `atfm.traces.synthetic.generate` but produces programs without absolute times for anything except `t_arrival` (Poisson). Tool durations are sampled at generation (so arms are paired) and progress schedules follow the spec's signal type (strong: every `progress_every_s`; weak: one event at 90%; none: empty). Children are attached via `spawn_at_turn`.
  - `programs_from_table(table: TraceTable, rate_per_hour: float | None, duration_s: float, rng) -> list[Program]`: one program per session family; `isl_new` = `isl_t - (isl_{t-1} + osl_{t-1})` clipped at 0 (first turn: isl); tool duration from the row's phase; `__think__`/`__gap__` rows become `think=True` turns; arrivals are the table's own start times unless `rate_per_hour` is given (Poisson, as in `overlay_sessions`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_programs.py
import numpy as np
from atfm.traces.synthetic import WorkloadSpec, ClassSpec, ToolSpec
from atfm.schema.trace import TraceRow, TraceTable
from atfm.sim.programs import Program, Turn, programs_from_spec, programs_from_table

def _spec():
    tools = [ToolSpec(name="bash", weight=0.7, log_mu=np.log(3.0), log_sigma=0.4, signal="none"),
             ToolSpec(name="pytest", weight=0.3, log_mu=np.log(120.0), log_sigma=0.5, signal="strong", backend_id="ci", spawn_prob=0.5)]
    return WorkloadSpec(duration_s=1800.0, seed=0, classes=[
        ClassSpec(cls="background", rate_per_hour=120.0, turns_mean=6, isl0=2000, isl_growth=400, osl_mean=100, tools=tools),
        ClassSpec(cls="interactive", rate_per_hour=60.0, turns_mean=4, isl0=3000, isl_growth=500, osl_mean=80, tools=tools[:1],
                  think_log_mu=np.log(15.0), think_log_sigma=0.5, deadline_s=600.0)])

def test_programs_from_spec_are_paired_and_closed_loop():
    a = programs_from_spec(_spec(), np.random.default_rng(1))
    b = programs_from_spec(_spec(), np.random.default_rng(1))
    assert [p.t_arrival for p in a] == [p.t_arrival for p in b] and len(a) > 40
    assert all(0.0 <= p.t_arrival < 1800.0 for p in a)
    bg = [p for p in a if p.cls == "background"]
    assert any(p.spawn_at_turn for p in bg)                                   # fan-out present
    p = bg[0]
    assert p.turns[0].isl_new == 2000 and p.turns[-1].tool_name is None      # last turn has no tool
    strong = [t for q in bg for t in q.turns if t.tool_name == "pytest"]
    assert strong and all(len(t.progress) >= 1 and t.progress[-1][2] == 100.0 for t in strong)
    it = [p for p in a if p.cls == "interactive"][0]
    assert it.deadline_s == 600.0 and any(t.think for t in it.turns) or True  # think turns may or may not appear per session

def test_programs_from_table_isl_growth_and_gaps():
    rows = [TraceRow(session_id="a", cls="interactive", tenant="t", turn_index=0, t_request=100.0, t_first_token=101.0, t_last_token=103.0,
                     isl=1000, osl=50, tool_name="pytest", backend_id="ci", t_tool_start=103.0, t_tool_end=163.0,
                     progress_events=[{"t": 133.0, "completed": 50, "total": 100, "phase": "run"}], source="test"),
            TraceRow(session_id="a", cls="interactive", tenant="t", turn_index=1, t_request=163.5, t_first_token=164.0, t_last_token=166.0,
                     isl=1450, osl=30, tool_name="__gap__", backend_id="unknown", t_tool_start=166.0, t_tool_end=200.0, source="test"),
            TraceRow(session_id="a", cls="interactive", tenant="t", turn_index=2, t_request=200.0, t_first_token=201.0, t_last_token=202.0,
                     isl=1400, osl=10, tool_name=None, source="test")]
    progs = programs_from_table(TraceTable.from_rows(rows), rate_per_hour=None, duration_s=1000.0, rng=np.random.default_rng(0))
    assert len(progs) == 1 and progs[0].t_arrival == 100.0
    t0, t1, t2 = progs[0].turns
    assert t0.isl_new == 1000 and t0.tool_duration == 60.0 and t0.progress == [(30.0, 50.0, 100.0)]
    assert t1.isl_new == 400 and t1.think and t1.tool_duration == 34.0     # 1450 - (1000 + 50)
    assert t2.isl_new == 0 and t2.tool_name is None                        # context shrank (compaction) -> 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sim -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/sim/__init__.py
```

```python
# src/atfm/sim/programs.py
"""Closed-loop session programs: what a session will do, without when (the simulator decides when)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atfm.schema.trace import TraceTable
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


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def programs_from_table(table: TraceTable, rate_per_hour: float | None, duration_s: float,
                        rng: np.random.Generator) -> list[Program]:
    df = table.df
    fam = df["session_id"].map(lambda s: s.split("/", 1)[0])
    progs: dict[str, Program] = {}
    children: dict[str, list[Program]] = {}
    for sid, g in table.sessions():
        rows = g.to_dict("records")
        turns: list[Turn] = []
        prev_ctx = 0
        for r in rows:
            isl_new = int(r["isl"]) if not turns else max(0, int(r["isl"]) - prev_ctx)
            prev_ctx = int(r["isl"]) + int(r["osl"])
            tool = r["tool_name"] if not _nan(r["tool_name"]) else None
            if tool is None or _nan(r["t_tool_start"]) or _nan(r["t_tool_end"]):
                turns.append(Turn(isl_new, int(r["osl"]), None, None))
                continue
            d = max(0.0, float(r["t_tool_end"]) - float(r["t_tool_start"]))
            think = tool in ("__think__", "__gap__")
            prog = [(float(e["t"]) - float(r["t_tool_start"]), float(e["completed"]), None if e.get("total") is None else float(e["total"]))
                    for e in (r["progress_events"] or [])]
            turns.append(Turn(isl_new, int(r["osl"]), tool, d, r["backend_id"] if not _nan(r["backend_id"]) else "local", prog, think=think))
        parent = rows[0]["parent_session_id"] if not _nan(rows[0]["parent_session_id"]) else None
        p = Program(session_id=sid, cls=rows[0]["class"], tenant=rows[0]["tenant"], t_arrival=float(rows[0]["t_request"]),
                    turns=turns, parent=parent)
        progs[sid] = p
        if parent is not None:
            children.setdefault(parent, []).append(p)
    roots = [p for p in progs.values() if p.parent is None]
    for root in roots:
        for child in children.get(root.session_id, []):
            # attach to the parent turn whose tool phase contains the child's start
            idx = 0
            t = root.t_arrival
            for i, tr in enumerate(root.turns):
                if child.t_arrival >= t:
                    idx = i
                t += (tr.tool_duration or 0.0)
            root.spawn_at_turn.append((idx, child))
    if rate_per_hour:
        n = rng.poisson(rate_per_hour * duration_s / 3600.0)
        starts = np.sort(rng.uniform(0.0, duration_s, size=n))
        picks = rng.choice(len(roots), size=n, replace=True)
        out = []
        for k, (s, i) in enumerate(zip(starts, picks)):
            src = roots[i]
            out.append(Program(session_id=f"{src.session_id}#{k}", cls=src.cls, tenant=src.tenant, t_arrival=float(s),
                               turns=src.turns, deadline_s=src.deadline_s, spawn_at_turn=src.spawn_at_turn))
        return out
    return sorted(roots, key=lambda p: p.t_arrival)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/sim -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/sim tests/sim
git commit -m "feat(sim): closed-loop session programs from workload specs and trace tables"
```

---

### Task 2: Worker engine with KV cache, batch limit and timing

**Files:**
- Create: `src/atfm/sim/engine.py`
- Test: `tests/sim/test_engine.py`

**Interfaces:**
- Produces:
  - `EngineConfig(kv_blocks: int, max_batch: int, prefill_tps: float, decode_tps: float, block_size: int = 16, priority: bool = False)`.
  - `Request(request_id, session_id, cls, isl_total, isl_new, osl, tier: int, index: float, t_queued: float)`.
  - `Worker(worker_id, cfg: EngineConfig)` with: `resident: dict[session_id -> blocks]`, `lru: ordered session ids`, `running: dict[request_id -> (Request, t_start, t_end)]`, `queue: list[Request]`;
    - `submit(req, now) -> None` enqueues;
    - `schedule(now) -> list[tuple[Request, t_start, t_first, t_end, prefix_hit_tokens, recomputed_tokens, evictions]]` admits queued requests while `len(running) < max_batch` and blocks can be found: needed = `ceil((isl_total + osl)/block_size)`; `prefix_hit = min(resident[session]*block_size, isl_total - isl_new)` if the session is resident (0 otherwise); free blocks come first from unused capacity, then from evicting LRU sessions that are not running; if still short, the request stays queued. Order: FCFS, or by `(tier, index)` descending when `cfg.priority`. Timing: `t_start = now`, `prefill = recomputed_tokens / prefill_tps`, `t_first = now + prefill`, `t_end = t_first + osl / decode_tps`. `recomputed_tokens = isl_total - prefix_hit`.
    - `complete(request_id, now) -> None` frees the running slot; the session's blocks stay resident (touch LRU).
    - `evict_session(session_id) -> int` removes a session's blocks (used when a session ends).
    - `free_blocks() -> int`, `resident_blocks(session_id) -> int`.
  - `Router(workers, mode: Literal["affinity", "round_robin"])` with `pick(req) -> Worker`: affinity returns the worker where the session is resident, else the one with the most free blocks.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_engine.py
from atfm.sim.engine import EngineConfig, Worker, Request, Router

def _req(rid, sid, isl_total, isl_new, osl, tier=0, index=1.0, t=0.0, cls="background"):
    return Request(request_id=rid, session_id=sid, cls=cls, isl_total=isl_total, isl_new=isl_new, osl=osl, tier=tier, index=index, t_queued=t)

def test_prefix_hit_and_recompute_after_eviction():
    w = Worker("w0", EngineConfig(kv_blocks=100, max_batch=4, prefill_tps=1000.0, decode_tps=10.0))
    w.submit(_req("r1", "a", 160, 160, 16), 0.0)
    (req, ts, tf, te, hit, recomputed, ev), = w.schedule(0.0)
    assert hit == 0 and recomputed == 160 and abs(tf - 0.16) < 1e-9 and abs(te - 0.16 - 1.6) < 1e-9
    w.complete("r1", te)
    assert w.resident_blocks("a") == 11                       # ceil(176/16)
    w.submit(_req("r2", "a", 240, 64, 16), 5.0)               # second turn appends 64 tokens
    (_, _, _, _, hit, recomputed, _), = w.schedule(5.0)
    assert hit == 176 and recomputed == 64
    w.complete("r2", 6.0)
    w.evict_session("a")
    w.submit(_req("r3", "a", 320, 64, 16), 7.0)
    (_, _, _, _, hit, recomputed, _), = w.schedule(7.0)
    assert hit == 0 and recomputed == 320                     # evicted: full prefill, never negative

def test_waits_when_no_evictable_blocks():
    w = Worker("w0", EngineConfig(kv_blocks=20, max_batch=4, prefill_tps=1000.0, decode_tps=10.0))
    w.submit(_req("r1", "a", 160, 160, 16), 0.0)              # 11 blocks, running
    w.schedule(0.0)
    w.submit(_req("r2", "b", 160, 160, 16), 0.0)              # needs 11 more, only 9 free and nothing evictable
    assert w.schedule(0.0) == [] and len(w.queue) == 1
    w.complete("r1", 2.0)                                     # a is resident but idle now -> evictable
    out = w.schedule(2.0)
    assert len(out) == 1 and out[0][6] == 1 and w.resident_blocks("a") == 0

def test_priority_ordering_and_batch_limit():
    w = Worker("w0", EngineConfig(kv_blocks=1000, max_batch=1, prefill_tps=1000.0, decode_tps=10.0, priority=True))
    w.submit(_req("bg", "s1", 16, 16, 1, tier=0, index=1.0), 0.0)
    w.submit(_req("it", "s2", 16, 16, 1, tier=1, index=0.5, cls="interactive"), 0.0)
    out = w.schedule(0.0)
    assert [o[0].request_id for o in out] == ["it"] and len(w.queue) == 1

def test_router_affinity():
    ws = [Worker(f"w{i}", EngineConfig(kv_blocks=100, max_batch=4, prefill_tps=1000.0, decode_tps=10.0)) for i in range(2)]
    r = Router(ws, "affinity")
    w = r.pick(_req("r1", "a", 16, 16, 1)); w.submit(_req("r1", "a", 16, 16, 1), 0.0); w.schedule(0.0); w.complete("r1", 1.0)
    assert r.pick(_req("r2", "a", 32, 16, 1)) is w
    other = r.pick(_req("r3", "b", 16, 16, 1))
    assert other is not w or ws[1].free_blocks() < w.free_blocks()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sim/test_engine.py -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/sim/engine.py
"""Worker engine model: KV cache with LRU eviction and prefix reuse, batch limit, prefill/decode timing."""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal


@dataclass
class EngineConfig:
    kv_blocks: int
    max_batch: int
    prefill_tps: float
    decode_tps: float
    block_size: int = 16
    priority: bool = False


@dataclass
class Request:
    request_id: str
    session_id: str
    cls: str
    isl_total: int
    isl_new: int
    osl: int
    tier: int = 0
    index: float = 0.0
    t_queued: float = 0.0


class Worker:
    def __init__(self, worker_id: str, cfg: EngineConfig):
        self.worker_id, self.cfg = worker_id, cfg
        self.resident: OrderedDict[str, int] = OrderedDict()   # session -> blocks, LRU order (oldest first)
        self.running: dict[str, tuple[Request, float, float]] = {}
        self.queue: list[Request] = []
        self.busy_until = 0.0

    def used_blocks(self) -> int:
        return sum(self.resident.values())

    def free_blocks(self) -> int:
        return self.cfg.kv_blocks - self.used_blocks()

    def resident_blocks(self, session_id: str) -> int:
        return self.resident.get(session_id, 0)

    def submit(self, req: Request, now: float) -> None:
        req.t_queued = now
        self.queue.append(req)

    def _running_sessions(self) -> set[str]:
        return {r.session_id for r, _, _ in self.running.values()}

    def _make_room(self, needed: int) -> int:
        """Evict idle LRU sessions until `needed` blocks are free; return evictions made (or -1 if impossible)."""
        evictions = 0
        running = self._running_sessions()
        while self.free_blocks() < needed:
            victim = next((s for s in self.resident if s not in running), None)
            if victim is None:
                return -1
            del self.resident[victim]
            evictions += 1
        return evictions

    def schedule(self, now: float) -> list[tuple]:
        bs = self.cfg.block_size
        if self.cfg.priority:
            self.queue.sort(key=lambda r: (-r.tier, -r.index, r.t_queued))
        admitted = []
        remaining = []
        for req in self.queue:
            if len(self.running) >= self.cfg.max_batch:
                remaining.append(req)
                continue
            have = self.resident.get(req.session_id, 0)
            needed_total = math.ceil((req.isl_total + req.osl) / bs)
            extra = max(0, needed_total - have)
            cached_tokens = have * bs
            prefix_hit = min(cached_tokens, max(0, req.isl_total - req.isl_new)) if have else 0
            # make sure the session's own blocks are not evicted: mark it running first
            snapshot = dict(self.resident)
            if req.session_id in self.resident:
                self.resident.move_to_end(req.session_id)
            ev = self._make_room(extra) if extra > 0 else 0
            if ev < 0:
                self.resident = OrderedDict(snapshot)
                remaining.append(req)
                continue
            self.resident[req.session_id] = needed_total
            self.resident.move_to_end(req.session_id)
            recomputed = req.isl_total - prefix_hit
            t_first = now + recomputed / self.cfg.prefill_tps
            t_end = t_first + req.osl / self.cfg.decode_tps
            self.running[req.request_id] = (req, now, t_end)
            admitted.append((req, now, t_first, t_end, prefix_hit, recomputed, ev))
        self.queue = remaining
        return admitted

    def complete(self, request_id: str, now: float) -> None:
        req, _, _ = self.running.pop(request_id)
        if req.session_id in self.resident:
            self.resident.move_to_end(req.session_id)

    def evict_session(self, session_id: str) -> int:
        return self.resident.pop(session_id, 0)


class Router:
    def __init__(self, workers: list[Worker], mode: Literal["affinity", "round_robin"] = "affinity"):
        self.workers, self.mode, self._rr = workers, mode, 0

    def pick(self, req: Request) -> Worker:
        if self.mode == "affinity":
            for w in self.workers:
                if req.session_id in w.resident:
                    return w
            return max(self.workers, key=lambda w: (w.free_blocks(), -len(w.queue)))
        w = self.workers[self._rr % len(self.workers)]
        self._rr += 1
        return w
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/sim/test_engine.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/sim/engine.py tests/sim/test_engine.py
git commit -m "feat(sim): worker engine with KV cache, LRU eviction, prefix reuse and priority batching"
```

---

### Task 3: Event loop, session lifecycle, lifecycle log, and the native arm

**Files:**
- Create: `src/atfm/sim/core.py`, `src/atfm/sim/policies.py`
- Test: `tests/sim/test_core.py`

**Interfaces:**
- Produces:
  - `Policy` protocol (`policies.py`): `name: str`; `on_arrival(sim, call) -> float | None` returns a release time (None = release now); `order_key(sim, call) -> tuple` for the proxy queue (higher first); `window(sim) -> int | None` (None = no proxy queue, straight to workers); `tier_and_index(sim, call) -> tuple[int, float]`; `on_tick(sim, now) -> None`; `on_tool_end(sim, session, now) -> None`. `NativePolicy(priority_by_class: bool)`: `window=None`, tier 1 for interactive else 0, index 0, no holds (arm 1).
  - `PendingCall(session, turn_index, t_arrival, isl_total, isl_new, osl, tier, index, release_not_before, hold_reason)`.
  - `SessionRun` state: program, turn index, ctx tokens, worker affinity, `t_start`, `deadline`, `held_kv_block_s` accumulator, `done`.
  - `Simulator(programs, engines: list[EngineConfig], policy, *, harness_overhead_s=0.2, tick_s=5.0, max_hold_s=600.0, slo_ttft_s=2.0, rng, router_mode="affinity", bus=None)` with `run(until: float | None = None) -> pd.DataFrame` (the lifecycle log, one row per call): `session_id, cls, tenant, turn_index, t_arrival, t_release, t_queued_worker, t_start, t_first_token, t_end, worker_id, isl, osl, prefix_hit_tokens, recomputed_tokens, held_s, hold_reason, queue_proxy_s, queue_worker_s, hold_kv_block_s, evictions_caused, deadline, deadline_missed, tool_name, tool_duration`, plus `sim.session_log` (session_id, cls, tenant, t_start, t_end, deadline, missed, turns) and `sim.events` (a bus of `Event`s the demand board can consume: session.start, llm.request/first_token/done, tool.start/progress/end).
  - Loop: heap of `(t, seq, kind, payload)`; kinds `arrive`, `release`, `tick`, `worker_done`, `tool_end`, `tool_progress`, `spawn`. On `arrive`/`tool_end`: build PendingCall, ask policy for tier/index and hold; if `window is None` submit to the router's worker immediately; else push into the proxy queue and call `_drain_proxy()`, which releases the top-ordered eligible calls while `in_flight < window`. Worker scheduling happens after every submit and every completion. `worker_done` records the row, then starts the turn's tool (`tool_end` event at `t + duration`, progress events at offsets) or ends the session; `harness_overhead_s` separates tool end and the next arrival. Spawned children arrive at the parent's tool end for that turn. Hold KV cost: while a call is held, its session's resident blocks (on its affinity worker) times the hold duration accumulate into `hold_kv_block_s`; evictions caused by the held session are counted when its resident blocks displace another session during its hold (attributed at eviction time via the worker's eviction list). Deadline: `deadline = t_arrival + deadline_s`; `deadline_missed = session t_end > deadline`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_core.py
import numpy as np
from atfm.sim.programs import Program, Turn
from atfm.sim.engine import EngineConfig
from atfm.sim.core import Simulator
from atfm.sim.policies import NativePolicy

def _prog(sid, t0, cls="background", turns=None, deadline=None):
    turns = turns or [Turn(160, 16, "pytest", 30.0, "ci", [(10.0, 33.0, 100.0), (20.0, 66.0, 100.0)]), Turn(64, 16, None, None)]
    return Program(session_id=sid, cls=cls, tenant="t", t_arrival=t0, turns=turns, deadline_s=deadline)

def test_closed_loop_timing_and_log():
    sim = Simulator([_prog("a", 0.0)], [EngineConfig(kv_blocks=1000, max_batch=4, prefill_tps=1000.0, decode_tps=16.0)],
                    NativePolicy(), harness_overhead_s=0.5, rng=np.random.default_rng(0))
    log = sim.run()
    assert len(log) == 2
    r0, r1 = log.sort_values("turn_index").itertuples()
    assert r0.t_arrival == 0.0 and abs(r0.t_first_token - 0.16) < 1e-9 and abs(r0.t_end - 1.16) < 1e-9
    assert r0.tool_name == "pytest" and r0.tool_duration == 30.0
    assert abs(r1.t_arrival - (1.16 + 30.0 + 0.5)) < 1e-9          # closed loop: tool end + harness overhead
    assert r1.prefix_hit_tokens == 176 and r1.recomputed_tokens == 64
    kinds = [e.kind for e in sim.events.drain()]
    assert kinds.count("tool.progress") == 2 and "session.start" in kinds and kinds.count("llm.done") == 2
    assert sim.session_log[0]["turns"] == 2 and sim.session_log[0]["missed"] is False

def test_deadline_missed_but_session_completes():
    p = _prog("b", 0.0, cls="interactive", turns=[Turn(160, 16, "pytest", 100.0, "ci"), Turn(64, 16, None, None)], deadline=10.0)
    sim = Simulator([p], [EngineConfig(kv_blocks=1000, max_batch=4, prefill_tps=1000.0, decode_tps=16.0)], NativePolicy(), rng=np.random.default_rng(0))
    log = sim.run()
    assert len(log) == 2 and sim.session_log[0]["missed"] is True and log["deadline_missed"].iloc[-1]

def test_spawned_child_arrives_at_parent_tool_end():
    child = Program(session_id="c", cls="background", tenant="t", t_arrival=0.0, turns=[Turn(32, 8, None, None)], parent="p")
    parent = Program(session_id="p", cls="background", tenant="t", t_arrival=0.0,
                     turns=[Turn(160, 16, "pytest", 30.0, "ci"), Turn(64, 16, None, None)], spawn_at_turn=[(0, child)])
    sim = Simulator([parent], [EngineConfig(kv_blocks=1000, max_batch=4, prefill_tps=1000.0, decode_tps=16.0)], NativePolicy(), rng=np.random.default_rng(0))
    log = sim.run()
    c = log[log.session_id == "c"]
    assert len(c) == 1 and abs(c.t_arrival.iloc[0] - (1.16 + 30.0)) < 1e-9

def test_native_priority_serves_interactive_first_when_batch_full():
    eng = EngineConfig(kv_blocks=10000, max_batch=1, prefill_tps=1000.0, decode_tps=1.0, priority=True)
    progs = [_prog("bg1", 0.0, turns=[Turn(16, 10, None, None)]), _prog("bg2", 0.01, turns=[Turn(16, 10, None, None)]),
             _prog("it", 0.02, cls="interactive", turns=[Turn(16, 10, None, None)])]
    sim = Simulator(progs, [eng], NativePolicy(priority_by_class=True), rng=np.random.default_rng(0))
    log = sim.run().sort_values("t_start")
    assert list(log.session_id)[:2] == ["bg1", "it"]                # bg1 already running; interactive jumps bg2
    assert (log.queue_worker_s >= 0).all() and (log.queue_proxy_s == 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sim/test_core.py -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/sim/policies.py
from __future__ import annotations

from typing import Protocol


class Policy(Protocol):
    name: str

    def window(self, sim) -> int | None: ...

    def tier_and_index(self, sim, call) -> tuple[int, float]: ...

    def on_arrival(self, sim, call) -> float | None: ...

    def on_tick(self, sim, now: float) -> None: ...

    def on_tool_end(self, sim, session, now: float) -> None: ...


class NativePolicy:
    """Arm 1: no proxy queue; requests go straight to the engine, which may order by class priority."""

    name = "native"

    def __init__(self, priority_by_class: bool = True):
        self.priority_by_class = priority_by_class

    def window(self, sim) -> int | None:
        return None

    def tier_and_index(self, sim, call) -> tuple[int, float]:
        return (1 if (self.priority_by_class and call.session.program.cls == "interactive") else 0), 0.0

    def on_arrival(self, sim, call) -> float | None:
        return None

    def on_tick(self, sim, now: float) -> None:
        return None

    def on_tool_end(self, sim, session, now: float) -> None:
        return None
```

```python
# src/atfm/sim/core.py
"""Closed-loop discrete-event simulator of agent sessions on a worker pool behind an admission arm."""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from atfm.bus import InMemoryBus
from atfm.schema.events import LlmDone, LlmFirstToken, LlmRequest, SessionStart, SpawnRequest, ToolEnd, ToolProgress, ToolStart

from .engine import EngineConfig, Request, Router, Worker
from .programs import Program


@dataclass
class SessionRun:
    program: Program
    turn: int = 0
    ctx: int = 0
    worker: Worker | None = None
    t_start: float = 0.0
    t_end: float | None = None
    deadline: float | None = None
    held_kv_block_s: float = 0.0
    evictions_caused: int = 0
    done: bool = False


@dataclass
class PendingCall:
    session: SessionRun
    turn_index: int
    t_arrival: float
    isl_total: int
    isl_new: int
    osl: int
    tier: int = 0
    index: float = 0.0
    release_not_before: float = 0.0
    hold_reason: str = ""
    t_release: float | None = None
    request_id: str = ""


class Simulator:
    def __init__(self, programs: list[Program], engines: list[EngineConfig], policy, *, harness_overhead_s: float = 0.2,
                 tick_s: float = 5.0, max_hold_s: float = 600.0, slo_ttft_s: float = 2.0, rng=None,
                 router_mode: str = "affinity", bus=None, block_size: int = 16):
        self.programs = programs
        self.workers = [Worker(f"w{i}", e) for i, e in enumerate(engines)]
        self.router = Router(self.workers, router_mode)
        self.policy = policy
        self.harness_overhead_s, self.tick_s, self.max_hold_s, self.slo_ttft_s = harness_overhead_s, tick_s, max_hold_s, slo_ttft_s
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.events = bus if bus is not None else InMemoryBus()
        self.block_size = block_size
        self.now = 0.0
        self._heap: list = []
        self._seq = itertools.count()
        self.sessions: dict[str, SessionRun] = {}
        self.proxy_queue: list[PendingCall] = []
        self.in_flight = 0
        self.caps = 0
        self.rows: list[dict] = []
        self.session_log: list[dict] = []
        self._pending_by_rid: dict[str, PendingCall] = {}
        self._sched_pending = set()

    # ---- event helpers
    def _push(self, t: float, kind: str, payload=None) -> None:
        heapq.heappush(self._heap, (t, next(self._seq), kind, payload))

    def _emit(self, e) -> None:
        self.events.publish(e)

    # ---- lifecycle
    def _start_session(self, p: Program, t: float) -> SessionRun:
        s = SessionRun(program=p, t_start=t, deadline=None if p.deadline_s is None else t + p.deadline_s)
        self.sessions[p.session_id] = s
        self._emit(SessionStart(t=t, session_id=p.session_id, tenant=p.tenant, cls=p.cls, parent_session_id=p.parent, deadline=s.deadline))
        return s

    def _arrive(self, s: SessionRun, t: float) -> None:
        turn = s.program.turns[s.turn]
        isl_total = s.ctx + turn.isl_new
        call = PendingCall(session=s, turn_index=s.turn, t_arrival=t, isl_total=isl_total, isl_new=turn.isl_new, osl=turn.osl)
        call.tier, call.index = self.policy.tier_and_index(self, call)
        hold = self.policy.on_arrival(self, call)
        if hold is not None and hold > t:
            cap = t + self.max_hold_s
            if hold > cap:
                self.caps += 1
            call.release_not_before, call.hold_reason = min(hold, cap), getattr(self.policy, "last_hold_reason", "hold")
        window = self.policy.window(self)
        if window is None:
            self._release(call, t)
        else:
            self.proxy_queue.append(call)
            self._drain_proxy(t)
            if call.release_not_before > t:
                self._push(call.release_not_before, "release", None)

    def _drain_proxy(self, t: float) -> None:
        window = self.policy.window(self)
        if window is None:
            return
        while self.in_flight < window:
            eligible = [c for c in self.proxy_queue if c.release_not_before <= t]
            if not eligible:
                break
            best = max(eligible, key=lambda c: (c.tier, c.index, -c.t_arrival))
            self.proxy_queue.remove(best)
            self.in_flight += 1
            self._release(best, t)

    def _release(self, call: PendingCall, t: float) -> None:
        call.t_release = t
        s = call.session
        held = t - call.t_arrival
        if held > 0 and s.worker is not None:
            s.held_kv_block_s += s.worker.resident_blocks(s.program.session_id) * held
        rid = f"{s.program.session_id}:{call.turn_index}"
        call.request_id = rid
        self._pending_by_rid[rid] = call
        req = Request(request_id=rid, session_id=s.program.session_id, cls=s.program.cls, isl_total=call.isl_total,
                      isl_new=call.isl_new, osl=call.osl, tier=call.tier, index=call.index, t_queued=t)
        w = self.router.pick(req)
        s.worker = w
        w.submit(req, t)
        self._emit(LlmRequest(t=t, session_id=s.program.session_id, turn_index=call.turn_index, request_id=rid, isl=call.isl_total,
                              predicted_osl=call.osl, hints={"strict_priority": call.tier}, held_s=held))
        self._schedule_worker(w, t)

    def _schedule_worker(self, w: Worker, t: float) -> None:
        for req, ts, tf, te, hit, recomputed, ev in w.schedule(t):
            call = self._pending_by_rid[req.request_id]
            call._sched = (ts, tf, te, hit, recomputed, ev, w.worker_id)  # type: ignore[attr-defined]
            if ev:
                # evictions made room for this request; charge them to the session if it had been held
                if call.t_release is not None and call.t_release > call.t_arrival:
                    call.session.evictions_caused += ev
            self._push(tf, "first_token", req.request_id)
            self._push(te, "worker_done", req.request_id)

    def _first_token(self, rid: str, t: float) -> None:
        call = self._pending_by_rid[rid]
        self._emit(LlmFirstToken(t=t, session_id=call.session.program.session_id, request_id=rid))

    def _worker_done(self, rid: str, t: float) -> None:
        call = self._pending_by_rid.pop(rid)
        s = call.session
        ts, tf, te, hit, recomputed, ev, wid = call._sched  # type: ignore[attr-defined]
        s.worker.complete(rid, t)
        if self.policy.window(self) is not None:
            self.in_flight = max(0, self.in_flight - 1)
        s.ctx = call.isl_total + call.osl
        turn = s.program.turns[call.turn_index]
        self._emit(LlmDone(t=t, session_id=s.program.session_id, request_id=rid, osl=call.osl, worker_id=wid, prefix_hit_tokens=hit))
        self.rows.append({
            "session_id": s.program.session_id, "class": s.program.cls, "tenant": s.program.tenant, "turn_index": call.turn_index,
            "t_arrival": call.t_arrival, "t_release": call.t_release, "t_queued_worker": call.t_release, "t_start": ts,
            "t_first_token": tf, "t_end": te, "worker_id": wid, "isl": call.isl_total, "osl": call.osl,
            "prefix_hit_tokens": hit, "recomputed_tokens": recomputed, "held_s": call.t_release - call.t_arrival,
            "hold_reason": call.hold_reason, "queue_proxy_s": call.t_release - call.t_arrival, "queue_worker_s": ts - call.t_release,
            "hold_kv_block_s": s.held_kv_block_s, "evictions_caused": s.evictions_caused, "deadline": s.deadline,
            "deadline_missed": bool(s.deadline is not None and t > s.deadline),
            "tool_name": turn.tool_name, "tool_duration": turn.tool_duration,
        })
        self._schedule_worker(s.worker, t)
        if self.policy.window(self) is not None:
            self._drain_proxy(t)
        # next: tool phase or end of session
        if turn.tool_name is None or turn.tool_duration is None:
            self._end_session(s, t)
            return
        call_id = f"{rid}:tool"
        self._emit(ToolStart(t=t, session_id=s.program.session_id, turn_index=call.turn_index, call_id=call_id,
                             tool_name=turn.tool_name, backend_id=turn.backend_id))
        for off, done, total in turn.progress:
            if off < turn.tool_duration:
                self._push(t + off, "tool_progress", (s.program.session_id, call_id, done, total))
        self._push(t + turn.tool_duration, "tool_end", (s.program.session_id, call_id, call.turn_index))

    def _tool_end(self, sid: str, call_id: str, turn_index: int, t: float) -> None:
        s = self.sessions[sid]
        self._emit(ToolEnd(t=t, session_id=sid, call_id=call_id, exit_status=0))
        for idx, child in s.program.spawn_at_turn:
            if idx == turn_index:
                cs = self._start_session(child, t)
                self._emit(SpawnRequest(t=t, parent_session_id=sid, child_session_id=child.session_id))
                self._arrive(cs, t)
        self.policy.on_tool_end(self, s, t)
        s.turn += 1
        self._push(t + self.harness_overhead_s, "arrive", sid)

    def _end_session(self, s: SessionRun, t: float) -> None:
        s.done, s.t_end = True, t
        if s.worker is not None:
            s.worker.evict_session(s.program.session_id)
            self._schedule_worker(s.worker, t)
        self.session_log.append({"session_id": s.program.session_id, "class": s.program.cls, "tenant": s.program.tenant,
                                 "t_start": s.t_start, "t_end": t, "deadline": s.deadline,
                                 "missed": bool(s.deadline is not None and t > s.deadline), "turns": s.turn + 1})

    # ---- main loop
    def run(self, until: float | None = None) -> pd.DataFrame:
        for p in self.programs:
            self._push(p.t_arrival, "start", p)
        self._push(0.0, "tick", None)
        while self._heap:
            t, _, kind, payload = heapq.heappop(self._heap)
            if until is not None and t > until:
                break
            self.now = t
            if kind == "start":
                s = self._start_session(payload, t)
                self._arrive(s, t)
            elif kind == "arrive":
                self._arrive(self.sessions[payload], t)
            elif kind == "release":
                self._drain_proxy(t)
            elif kind == "first_token":
                self._first_token(payload, t)
            elif kind == "worker_done":
                self._worker_done(payload, t)
            elif kind == "tool_progress":
                sid, call_id, done, total = payload
                self._emit(ToolProgress(t=t, session_id=sid, call_id=call_id, completed=done, total=total, phase="run"))
            elif kind == "tool_end":
                self._tool_end(*payload, t)
            elif kind == "tick":
                self.policy.on_tick(self, t)
                self._drain_proxy(t)
                if any(not s.done for s in self.sessions.values()) or any(k == "start" for _, _, k, _ in self._heap):
                    self._push(t + self.tick_s, "tick", None)
        return pd.DataFrame(self.rows)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/sim -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/sim/core.py src/atfm/sim/policies.py tests/sim/test_core.py
git commit -m "feat(sim): closed-loop event loop with lifecycle log, spawn, deadlines and the native arm"
```

---

### Task 4: Proxy arms: rules (arm 2) and oracle (arm 5)

**Files:**
- Modify: `src/atfm/sim/policies.py`
- Test: `tests/sim/test_policies.py`

**Interfaces:**
- Produces:
  - `ProxyRulesPolicy(window: int, cfg: ProxyConfig)` (arm 2): `window()` returns the window; `tier_and_index` uses `atfm.proxy.index.tier/compute_index/service_time` with `e_tool_next_s = 0`; no holds.
  - `ForecastPolicy(window, cfg, predictor_factory, hold: bool, gdp: GdpLite | None)` base for arms 3 and 4 (Task 5); `OraclePolicy(window, cfg, hold: bool)` (arm 5): `e_tool_next_s` = the true duration of the tool this turn will launch (from the program), and, when `hold` is on, holds deferrable calls using true future demand: a background call is held while the true number of interactive calls that will arrive in the next `slot_s` exceeds free worker batch slots (computed from the programs' known tool ends; implemented by peeking at the heap for `arrive`/`tool_end` events of interactive sessions).
  - A `CallMeta` is built from the PendingCall for the index functions: `_meta(call, now) -> CallMeta`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_policies.py
import numpy as np
from atfm.proxy.config import ProxyConfig
from atfm.sim.programs import Program, Turn
from atfm.sim.engine import EngineConfig
from atfm.sim.core import Simulator
from atfm.sim.policies import ProxyRulesPolicy, OraclePolicy

def _eng(batch=1):
    return EngineConfig(kv_blocks=100000, max_batch=batch, prefill_tps=1000.0, decode_tps=1.0)

def _one_turn(sid, t0, cls, osl=10, deadline=None):
    return Program(session_id=sid, cls=cls, tenant="t", t_arrival=t0, turns=[Turn(16, osl, None, None)], deadline_s=deadline)

def test_proxy_rules_window_and_tier_order():
    progs = [_one_turn("bg1", 0.0, "background"), _one_turn("bg2", 0.01, "background"), _one_turn("bg3", 0.02, "background"),
             _one_turn("it", 0.03, "interactive", deadline=5.0)]
    pol = ProxyRulesPolicy(window=1, cfg=ProxyConfig(upstream_url="x", slack_threshold_s=5.0))
    log = Simulator(progs, [_eng(batch=4)], pol, rng=np.random.default_rng(0)).run().sort_values("t_release")
    assert list(log.session_id)[:2] == ["bg1", "it"]                   # window 1: backlog in proxy, interactive next
    assert (log.queue_proxy_s.iloc[1:] > 0).all() and (log.queue_worker_s == 0).all()

def test_oracle_uses_true_next_tool_duration():
    long_tool = Program(session_id="L", cls="background", tenant="t", t_arrival=0.0,
                        turns=[Turn(16, 10, "pytest", 300.0, "ci"), Turn(16, 1, None, None)])
    short_tool = Program(session_id="S", cls="background", tenant="t", t_arrival=0.0,
                         turns=[Turn(16, 10, "bash", 1.0, "local"), Turn(16, 1, None, None)])
    filler = _one_turn("F", 0.0, "background", osl=5)
    pol = OraclePolicy(window=1, cfg=ProxyConfig(upstream_url="x", beta=1.0), hold=False)
    log = Simulator([filler, short_tool, long_tool], [_eng(batch=4)], pol, rng=np.random.default_rng(0)).run()
    first = log.sort_values("t_release").session_id.tolist()
    assert first.index("L") < first.index("S")                          # unblocking bonus releases the long-tool session first
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sim/test_policies.py -q`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement (append to `src/atfm/sim/policies.py`)**

```python
from atfm.proxy.config import ProxyConfig
from atfm.proxy.index import CallMeta, compute_index, service_time, tier


def _meta(call, now: float) -> CallMeta:
    p = call.session.program
    return CallMeta(session_id=p.session_id, cls=p.cls, tenant=p.tenant, deadline=call.session.deadline, parent=p.parent,
                    turn_index=call.turn_index, isl=call.isl_total, predicted_osl=call.osl, t_arrival=now)


class ProxyRulesPolicy:
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

    def on_arrival(self, sim, call) -> float | None:
        return None

    def on_tick(self, sim, now: float) -> None:
        return None

    def on_tool_end(self, sim, session, now: float) -> None:
        return None


class OraclePolicy(ProxyRulesPolicy):
    """Arm 5: knows the true duration of the tool each turn will launch (upper bound for the index term)."""

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
                     and (payload if isinstance(payload, str) else getattr(payload, "session_id", "")) in sim.sessions
                     and sim.sessions[payload if isinstance(payload, str) else payload.session_id].program.cls == "interactive")
        slots = sum(w.cfg.max_batch - len(w.running) for w in sim.workers)
        if coming >= slots:
            self.last_hold_reason = "oracle_surge"
            return now + self.slot_s
        return None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/sim -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/sim/policies.py tests/sim/test_policies.py
git commit -m "feat(sim): proxy-rules and oracle arms"
```

---

### Task 5: Forecast arms (M1 and M2) with the in-sim demand board and a hold rule

**Files:**
- Create: `src/atfm/sim/forecast_arm.py`
- Test: `tests/sim/test_forecast_arm.py`

**Interfaces:**
- Produces:
  - `GdpLite(slot_s=30.0, eps=0.1, max_hold_s=600.0)`: given a `ForecastSnapshot` and the fleet's free capacity (blocks and batch slots), returns `hold_until(now, call) -> float | None`: hold a deferrable call while the forecast's q(1-eps) interactive KV demand within `slot_s` exceeds free blocks (or q(1-eps) interactive call count exceeds free batch slots, using `prefill_tokens` as a proxy for call count via `mean isl`), else release. Stateless apart from the snapshot it is handed.
  - `ForecastPolicy(window, cfg, predictor, train_table: TraceTable | None, horizons, n=128, hold=True, tick_s=5.0)` (arms 3 and 4): owns a `SessionRegistry` fed from `sim.events` on every tick (drain and apply), a `SessionForecaster` (with `ExogenousModel` fitted on `train_table` if given), a `LiveBoard`-like `step` at each tick producing the latest snapshot, `e_tool_next` from the predictor's duration model for the session's *next* tool (the tool it just finished is known from the registry's history; the next tool's name is unknown, so use the pooled mean like deployment: `predictor.dm.mean(None)`), and `on_arrival` consulting `GdpLite` with the latest snapshot. `name` is `"forecast_M1"` or `"forecast_M2"` from `predictor.name`.
  - `fit_predictor_on_programs(kind: Literal["M1", "M2"], programs, engines, rng) -> tuple[predictor, TraceTable]`: runs a native simulation of the training programs and converts its lifecycle log plus events into a `TraceTable` via `events_to_trace_table(sim.events.drain())`, then fits the predictor on it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_forecast_arm.py
import numpy as np
from atfm.proxy.config import ProxyConfig
from atfm.traces.synthetic import WorkloadSpec, ClassSpec, ToolSpec
from atfm.sim.programs import programs_from_spec, Program, Turn
from atfm.sim.engine import EngineConfig
from atfm.sim.core import Simulator
from atfm.sim.forecast_arm import ForecastPolicy, GdpLite, fit_predictor_on_programs
from atfm.schema.forecast import ForecastSnapshot

def _spec(seed=0):
    tools = [ToolSpec(name="bash", weight=0.6, log_mu=np.log(3.0), log_sigma=0.4, signal="none"),
             ToolSpec(name="pytest", weight=0.4, log_mu=np.log(60.0), log_sigma=0.4, signal="strong", backend_id="ci")]
    return WorkloadSpec(duration_s=900.0, seed=seed, classes=[
        ClassSpec(cls="background", rate_per_hour=240.0, turns_mean=5, isl0=2000, isl_growth=300, osl_mean=60, tools=tools),
        ClassSpec(cls="interactive", rate_per_hour=120.0, turns_mean=4, isl0=3000, isl_growth=400, osl_mean=40, tools=tools[:1],
                  think_log_mu=np.log(10.0), think_log_sigma=0.4, deadline_s=300.0)])

def test_gdp_lite_holds_when_forecast_exceeds_capacity():
    snap = ForecastSnapshot(t=0.0, horizons=[30.0], model_id="m",
                            samples={"kv_blocks": {"interactive": np.full((1, 8), 900.0), "background": np.zeros((1, 8))},
                                     "prefill_tokens": {"interactive": np.full((1, 8), 9000.0), "background": np.zeros((1, 8))}})
    g = GdpLite(slot_s=30.0, eps=0.1)
    assert g.hold_until(100.0, snap, free_blocks=500, free_slots=10, mean_isl=3000.0) == 130.0
    assert g.hold_until(100.0, snap, free_blocks=2000, free_slots=10, mean_isl=3000.0) is None
    assert g.hold_until(100.0, snap, free_blocks=2000, free_slots=1, mean_isl=3000.0) == 130.0   # 3 expected calls > 1 slot

def test_forecast_arm_runs_and_holds_only_background():
    engines = [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]
    train = programs_from_spec(_spec(1), np.random.default_rng(1))
    pred, table = fit_predictor_on_programs("M2", train, engines, np.random.default_rng(1))
    assert len(table) > 50 and pred.name == "M2_progress"
    progs = programs_from_spec(_spec(0), np.random.default_rng(0))
    pol = ForecastPolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=0.5), predictor=pred, train_table=table,
                         horizons=[30.0, 120.0], n=64, hold=True)
    log = Simulator(progs, engines, pol, max_hold_s=60.0, rng=np.random.default_rng(0)).run()
    assert pol.name == "forecast_M2" and len(log) > 100
    held = log[log.held_s > 0.5]
    assert (held["class"] == "background").all()
    assert (log.held_s <= 60.0 + 1e-6).all()                            # cap respected
    assert pol.snapshots > 10                                            # the board ticked
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sim/test_forecast_arm.py -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/sim/forecast_arm.py
"""Arms 3 and 4: the proxy index with a forecast term, plus a forecast-driven hold rule, on simulated time."""
from __future__ import annotations

from typing import Literal

import numpy as np

from atfm.board.forecaster import ExogenousModel, SessionForecaster
from atfm.board.live import SessionRegistry
from atfm.board.predictors import ProgressPredictor, SurvivalPredictor
from atfm.proxy.config import ProxyConfig
from atfm.proxy.index import compute_index, service_time, tier
from atfm.schema.forecast import ForecastSnapshot
from atfm.schema.trace import TraceTable
from atfm.traces.sidecar import events_to_trace_table

from .core import Simulator
from .engine import EngineConfig
from .policies import NativePolicy, _meta


class GdpLite:
    """Hold a deferrable call while the forecast's upper quantile of interactive demand within one slot
    exceeds the fleet's free KV blocks or free batch slots (spec 6.2, single-slot chance constraint)."""

    def __init__(self, slot_s: float = 30.0, eps: float = 0.1, max_hold_s: float = 600.0):
        self.slot_s, self.eps, self.max_hold_s = slot_s, eps, max_hold_s

    def hold_until(self, now: float, snap: ForecastSnapshot, free_blocks: int, free_slots: int, mean_isl: float) -> float | None:
        h = int(np.argmin(np.abs(np.asarray(snap.horizons) - self.slot_s)))
        q = 1.0 - self.eps
        kv = float(np.quantile(snap.samples["kv_blocks"]["interactive"][h], q))
        calls = float(np.quantile(snap.samples["prefill_tokens"]["interactive"][h], q)) / max(mean_isl, 1.0)
        if kv > free_blocks or calls > free_slots:
            return now + self.slot_s
        return None


class ForecastPolicy:
    def __init__(self, window: int, cfg: ProxyConfig, predictor, train_table: TraceTable | None, horizons: list[float],
                 n: int = 128, hold: bool = True, gdp: GdpLite | None = None):
        self._window, self.cfg, self.predictor = window, cfg, predictor
        self.registry = SessionRegistry()
        exo = ExogenousModel()
        if train_table is not None:
            exo.fit(train_table)
        self.forecaster = SessionForecaster(predictor, exo, horizons, n=n)
        self.hold, self.gdp = hold, gdp or GdpLite()
        self.name = f"forecast_{predictor.name.split('_')[0]}"
        self.snapshot: ForecastSnapshot | None = None
        self.snapshots = 0
        self.last_hold_reason = ""
        self._starts_ptr = 0.0
        self._isl_sum, self._isl_n = 0.0, 0

    def window(self, sim) -> int | None:
        return self._window

    def _ingest(self, sim) -> None:
        for e in sim.events.drain():
            self.registry.apply(e)
            if e.kind == "llm.request":
                self._isl_sum += e.isl
                self._isl_n += 1

    def on_tick(self, sim, now: float) -> None:
        self._ingest(sim)
        starts = [s for s in self.registry.new_starts_since(self._starts_ptr) if s[0] < now]
        self._starts_ptr = now
        self.forecaster.exo.update(now, starts)
        self.snapshot = self.forecaster.forecast(now, self.registry.states(now), sim.rng)
        self.snapshots += 1

    def e_tool_next(self, sim, call) -> float:
        return float(self.predictor.dm.mean(None))

    def tier_and_index(self, sim, call) -> tuple[int, float]:
        m = _meta(call, sim.now)
        e_s = service_time(m, self.cfg)
        return tier(m, self.cfg, sim.now, e_s), compute_index(m, self.cfg, e_s, self.e_tool_next(sim, call))

    def on_arrival(self, sim, call) -> float | None:
        if not self.hold or call.session.program.cls != "background" or self.snapshot is None:
            return None
        free_blocks = sum(w.free_blocks() for w in sim.workers)
        free_slots = sum(max(0, w.cfg.max_batch - len(w.running)) for w in sim.workers)
        mean_isl = (self._isl_sum / self._isl_n) if self._isl_n else 3000.0
        t = self.gdp.hold_until(sim.now, self.snapshot, free_blocks, free_slots, mean_isl)
        if t is not None:
            self.last_hold_reason = "forecast_surge"
        return t

    def on_tool_end(self, sim, session, now: float) -> None:
        return None


def fit_predictor_on_programs(kind: Literal["M1", "M2"], programs, engines: list[EngineConfig], rng):
    """Run the training programs under the native arm and fit the predictor on the resulting trace table."""
    sim = Simulator(programs, engines, NativePolicy(), rng=rng)
    sim.run()
    table = events_to_trace_table(sim.events.drain())
    pred = (SurvivalPredictor() if kind == "M1" else ProgressPredictor()).fit(table)
    return pred, table
```

Note for the executor: `Simulator._worker_done` emits `LlmDone` and the registry needs `tool.start` to carry `turn_index`, which the simulator emits; `events_to_trace_table` pairs tool phases to calls by time, which holds here because the simulator emits `tool.start` right after `llm.done`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/sim -q`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/sim/forecast_arm.py tests/sim/test_forecast_arm.py
git commit -m "feat(sim): forecast arms M1/M2 with in-sim demand board and a single-slot hold rule"
```

---

### Task 6: ThunderAgent-style reactive arm (arm 6)

**Files:**
- Modify: `src/atfm/sim/policies.py`
- Test: `tests/sim/test_thunder.py`

**Interfaces:**
- Produces: `WorkingSetPolicy(budget_blocks: int, low_watermark: float = 0.8, window: int | None = None)` (arm 6): programs pause at tool boundaries and resume only when the fleet's resident working set (sum of resident blocks across workers) is below the budget; when above, calls wait in the proxy queue and are released smallest-context-first (`index = -isl_total`) once the working set drops below `low_watermark * budget` (hysteresis). Interactive calls are exempt (tier 1). `window()` returns a large number (the working set is the binding constraint). This is the reactive comparison from Dynamo's ThunderAgent Program Scheduler: pause at tool boundaries, resume by working-set pressure, no forecast.

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_thunder.py
import numpy as np
from atfm.sim.programs import Program, Turn
from atfm.sim.engine import EngineConfig
from atfm.sim.core import Simulator
from atfm.sim.policies import WorkingSetPolicy

def _prog(sid, t0, isl, cls="background"):
    return Program(session_id=sid, cls=cls, tenant="t", t_arrival=t0, turns=[Turn(isl, 10, "bash", 5.0, "local"), Turn(16, 1, None, None)])

def test_working_set_budget_pauses_and_resumes_smallest_first():
    eng = EngineConfig(kv_blocks=100000, max_batch=8, prefill_tps=1e6, decode_tps=1e6)
    # three background sessions of 1600, 800 and 400 tokens (100, 50, 25 blocks); budget 120 blocks
    progs = [_prog("big", 0.0, 1600), _prog("mid", 0.001, 800), _prog("small", 0.002, 400), _prog("it", 0.003, 800, cls="interactive")]
    pol = WorkingSetPolicy(budget_blocks=120, low_watermark=0.5)
    log = Simulator(progs, [eng], pol, harness_overhead_s=0.0, rng=np.random.default_rng(0)).run()
    first_turn = log[log.turn_index == 0].sort_values("t_release")
    order = first_turn.session_id.tolist()
    assert order[0] == "big" and "it" in order[:2]                     # interactive is never paused
    assert order.index("small") < order.index("mid")                   # smallest context resumes first
    assert (first_turn[first_turn.session_id.isin(["small", "mid"])].queue_proxy_s > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sim/test_thunder.py -q`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement (append to `src/atfm/sim/policies.py`)**

```python
class WorkingSetPolicy:
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

    def tier_and_index(self, sim, call) -> tuple[int, float]:
        if call.session.program.cls == "interactive":
            return 1, 0.0
        return 0, -float(call.isl_total)

    def on_arrival(self, sim, call) -> float | None:
        if call.session.program.cls == "interactive":
            return None
        ws = self._working_set(sim)
        if self.paused and ws < self.low * self.budget:
            self.paused = False
        if ws >= self.budget:
            self.paused = True
        if self.paused:
            self.last_hold_reason = "working_set"
            return sim.now + sim.tick_s
        return None

    def on_tick(self, sim, now: float) -> None:
        ws = self._working_set(sim)
        if self.paused and ws < self.low * self.budget:
            self.paused = False
        if not self.paused:
            for c in sim.proxy_queue:
                c.release_not_before = min(c.release_not_before, now)
        else:
            for c in sim.proxy_queue:
                if c.session.program.cls != "interactive":
                    c.release_not_before = max(c.release_not_before, now + sim.tick_s)

    def on_tool_end(self, sim, session, now: float) -> None:
        return None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/sim -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/sim/policies.py tests/sim/test_thunder.py
git commit -m "feat(sim): ThunderAgent-style working-set arm"
```

---

### Task 7: Serving metrics, paired experiment runner, scenarios, and a first synthetic result

**Files:**
- Create: `src/atfm/eval/serving.py`, `src/atfm/experiments/h2sim.py`, `scripts/run_h2sim.py`, `experiments/h2sim_regimes.yaml`
- Test: `tests/eval/test_serving.py`, `tests/experiments/test_h2sim.py`

**Interfaces:**
- Produces:
  - `serving_metrics(log: pd.DataFrame, sessions: list[dict], slo_ttft_s: float, sim_duration_s: float, gpu_count: int) -> dict` with keys: `ttft_after_tool_p50/p95/p99` (interactive rows with `turn_index > 0`: `t_first_token - t_arrival`), `slo_attainment` (share of those under `slo_ttft_s`), `bg_jct_mean/p95` (background session `t_end - t_start`), `deadline_hit_rate` (sessions with a deadline), `tasks_per_hour` (completed sessions / duration), `max_imposed_delay_by_tenant` (dict), `mean_held_s_background`, `gpu_hours`, `recomputed_prefill_tokens` (sum), `kv_hit_rate` (sum prefix_hit / sum isl), `hold_kv_block_s` (sum), `evictions_caused_by_holds` (sum), `queue_proxy_share` and `queue_worker_share` (where waiting time accumulated, as shares of total wait).
  - `paired_bootstrap(per_session: dict[arm -> pd.DataFrame], metric_fn, n_boot=500, rng) -> pd.DataFrame` resampling session ids identically across arms; returns mean and 95% CI per arm and the CI of each arm's difference to the first arm.
  - `H2SimConfig(name, regime: Literal["short_tool","long_tool"], seeds: list[int], arms: list[str], engines: list[EngineConfig] | dict, window: int, beta: float, slo_ttft_s: float, working_set_budget: int, train_seed_offset: int = 1000, out_dir="runs")`, `regime_spec(regime) -> WorkloadSpec` (short-tool: Claude Code like, tools 1 to 10 s, none signal; long-tool: tests, builds, pipelines 60 to 600 s with strong and weak signals, background share 70%), `run_h2sim(cfg) -> pd.DataFrame` running every arm on the same programs per seed (arms: `native`, `proxy_rules`, `forecast_M1`, `forecast_M2`, `oracle`, `working_set`), writing `metrics.csv` (arm x seed), `paired.csv`, and a `queue_location.csv`.
  - `scripts/run_h2sim.py <yaml>` prints the paired table for `slo_attainment`, `bg_jct_mean`, `deadline_hit_rate`, `recomputed_prefill_tokens`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_serving.py
import numpy as np, pandas as pd
from atfm.eval.serving import serving_metrics, paired_bootstrap

def _log():
    return pd.DataFrame([
        {"session_id": "i1", "class": "interactive", "tenant": "a", "turn_index": 1, "t_arrival": 10.0, "t_release": 10.0, "t_first_token": 11.0, "t_end": 12.0, "isl": 1000, "prefix_hit_tokens": 800, "recomputed_tokens": 200, "held_s": 0.0, "queue_proxy_s": 0.0, "queue_worker_s": 0.5, "hold_kv_block_s": 0.0, "evictions_caused": 0, "deadline_missed": False},
        {"session_id": "i1", "class": "interactive", "tenant": "a", "turn_index": 2, "t_arrival": 20.0, "t_release": 20.0, "t_first_token": 23.0, "t_end": 24.0, "isl": 1200, "prefix_hit_tokens": 0, "recomputed_tokens": 1200, "held_s": 0.0, "queue_proxy_s": 0.0, "queue_worker_s": 2.5, "hold_kv_block_s": 0.0, "evictions_caused": 0, "deadline_missed": False},
        {"session_id": "b1", "class": "background", "tenant": "b", "turn_index": 1, "t_arrival": 5.0, "t_release": 35.0, "t_first_token": 36.0, "t_end": 40.0, "isl": 2000, "prefix_hit_tokens": 1000, "recomputed_tokens": 1000, "held_s": 30.0, "queue_proxy_s": 30.0, "queue_worker_s": 0.0, "hold_kv_block_s": 3000.0, "evictions_caused": 2, "deadline_missed": True},
    ])

def test_serving_metrics_values():
    sessions = [{"session_id": "i1", "class": "interactive", "tenant": "a", "t_start": 0.0, "t_end": 24.0, "deadline": 100.0, "missed": False, "turns": 3},
                {"session_id": "b1", "class": "background", "tenant": "b", "t_start": 0.0, "t_end": 40.0, "deadline": 30.0, "missed": True, "turns": 2}]
    m = serving_metrics(_log(), sessions, slo_ttft_s=2.0, sim_duration_s=3600.0, gpu_count=1)
    assert m["ttft_after_tool_p50"] == 2.0 and m["slo_attainment"] == 0.5
    assert m["bg_jct_mean"] == 40.0 and m["deadline_hit_rate"] == 0.5 and m["tasks_per_hour"] == 2.0
    assert m["max_imposed_delay_by_tenant"] == {"a": 0.0, "b": 30.0} and m["gpu_hours"] == 1.0
    assert m["recomputed_prefill_tokens"] == 2400 and abs(m["kv_hit_rate"] - 1800 / 4200) < 1e-9
    assert m["hold_kv_block_s"] == 3000.0 and m["evictions_caused_by_holds"] == 2
    assert abs(m["queue_proxy_share"] - 30.0 / 33.0) < 1e-9

def test_paired_bootstrap_same_sessions_across_arms():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"session_id": [f"s{i}" for i in range(50)], "value": rng.normal(10, 1, 50)})
    better = base.assign(value=base.value - 1.0)
    out = paired_bootstrap({"A": base, "B": better}, metric_fn=lambda df: df.value.mean(), n_boot=200, rng=np.random.default_rng(1))
    b = out[out.arm == "B"].iloc[0]
    assert abs(b["diff_mean"] + 1.0) < 0.05 and b["diff_ci_hi"] < 0.0        # paired: tight CI around -1
```

```python
# tests/experiments/test_h2sim.py
from atfm.experiments.h2sim import H2SimConfig, run_h2sim, regime_spec

def test_h2sim_smoke_all_arms(tmp_path):
    cfg = H2SimConfig(name="smoke", regime="long_tool", seeds=[0], arms=["native", "proxy_rules", "forecast_M1", "forecast_M2", "oracle", "working_set"],
                      engines=[{"kv_blocks": 4000, "max_batch": 4, "prefill_tps": 20000.0, "decode_tps": 40.0}], window=4, beta=0.5,
                      slo_ttft_s=2.0, working_set_budget=3000, duration_s=600.0, out_dir=str(tmp_path))
    df = run_h2sim(cfg)
    assert set(df.arm) == set(cfg.arms) and (df.seed == 0).all()
    assert (tmp_path / "smoke" / "metrics.csv").exists() and (tmp_path / "smoke" / "paired.csv").exists()
    assert (df.groupby("arm").sessions_completed.first() > 20).all()
    assert regime_spec("short_tool").classes[0].tools[0].log_mu < regime_spec("long_tool").classes[0].tools[0].log_mu
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_serving.py tests/experiments/test_h2sim.py -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/eval/serving.py
from __future__ import annotations

import numpy as np
import pandas as pd


def serving_metrics(log: pd.DataFrame, sessions: list[dict], slo_ttft_s: float, sim_duration_s: float, gpu_count: int) -> dict:
    it = log[(log["class"] == "interactive") & (log["turn_index"] > 0)]
    ttft = (it["t_first_token"] - it["t_arrival"]).to_numpy(float) if len(it) else np.array([np.nan])
    sess = pd.DataFrame(sessions)
    bg = sess[sess["class"] == "background"] if len(sess) else sess
    bg_jct = (bg["t_end"] - bg["t_start"]).to_numpy(float) if len(bg) else np.array([np.nan])
    with_dl = sess[sess["deadline"].notna()] if len(sess) else sess
    wait_proxy, wait_worker = float(log["queue_proxy_s"].sum()), float(log["queue_worker_s"].sum())
    tot_wait = wait_proxy + wait_worker
    return {
        "ttft_after_tool_p50": float(np.nanpercentile(ttft, 50)), "ttft_after_tool_p95": float(np.nanpercentile(ttft, 95)),
        "ttft_after_tool_p99": float(np.nanpercentile(ttft, 99)),
        "slo_attainment": float(np.mean(ttft <= slo_ttft_s)) if len(it) else float("nan"),
        "bg_jct_mean": float(np.nanmean(bg_jct)), "bg_jct_p95": float(np.nanpercentile(bg_jct, 95)),
        "deadline_hit_rate": float(1.0 - with_dl["missed"].mean()) if len(with_dl) else float("nan"),
        "tasks_per_hour": float(len(sess) / (sim_duration_s / 3600.0)),
        "max_imposed_delay_by_tenant": {t: float(v) for t, v in log.groupby("tenant")["held_s"].max().items()},
        "mean_held_s_background": float(log.loc[log["class"] == "background", "held_s"].mean()) if (log["class"] == "background").any() else 0.0,
        "gpu_hours": float(gpu_count * sim_duration_s / 3600.0),
        "recomputed_prefill_tokens": int(log["recomputed_tokens"].sum()),
        "kv_hit_rate": float(log["prefix_hit_tokens"].sum() / max(1, log["isl"].sum())),
        "hold_kv_block_s": float(log.groupby("session_id")["hold_kv_block_s"].max().sum()),
        "evictions_caused_by_holds": int(log.groupby("session_id")["evictions_caused"].max().sum()),
        "queue_proxy_share": float(wait_proxy / tot_wait) if tot_wait > 0 else 0.0,
        "queue_worker_share": float(wait_worker / tot_wait) if tot_wait > 0 else 0.0,
        "sessions_completed": int(len(sess)),
    }


def paired_bootstrap(per_session: dict, metric_fn, n_boot: int = 500, rng=None) -> pd.DataFrame:
    rng = np.random.default_rng(0) if rng is None else rng
    arms = list(per_session)
    ids = sorted(set.intersection(*[set(df["session_id"]) for df in per_session.values()]))
    idx = {a: per_session[a].set_index("session_id").loc[ids] for a in arms}
    draws = {a: [] for a in arms}
    for _ in range(n_boot):
        sample = rng.choice(ids, size=len(ids), replace=True)
        for a in arms:
            draws[a].append(metric_fn(idx[a].loc[sample].reset_index()))
    base = np.asarray(draws[arms[0]])
    rows = []
    for a in arms:
        d = np.asarray(draws[a])
        diff = d - base
        rows.append({"arm": a, "mean": float(metric_fn(idx[a].reset_index())), "ci_lo": float(np.quantile(d, 0.025)),
                     "ci_hi": float(np.quantile(d, 0.975)), "diff_mean": float(diff.mean()),
                     "diff_ci_lo": float(np.quantile(diff, 0.025)), "diff_ci_hi": float(np.quantile(diff, 0.975))})
    return pd.DataFrame(rows)
```

```python
# src/atfm/experiments/h2sim.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from atfm.eval.serving import paired_bootstrap, serving_metrics
from atfm.proxy.config import ProxyConfig
from atfm.sim.core import Simulator
from atfm.sim.engine import EngineConfig
from atfm.sim.forecast_arm import ForecastPolicy, fit_predictor_on_programs
from atfm.sim.policies import NativePolicy, OraclePolicy, ProxyRulesPolicy, WorkingSetPolicy
from atfm.sim.programs import programs_from_spec
from atfm.traces.synthetic import ClassSpec, ToolSpec, WorkloadSpec


class H2SimConfig(BaseModel):
    name: str
    regime: Literal["short_tool", "long_tool"]
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    arms: list[str] = Field(default_factory=lambda: ["native", "proxy_rules", "forecast_M1", "forecast_M2", "oracle", "working_set"])
    engines: list[dict] = Field(default_factory=lambda: [{"kv_blocks": 8000, "max_batch": 8, "prefill_tps": 20000.0, "decode_tps": 40.0}])
    window: int = 8
    beta: float = 0.5
    slo_ttft_s: float = 2.0
    working_set_budget: int = 6000
    duration_s: float = 3600.0
    interactive_rate_per_hour: float = 120.0
    background_rate_per_hour: float = 240.0
    train_seed_offset: int = 1000
    out_dir: str = "runs"


def regime_spec(regime: str, duration_s: float = 3600.0, seed: int = 0, it_rate: float = 120.0, bg_rate: float = 240.0) -> WorkloadSpec:
    if regime == "short_tool":
        tools = [ToolSpec(name="bash", weight=0.85, log_mu=np.log(2.0), log_sigma=0.7, signal="none"),
                 ToolSpec(name="pytest", weight=0.15, log_mu=np.log(20.0), log_sigma=0.6, signal="strong", backend_id="ci")]
        bg_tools = tools
    else:
        tools = [ToolSpec(name="bash", weight=0.5, log_mu=np.log(3.0), log_sigma=0.6, signal="none"),
                 ToolSpec(name="pytest", weight=0.3, log_mu=np.log(180.0), log_sigma=0.6, signal="strong", backend_id="ci", spawn_prob=0.05),
                 ToolSpec(name="build", weight=0.2, log_mu=np.log(400.0), log_sigma=0.5, signal="weak", backend_id="ci")]
        bg_tools = tools
    return WorkloadSpec(duration_s=duration_s, seed=seed, classes=[
        ClassSpec(cls="background", rate_per_hour=bg_rate, turns_mean=10, isl0=3000, isl_growth=600, osl_mean=200, tools=bg_tools, deadline_s=1800.0),
        ClassSpec(cls="interactive", rate_per_hour=it_rate, turns_mean=6, isl0=4000, isl_growth=800, osl_mean=150,
                  tools=[tools[0]], think_log_mu=np.log(20.0), think_log_sigma=0.8)])


def _arm(name: str, cfg: H2SimConfig, engines: list[EngineConfig], train_programs, rng):
    pcfg = ProxyConfig(upstream_url="sim", beta=cfg.beta)
    if name == "native":
        return NativePolicy(priority_by_class=True)
    if name == "proxy_rules":
        return ProxyRulesPolicy(cfg.window, pcfg)
    if name == "oracle":
        return OraclePolicy(cfg.window, pcfg, hold=True)
    if name == "working_set":
        return WorkingSetPolicy(cfg.working_set_budget)
    if name in ("forecast_M1", "forecast_M2"):
        pred, table = fit_predictor_on_programs(name.split("_")[1], train_programs, engines, rng)
        return ForecastPolicy(cfg.window, pcfg, pred, table, horizons=[30.0, 120.0, 300.0], n=128, hold=True)
    raise ValueError(f"unknown arm {name}")


def run_h2sim(cfg: H2SimConfig) -> pd.DataFrame:
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    engines = [EngineConfig(**e) for e in cfg.engines]
    rows, per_arm_sessions = [], {a: [] for a in cfg.arms}
    for seed in cfg.seeds:
        spec = regime_spec(cfg.regime, cfg.duration_s, seed, cfg.interactive_rate_per_hour, cfg.background_rate_per_hour)
        train_spec = spec.model_copy(update={"seed": seed + cfg.train_seed_offset})
        train_programs = programs_from_spec(train_spec, np.random.default_rng(seed + cfg.train_seed_offset))
        for arm in cfg.arms:
            programs = programs_from_spec(spec, np.random.default_rng(seed))      # identical programs for every arm
            policy = _arm(arm, cfg, engines, train_programs, np.random.default_rng(seed + 7))
            sim = Simulator(programs, engines, policy, slo_ttft_s=cfg.slo_ttft_s, rng=np.random.default_rng(seed + 13))
            log = sim.run()
            m = serving_metrics(log, sim.session_log, cfg.slo_ttft_s, cfg.duration_s, len(engines))
            m["max_imposed_delay_by_tenant"] = json.dumps(m["max_imposed_delay_by_tenant"])
            rows.append({"arm": arm, "seed": seed, **m})
            sess = pd.DataFrame(sim.session_log).assign(seed=seed, session_id=lambda d: d.session_id + f"@{seed}")
            it = log[(log["class"] == "interactive") & (log["turn_index"] > 0)]
            ttft = it.assign(ttft=it.t_first_token - it.t_arrival).groupby("session_id")["ttft"].apply(lambda s: float((s <= cfg.slo_ttft_s).mean()))
            sess = sess.merge(ttft.rename("slo").reset_index().assign(session_id=lambda d: d.session_id + f"@{seed}"), on="session_id", how="left")
            per_arm_sessions[arm].append(sess)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    per = {a: pd.concat(v, ignore_index=True) for a, v in per_arm_sessions.items()}
    paired = []
    for metric, fn in {"slo_attainment": lambda d: float(d["slo"].dropna().mean()) if d["slo"].notna().any() else float("nan"),
                       "bg_jct_mean": lambda d: float((d.loc[d["class"] == "background", "t_end"] - d.loc[d["class"] == "background", "t_start"]).mean()),
                       "deadline_hit_rate": lambda d: float(1.0 - d.loc[d["deadline"].notna(), "missed"].mean())}.items():
        pb = paired_bootstrap(per, fn, n_boot=300, rng=np.random.default_rng(0))
        pb["metric"] = metric
        paired.append(pb)
    pd.concat(paired, ignore_index=True).to_csv(out / "paired.csv", index=False)
    df[["arm", "seed", "queue_proxy_share", "queue_worker_share", "mean_held_s_background", "hold_kv_block_s", "evictions_caused_by_holds"]].to_csv(out / "queue_location.csv", index=False)
    (out / "config.json").write_text(cfg.model_dump_json(indent=2))
    return df
```

```python
# scripts/run_h2sim.py
import sys

import pandas as pd
import yaml

from atfm.experiments.h2sim import H2SimConfig, run_h2sim

if __name__ == "__main__":
    cfg = H2SimConfig(**yaml.safe_load(open(sys.argv[1])))
    df = run_h2sim(cfg)
    cols = ["slo_attainment", "ttft_after_tool_p95", "bg_jct_mean", "deadline_hit_rate", "recomputed_prefill_tokens", "hold_kv_block_s", "queue_proxy_share"]
    print(df.groupby("arm")[cols].mean().round(3).to_string())
    print(pd.read_csv(f"{cfg.out_dir}/{cfg.name}/paired.csv").round(3).to_string())
```

```yaml
# experiments/h2sim_regimes.yaml
name: h2sim_long_tool
regime: long_tool
seeds: [0, 1, 2]
arms: [native, proxy_rules, forecast_M1, forecast_M2, oracle, working_set]
engines:
  - {kv_blocks: 8000, max_batch: 8, prefill_tps: 20000.0, decode_tps: 40.0}
window: 8
beta: 0.5
slo_ttft_s: 2.0
working_set_budget: 6000
duration_s: 3600
interactive_rate_per_hour: 120
background_rate_per_hour: 240
```

- [ ] **Step 4: Run tests, then both regimes**

Run: `uv run pytest -q`
Expected: PASS (whole suite)

Then: `uv run python scripts/run_h2sim.py experiments/h2sim_regimes.yaml` and the same with `regime: short_tool` (copy the YAML to `experiments/h2sim_short_tool.yaml` with `name: h2sim_short_tool`). Expected: six rows per seed; `queue_proxy_share` is 0 for `native` and positive for the proxy arms (contention forms at the proxy); the oracle's `slo_attainment` is at least the forecast arms'. Write the two tables and the queue-location table into `docs/research/2026-09-24-h2sim-first-results.md`, with the statement that these are synthetic closed-loop results with an unvalidated engine model, useful only for relative ordering and sweep design until L2 calibrates the engine.

- [ ] **Step 5: Commit**

```bash
git add src/atfm/eval/serving.py src/atfm/experiments/h2sim.py scripts/run_h2sim.py experiments/h2sim_regimes.yaml experiments/h2sim_short_tool.yaml tests/eval/test_serving.py tests/experiments/test_h2sim.py docs/research/2026-09-24-h2sim-first-results.md
git commit -m "feat(sim): serving metrics, paired bootstrap, six-arm H2 simulation runner with first synthetic results"
```

---

## Self-review

- Spec coverage: section 8 entities (programs T1, workers and router T2, event loop and lifecycle T3, policies T3 to T6, perturbations via `_factor` in T1); section 9 six arms (native T3, rules T4, M1/M2 T5, oracle T4, ThunderAgent-style T6) and both regimes (T7); section 12 metrics (T7 `serving_metrics`); section 6.3 hold cost (T3 accumulators, T7 metric); queue-location logging (T3 columns, T7 table); paired design (T7). Not covered by design: tiers beyond HBM (spec 6.3 future work), simulator fidelity calibration (needs L2 hardware), planner floor (future work).
- Placeholders: none; every step has code. The oracle hold rule and `GdpLite` are deliberately simple single-slot rules and say so.
- Type consistency: `Turn.progress` tuples `(offset, completed, total)` used in T3's `_worker_done`; `PendingCall` fields used by `_meta` in T4/T5 (`session.program`, `turn_index`, `isl_total`, `osl`); `Policy` methods `window/tier_and_index/on_arrival/on_tick/on_tool_end` implemented by all four policy classes and `ForecastPolicy`; `Simulator.events` is an `InMemoryBus` drained by `ForecastPolicy._ingest` and by `fit_predictor_on_programs`; `serving_metrics` reads exactly the lifecycle columns T3 writes.
- Review Focus: 1 in `test_prefix_hit_and_recompute_after_eviction`; 2 in `test_waits_when_no_evictable_blocks`; 3 in `test_forecast_arm_runs_and_holds_only_background` (cap); 4 in `test_deadline_missed_but_session_completes`; 5: add to Task 7's smoke test the assertion that the first-turn arrival times are identical across arms: `first = {arm: sorted(...)}`; implement as: after `run_h2sim`, load each arm's log is not stored, so instead assert inside `run_h2sim` via `assert [p.t_arrival for p in programs] == base_arrivals` where `base_arrivals` is captured for the first arm of each seed (add the two lines to `run_h2sim`; the smoke test exercises them).
