# L0 Core: Trace Schema, Predictor Ladder, Forecaster and H1 Harness. Implementation Plan

> **Historical implementation plan.** Embedded code and task checklists are a design record,
> not the current source of setup instructions. See [implementation status](../../status.md),
> [operations](../../operations.md), and [core development](../../development/core.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CPU-only core that turns agent traces (TraceLab or synthetic) into fleet demand forecasts from in-flight session state, scores the predictor ladder B0 to M3 against ground truth, and produces the H1 result.

**Architecture:** A canonical trace table (one row per LLM call, with the tool phase that call launched) is produced by adapters and by a synthetic generator. A replayer reconstructs the fleet's in-flight session states at each tick. Predictors implement one interface (session state in, samples of time-to-next-call out). A Monte Carlo forecaster aggregates per-session samples plus exogenous arrivals into demand distributions per horizon and class. Evaluation scores samples with CRPS, pinball loss, coverage and surge lead time on time-block splits.

**Tech Stack:** Python 3.12, uv, pydantic v2, numpy, scipy, pandas, pyarrow, pyyaml, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-atfm-architecture-design.md` (sections 3, 5, 8 workload spec, 9). Research context: `docs/research/*.md`.

## Global Constraints

- Python 3.12 exactly (`requires-python = ">=3.12,<3.13"`); managed with `uv`; package name `atfm`, src layout.
- All timestamps are float seconds since the Unix epoch, UTC. Durations are float seconds.
- Every stochastic function takes `rng: np.random.Generator`; no global random state.
- `np.inf` in a resumption sample means "never resumes within any horizon".
- KV blocks for a call = `ceil(isl / block_size)` with `block_size = 16` default.
- `class` values are exactly `"interactive"` or `"background"`; in pydantic the field is `cls` with alias `class`.
- Horizons default to `[10, 30, 120, 300, 900]` seconds. Monte Carlo sample count default `n = 512`.
- Data files live under `data/` which is git-ignored. Tests never read `data/`; they build fixtures inline.
- Time-block splits only; never random-row splits.

## Review Focus

1. A session whose tool has run longer than any duration in the training set: M1 must still return finite positive samples (tail extrapolation), not raise or return empty arrays. Test in Task 6.
2. A round with several tools launched in parallel: the tool phase is from the earliest `emitted_at` to the latest `result_at`, never a negative or zero duration for the turn. Test in Task 3.
3. A session with a single round and no tools (one-shot question): it must produce one row with `tool_name=None` and no tool phase, and the replayer must mark it ended after `t_last_token`. Test in Tasks 3 and 5.
4. Ticks where no session is in flight and no arrivals occur: the forecaster must emit all-zero samples with valid quantiles and `endogenous_fraction = 0.0`, not NaN. Test in Task 8.
5. CRPS with degenerate (point) forecasts must equal absolute error, and pinball loss at q90 must penalize under-forecasts nine times more than over-forecasts. Test in Task 9.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/atfm/__init__.py`, `tests/__init__.py`, `tests/test_scaffold.py`, `.gitignore`

**Interfaces:**
- Produces: importable package `atfm` with `atfm.__version__ == "0.1.0"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaffold.py
def test_import():
    import atfm
    assert atfm.__version__ == "0.1.0"
```

- [ ] **Step 2: Create pyproject.toml and package**

```toml
# pyproject.toml
[project]
name = "atfm"
version = "0.1.0"
description = "Agent traffic flow management for LLM serving"
requires-python = ">=3.12,<3.13"
dependencies = [
  "pydantic>=2.7",
  "numpy>=1.26",
  "scipy>=1.11",
  "pandas>=2.1",
  "pyarrow>=15",
  "pyyaml>=6",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov>=5"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/atfm"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# src/atfm/__init__.py
__version__ = "0.1.0"
```

```
# .gitignore
data/
.venv/
__pycache__/
*.pyc
runs/
.pytest_cache/
```

- [ ] **Step 3: Install and run test**

Run: `uv python install 3.12 && uv sync --extra dev && uv run pytest tests/test_scaffold.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/atfm/__init__.py tests/ .gitignore
git commit -m "feat: scaffold atfm package"
```

---

### Task 2: Trace schema and table

**Files:**
- Create: `src/atfm/schema/__init__.py`, `src/atfm/schema/trace.py`, `src/atfm/schema/forecast.py`
- Test: `tests/schema/test_trace.py`

**Interfaces:**
- Produces:
  - `ProgressEvent(t: float, completed: float, total: float | None, phase: str | None)`
  - `DataEvent(t: float, metric: str, value: float)`
  - `TraceRow` pydantic model (fields below), `TRACE_COLUMNS: list[str]`
  - `TraceTable` with `df: pd.DataFrame`, `from_rows(rows: list[TraceRow]) -> TraceTable`, `to_parquet(path)`, `from_parquet(path) -> TraceTable`, `sessions() -> Iterator[tuple[str, pd.DataFrame]]` (sorted by t_request), `time_range() -> tuple[float, float]`, `kv_blocks(block_size=16) -> np.ndarray`
  - `ForecastSnapshot(t: float, horizons: list[float], model_id: str, samples: dict[str, dict[str, np.ndarray]]` keyed `[target][class]` with array shape `(H, n)`, `endogenous_fraction: dict[str, np.ndarray]` shape `(H,)` per class), with `quantiles(target, cls, q) -> np.ndarray` shape `(H,)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/schema/test_trace.py
import numpy as np
import pandas as pd
from atfm.schema.trace import TraceRow, TraceTable, ProgressEvent, TRACE_COLUMNS
from atfm.schema.forecast import ForecastSnapshot

def _row(**kw):
    base = dict(session_id="s1", cls="interactive", tenant="t1", turn_index=0,
                t_request=100.0, t_first_token=101.0, t_last_token=102.0,
                isl=1000, osl=50, source="test")
    base.update(kw)
    return TraceRow(**base)

def test_row_alias_class():
    r = TraceRow(**{"session_id": "s", "class": "background", "tenant": "t", "turn_index": 0,
                    "t_request": 1.0, "isl": 16, "osl": 1, "source": "test"})
    assert r.cls == "background"
    assert r.model_dump(by_alias=True)["class"] == "background"

def test_table_roundtrip(tmp_path):
    rows = [_row(), _row(turn_index=1, t_request=200.0, tool_name="pytest",
                      t_tool_start=102.5, t_tool_end=199.0,
                      progress_events=[ProgressEvent(t=150.0, completed=10, total=20, phase="run")])]
    t = TraceTable.from_rows(rows)
    assert list(t.df.columns) == TRACE_COLUMNS
    p = tmp_path / "t.parquet"
    t.to_parquet(p)
    t2 = TraceTable.from_parquet(p)
    assert len(t2.df) == 2
    assert t2.df.loc[1, "progress_events"][0]["completed"] == 10
    assert t2.time_range() == (100.0, 200.0)

def test_kv_blocks():
    t = TraceTable.from_rows([_row(isl=17), _row(isl=16, turn_index=1, t_request=101.0)])
    assert t.kv_blocks(16).tolist() == [2, 1]

def test_sessions_sorted():
    t = TraceTable.from_rows([_row(t_request=300.0, turn_index=1), _row(t_request=100.0)])
    sid, df = next(t.sessions())
    assert sid == "s1" and df["t_request"].tolist() == [100.0, 300.0]

def test_snapshot_quantiles():
    s = ForecastSnapshot(t=0.0, horizons=[10.0, 30.0], model_id="m",
                         samples={"kv_blocks": {"interactive": np.array([[1, 2, 3, 4], [10, 20, 30, 40]], float)}},
                         endogenous_fraction={"interactive": np.array([1.0, 0.5])})
    q = s.quantiles("kv_blocks", "interactive", 0.5)
    assert q.shape == (2,) and q[0] == 2.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/schema -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement schema**

```python
# src/atfm/schema/__init__.py
from .trace import TraceRow, TraceTable, ProgressEvent, DataEvent, TRACE_COLUMNS
from .forecast import ForecastSnapshot
__all__ = ["TraceRow", "TraceTable", "ProgressEvent", "DataEvent", "TRACE_COLUMNS", "ForecastSnapshot"]
```

```python
# src/atfm/schema/trace.py
from __future__ import annotations
import math
from pathlib import Path
from typing import Iterator, Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

Class = Literal["interactive", "background"]

class ProgressEvent(BaseModel):
    t: float
    completed: float
    total: float | None = None
    phase: str | None = None

class DataEvent(BaseModel):
    t: float
    metric: str
    value: float

class TraceRow(BaseModel):
    """One LLM call plus the tool phase this call launched (if any)."""
    model_config = ConfigDict(populate_by_name=True)
    session_id: str
    parent_session_id: str | None = None
    cls: Class = Field(alias="class")
    tenant: str
    turn_index: int
    t_request: float
    t_first_token: float | None = None
    t_last_token: float | None = None
    isl: int
    osl: int
    prefix_hit_tokens: int = 0
    kv_blocks_by_tier: dict[str, int] = Field(default_factory=dict)
    worker_id: str | None = None
    tool_name: str | None = None
    tool_args_hash: str | None = None
    t_tool_start: float | None = None
    t_tool_end: float | None = None
    tool_exit_status: int | None = None
    progress_events: list[ProgressEvent] = Field(default_factory=list)
    data_events: list[DataEvent] = Field(default_factory=list)
    backend_id: str | None = None
    spawned_children: int = 0
    perturbation_flag: bool = False
    source: str

TRACE_COLUMNS: list[str] = [
    "session_id", "parent_session_id", "class", "tenant", "turn_index",
    "t_request", "t_first_token", "t_last_token", "isl", "osl", "prefix_hit_tokens",
    "kv_blocks_by_tier", "worker_id", "tool_name", "tool_args_hash",
    "t_tool_start", "t_tool_end", "tool_exit_status", "progress_events", "data_events",
    "backend_id", "spawned_children", "perturbation_flag", "source",
]

class TraceTable:
    def __init__(self, df: pd.DataFrame):
        missing = [c for c in TRACE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"trace table missing columns: {missing}")
        self.df = df[TRACE_COLUMNS].sort_values(["session_id", "t_request"], kind="stable").reset_index(drop=True)

    @classmethod
    def from_rows(cls, rows: list[TraceRow]) -> "TraceTable":
        recs = [r.model_dump(by_alias=True) for r in rows]
        df = pd.DataFrame.from_records(recs, columns=TRACE_COLUMNS)
        return cls(df)

    def to_parquet(self, path: str | Path) -> None:
        self.df.to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "TraceTable":
        df = pd.read_parquet(path)
        for col in ("progress_events", "data_events"):
            df[col] = df[col].apply(lambda v: [] if v is None else [dict(x) for x in v])
        df["kv_blocks_by_tier"] = df["kv_blocks_by_tier"].apply(lambda v: {} if v is None else dict(v))
        return cls(df)

    def sessions(self) -> Iterator[tuple[str, pd.DataFrame]]:
        for sid, g in self.df.groupby("session_id", sort=False):
            yield sid, g.sort_values("t_request")

    def time_range(self) -> tuple[float, float]:
        return float(self.df["t_request"].min()), float(self.df["t_request"].max())

    def kv_blocks(self, block_size: int = 16) -> np.ndarray:
        return np.ceil(self.df["isl"].to_numpy() / block_size).astype(int)

    def __len__(self) -> int:
        return len(self.df)
```

```python
# src/atfm/schema/forecast.py
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ForecastSnapshot:
    t: float
    horizons: list[float]
    model_id: str
    samples: dict[str, dict[str, np.ndarray]]          # [target][class] -> (H, n)
    endogenous_fraction: dict[str, np.ndarray] = field(default_factory=dict)  # [class] -> (H,)

    def quantiles(self, target: str, cls: str, q: float) -> np.ndarray:
        return np.quantile(self.samples[target][cls], q, axis=1)

    def total(self, target: str) -> np.ndarray:
        return sum(self.samples[target].values())
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/schema -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/schema tests/schema
git commit -m "feat: trace schema, trace table and forecast snapshot"
```

---

### Task 3: TraceLab adapter

**Files:**
- Create: `src/atfm/traces/__init__.py`, `src/atfm/traces/tracelab.py`
- Test: `tests/traces/test_tracelab.py`

**Interfaces:**
- Consumes: `TraceRow`, `TraceTable`, `ProgressEvent`.
- Produces: `load_tracelab(path: str | Path, limit_sessions: int | None = None) -> TraceTable` reading gzipped JSONL; `rounds_to_rows(rounds: list[dict]) -> list[TraceRow]` for one session's rounds sorted by `round_index`.

Mapping rules (from the real v0.0.1 format):
- `session_id` = record `session_id`; `tenant` = record `user`; `class` = `"interactive"` (all TraceLab sessions are human-driven); `source` = `"tracelab"`; `parent_session_id` = None.
- `t_request` = timestamp of the first `timing_events` entry (the input arriving: `user_message` or `tool_result`). If `timing_events` is empty, skip the round.
- `t_first_token` = timestamp of the first event with `source` starting with `assistant.`; `t_last_token` = timestamp of the last such event. If none, both None.
- `isl` = `input_tokens_total`; `prefix_hit_tokens` = `prefix_tokens`; `osl` = `output_tokens` (0 if None).
- Tool phase: if `tools` is non-empty, `tool_name` = name of the tool with the largest `tool_wall_latency_ms` (the turn's critical path), `t_tool_start` = min `emitted_at`, `t_tool_end` = max `result_at` (fall back to `emitted_at + tool_wall_latency_ms/1000` when `result_at` is None), `tool_exit_status` = 1 if any `is_error` else 0, `backend_id` = `"local"`.
- If `tools` is empty and this is not the last round of the session, the turn ended with text and the user thought: `tool_name = "__think__"`, `t_tool_start = t_last_token` (or `t_request` if None), `t_tool_end` = next round's `t_request`.
- If `tools` is empty and it is the last round: `tool_name = None`.
- Subagent spawns: `spawned_children` = count of tools named `"Agent"` or `"Task"` in the round.
- Timestamps parse ISO-8601 with `Z` to epoch seconds via `datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/traces/test_tracelab.py
import gzip, json
from atfm.traces.tracelab import rounds_to_rows, load_tracelab

def _ev(t, typ, src, **kw):
    return {"event_type": typ, "timestamp": t, "source": src, **kw}

def _round(idx, events, tools, isl=1000, osl=50):
    return {"provider": "claude", "session_id": "claude:abc", "round_index": idx, "round_id": f"r{idx}",
            "model": "m", "input_tokens_total": isl, "prefix_tokens": isl // 2, "newly_append_tokens": isl // 2,
            "output_tokens": osl, "timing_events": events, "tools": tools, "user": "user_1",
            "first_input_event_type": events[0]["event_type"] if events else None}

def _tool(name, emitted, result, ms, err=False, cid="c1"):
    return {"tool_index": 0, "tool_name": name, "tool_call_id": cid, "emitted_at": emitted,
            "input_chars": 10, "result_chars": 20, "tool_wall_latency_ms": ms,
            "tool_internal_latency_ms": None, "is_error": err, "result_at": result}

T0 = "2026-04-18T08:23:07.000Z"; T1 = "2026-04-18T08:23:09.000Z"; T2 = "2026-04-18T08:23:10.000Z"
T3 = "2026-04-18T08:23:12.000Z"; T4 = "2026-04-18T08:23:15.000Z"; T5 = "2026-04-18T08:23:20.000Z"; T6 = "2026-04-18T08:24:00.000Z"

def test_three_round_session():
    rounds = [
        _round(0, [_ev(T0, "user_message", "user.message"), _ev(T1, "text", "assistant.content.text"),
                   _ev(T2, "tool_call", "assistant.content.tool_use")],
               [_tool("Bash", T2, T3, 2000)]),
        _round(1, [_ev(T3, "tool_result", "user.content.tool_result"), _ev(T4, "text", "assistant.content.text")], []),
        _round(2, [_ev(T6, "user_message", "user.message"), _ev(T6, "text", "assistant.content.text")], []),
    ]
    rows = rounds_to_rows(rounds)
    assert [r.turn_index for r in rows] == [0, 1, 2]
    r0, r1, r2 = rows
    assert r0.tool_name == "Bash" and r0.t_tool_end - r0.t_tool_start == 2.0
    assert r0.t_first_token < r0.t_last_token and r0.prefix_hit_tokens == 500
    assert r1.tool_name == "__think__" and r1.t_tool_start == r1.t_last_token
    assert abs(r1.t_tool_end - r2.t_request) < 1e-6
    assert r2.tool_name is None and r2.t_tool_start is None
    assert all(r.cls == "interactive" and r.tenant == "user_1" for r in rows)

def test_parallel_tools_critical_path():
    rounds = [_round(0, [_ev(T0, "tool_result", "user.content.tool_result"), _ev(T1, "tool_call", "assistant.content.tool_use")],
                     [_tool("Read", T1, T2, 1000, cid="a"), _tool("Bash", T1, T4, 6000, cid="b"), _tool("Read", T2, T3, 2000, cid="c")])]
    r = rounds_to_rows(rounds)[0]
    assert r.tool_name == "Bash" and r.t_tool_end - r.t_tool_start == 6.0 and r.tool_exit_status == 0

def test_error_and_spawn_flags():
    rounds = [_round(0, [_ev(T0, "user_message", "user.message"), _ev(T1, "tool_call", "assistant.content.tool_use")],
                     [_tool("Agent", T1, T5, 11000, err=True)])]
    r = rounds_to_rows(rounds)[0]
    assert r.tool_exit_status == 1 and r.spawned_children == 1

def test_load_gz(tmp_path):
    p = tmp_path / "t.jsonl.gz"
    with gzip.open(p, "wt") as f:
        for rr in [_round(1, [_ev(T3, "tool_result", "user.content.tool_result")], []),
                   _round(0, [_ev(T0, "user_message", "user.message"), _ev(T2, "tool_call", "assistant.content.tool_use")], [_tool("Bash", T2, T3, 2000)])]:
            f.write(json.dumps(rr) + "\n")
    t = load_tracelab(p)
    assert len(t) == 2 and t.df["turn_index"].tolist() == [0, 1] and t.df["source"].iloc[0] == "tracelab"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/traces/test_tracelab.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement adapter**

```python
# src/atfm/traces/__init__.py
```

```python
# src/atfm/traces/tracelab.py
from __future__ import annotations
import gzip, json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from atfm.schema.trace import TraceRow, TraceTable

SPAWN_TOOLS = {"Agent", "Task"}
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
    tools = rec.get("tools") or []
    tool_name = t_start = t_end = exit_status = backend = None
    spawned = 0
    if tools:
        crit = max(tools, key=lambda t: t.get("tool_wall_latency_ms") or 0)
        tool_name = crit["tool_name"]
        starts = [_ts(t["emitted_at"]) for t in tools if t.get("emitted_at")]
        ends = []
        for t in tools:
            e = _ts(t.get("result_at"))
            if e is None and t.get("emitted_at") and t.get("tool_wall_latency_ms") is not None:
                e = _ts(t["emitted_at"]) + t["tool_wall_latency_ms"] / 1000.0
            if e is not None:
                ends.append(e)
        t_start = min(starts) if starts else t_last
        t_end = max(ends) if ends else None
        if t_start is not None and t_end is not None and t_end < t_start:
            t_end = t_start
        exit_status = 1 if any(t.get("is_error") for t in tools) else 0
        backend = "local"
        spawned = sum(1 for t in tools if t["tool_name"] in SPAWN_TOOLS)
    elif next_t_request is not None:
        tool_name = THINK
        t_start = t_last if t_last is not None else t_request
        t_end = max(next_t_request, t_start)
        backend = "human"
    return TraceRow(
        session_id=rec["session_id"], parent_session_id=None, cls="interactive",
        tenant=rec.get("user") or "unknown", turn_index=turn_index,
        t_request=t_request, t_first_token=t_first, t_last_token=t_last,
        isl=int(rec.get("input_tokens_total") or 0), osl=int(rec.get("output_tokens") or 0),
        prefix_hit_tokens=int(rec.get("prefix_tokens") or 0),
        tool_name=tool_name, t_tool_start=t_start, t_tool_end=t_end, tool_exit_status=exit_status,
        backend_id=backend, spawned_children=spawned, source="tracelab",
    )

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
    for sid, recs in by_session.items():
        rows.extend(rounds_to_rows(recs))
    return TraceTable.from_rows(rows)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/traces/test_tracelab.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Convert the real dataset once and commit code only**

Run: `uv run python -c "from atfm.traces.tracelab import load_tracelab; t=load_tracelab('data/tracelab/syfi_coding_trace.jsonl.gz'); t.to_parquet('data/tracelab/tracelab.parquet'); print(len(t), t.df.session_id.nunique())"`
Expected: prints roughly `357161 4265` (rounds without timing events are dropped, so slightly fewer rows is acceptable)

```bash
git add src/atfm/traces tests/traces
git commit -m "feat: TraceLab adapter to canonical trace table"
```

---

### Task 4: Overlay and time-block split

**Files:**
- Create: `src/atfm/traces/transform.py`
- Test: `tests/traces/test_transform.py`

**Interfaces:**
- Produces:
  - `split_by_time_blocks(table: TraceTable, block_seconds: float = 7*86400, test_fraction: float = 0.3, seed: int = 0) -> tuple[TraceTable, TraceTable]`: assigns each session to the block of its first `t_request`; a random subset of blocks (by seed) is the test set; sessions stay whole.
  - `overlay_sessions(table: TraceTable, rate_per_hour: float, duration_s: float, seed: int, t0: float = 0.0) -> TraceTable`: draws Poisson session start times in `[t0, t0 + duration_s)`, samples sessions with replacement from the table, shifts each sampled session so its first `t_request` equals its new start, keeps internal timing, and renames `session_id` to `f"{orig}#{k}"`. Rows whose `t_request` exceeds `t0 + duration_s` are kept (sessions run past the end).

- [ ] **Step 1: Write the failing tests**

```python
# tests/traces/test_transform.py
import numpy as np
from atfm.schema.trace import TraceRow, TraceTable
from atfm.traces.transform import split_by_time_blocks, overlay_sessions

def _sess(sid, t0, n=3, gap=100.0):
    rows = []
    for i in range(n):
        rows.append(TraceRow(session_id=sid, cls="interactive", tenant="t", turn_index=i,
                             t_request=t0 + i * gap, t_first_token=t0 + i * gap + 1, t_last_token=t0 + i * gap + 2,
                             isl=100, osl=10, tool_name="Bash" if i < n - 1 else None,
                             t_tool_start=(t0 + i * gap + 2) if i < n - 1 else None,
                             t_tool_end=(t0 + (i + 1) * gap) if i < n - 1 else None, source="test"))
    return rows

def test_split_keeps_sessions_whole():
    rows = []
    for k in range(20):
        rows += _sess(f"s{k}", t0=k * 86400.0 * 2)
    t = TraceTable.from_rows(rows)
    tr, te = split_by_time_blocks(t, block_seconds=7 * 86400, test_fraction=0.3, seed=1)
    assert set(tr.df.session_id) & set(te.df.session_id) == set()
    assert len(tr) + len(te) == len(t) and len(te) > 0

def test_overlay_preserves_internal_timing():
    t = TraceTable.from_rows(_sess("a", 5000.0) + _sess("b", 9000.0, gap=50.0))
    o = overlay_sessions(t, rate_per_hour=600.0, duration_s=3600.0, seed=3)
    assert o.df.session_id.nunique() > 50
    for sid, g in o.sessions():
        d = np.diff(g["t_request"].to_numpy())
        assert np.allclose(d, 100.0) or np.allclose(d, 50.0)
        assert g["t_request"].iloc[0] >= 0.0 and g["t_request"].iloc[0] < 3600.0
        assert g["t_tool_end"].iloc[0] - g["t_tool_start"].iloc[0] > 0

def test_overlay_deterministic():
    t = TraceTable.from_rows(_sess("a", 5000.0))
    a = overlay_sessions(t, 100.0, 3600.0, seed=7).df
    b = overlay_sessions(t, 100.0, 3600.0, seed=7).df
    assert a.equals(b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/traces/test_transform.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement**

```python
# src/atfm/traces/transform.py
from __future__ import annotations
import numpy as np
import pandas as pd
from atfm.schema.trace import TraceTable

TIME_COLS = ["t_request", "t_first_token", "t_last_token", "t_tool_start", "t_tool_end"]

def split_by_time_blocks(table: TraceTable, block_seconds: float = 7 * 86400, test_fraction: float = 0.3,
                         seed: int = 0) -> tuple[TraceTable, TraceTable]:
    df = table.df
    first = df.groupby("session_id")["t_request"].min()
    t0 = float(first.min())
    block = ((first - t0) // block_seconds).astype(int)
    blocks = np.array(sorted(block.unique()))
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(blocks) * test_fraction)))
    test_blocks = set(rng.choice(blocks, size=n_test, replace=False).tolist())
    test_sessions = set(block[block.isin(test_blocks)].index)
    is_test = df["session_id"].isin(test_sessions)
    return TraceTable(df[~is_test].copy()), TraceTable(df[is_test].copy())

def _shift_events(events: list, delta: float) -> list:
    return [{**e, "t": e["t"] + delta} for e in events]

def overlay_sessions(table: TraceTable, rate_per_hour: float, duration_s: float, seed: int,
                     t0: float = 0.0) -> TraceTable:
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate_per_hour * duration_s / 3600.0)
    starts = np.sort(rng.uniform(t0, t0 + duration_s, size=n))
    groups = {sid: g for sid, g in table.sessions()}
    sids = np.array(list(groups.keys()))
    picks = rng.choice(len(sids), size=n, replace=True)
    out = []
    for k, (start, idx) in enumerate(zip(starts, picks)):
        g = groups[sids[idx]].copy()
        delta = start - float(g["t_request"].iloc[0])
        for c in TIME_COLS:
            g[c] = g[c] + delta
        g["progress_events"] = g["progress_events"].apply(lambda ev: _shift_events(ev, delta))
        g["data_events"] = g["data_events"].apply(lambda ev: _shift_events(ev, delta))
        g["session_id"] = f"{sids[idx]}#{k}"
        g["parent_session_id"] = g["parent_session_id"].apply(lambda p: None if p is None else f"{p}#{k}")
        out.append(g)
    if not out:
        return TraceTable(table.df.iloc[0:0].copy())
    return TraceTable(pd.concat(out, ignore_index=True))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/traces/test_transform.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/traces/transform.py tests/traces/test_transform.py
git commit -m "feat: time-block split and Poisson overlay of sessions"
```

---

### Task 5: Session state replay and ground truth

**Files:**
- Create: `src/atfm/board/__init__.py`, `src/atfm/board/state.py`, `src/atfm/board/replay.py`
- Test: `tests/board/test_replay.py`

**Interfaces:**
- Produces:
  - `SessionState` dataclass: `session_id, cls, tenant, parent_session_id, phase: Literal["tool_running","llm_pending","llm_running","ended"], turn_index, tool_name, backend_id, t_tool_start, progress: list[dict], data: list[dict], ctx_tokens: int, tool_history: list[tuple[str, float]]` and method `elapsed(now) -> float`.
  - `FleetReplayer(table: TraceTable)` with `states_at(t: float) -> list[SessionState]` (only sessions that have started and not ended; a session ends at `t_last_token` of a row with `tool_name is None`, or at the row's `t_tool_end` if the last row has a tool but no next row), and `demand_truth(t: float, horizons: list[float], block_size: int = 16) -> dict` returning `{"kv_blocks": {cls: (H,)}, "prefill_tokens": {cls: (H,)}, "endogenous_kv_blocks": {cls: (H,)}}` where a call is endogenous if its session was active at `t` (started before `t` and not ended).
  - Phase rules at time t for a row r of a session: `t_request <= t < t_last_token` (or `t_request <= t < t_first_token` when first token None) means `llm_running`; `t_tool_start <= t < t_tool_end` means `tool_running`; between `t_last_token` and `t_tool_start` (rare gap) means `tool_running` with the row's tool; between `t_tool_end` and the next row's `t_request` means `llm_pending`.
  - `ctx_tokens` = `isl + osl` of the most recent row with `t_request <= t`. `tool_history` = list of `(tool_name, t_tool_end - t_tool_start)` for completed tool phases before t in this session.

- [ ] **Step 1: Write the failing tests**

```python
# tests/board/test_replay.py
import numpy as np
from atfm.schema.trace import TraceRow, TraceTable
from atfm.board.replay import FleetReplayer

def _rows():
    a = [TraceRow(session_id="a", cls="interactive", tenant="t", turn_index=0, t_request=0.0, t_first_token=1.0,
                  t_last_token=2.0, isl=160, osl=16, tool_name="pytest", backend_id="ci", t_tool_start=2.0, t_tool_end=62.0, source="test"),
         TraceRow(session_id="a", cls="interactive", tenant="t", turn_index=1, t_request=63.0, t_first_token=64.0,
                  t_last_token=65.0, isl=320, osl=16, tool_name=None, source="test")]
    b = [TraceRow(session_id="b", cls="background", tenant="t", turn_index=0, t_request=30.0, t_first_token=31.0,
                  t_last_token=32.0, isl=800, osl=8, tool_name="build", backend_id="ci", t_tool_start=32.0, t_tool_end=200.0, source="test")]
    return TraceTable.from_rows(a + b)

def test_phases():
    rep = FleetReplayer(_rows())
    s = {x.session_id: x for x in rep.states_at(1.5)}
    assert s["a"].phase == "llm_running" and "b" not in s
    s = {x.session_id: x for x in rep.states_at(40.0)}
    assert s["a"].phase == "tool_running" and s["a"].tool_name == "pytest" and abs(s["a"].elapsed(40.0) - 38.0) < 1e-9
    assert s["b"].phase == "tool_running" and s["b"].ctx_tokens == 808
    s = {x.session_id: x for x in rep.states_at(62.5)}
    assert s["a"].phase == "llm_pending"
    s = {x.session_id: x for x in rep.states_at(64.0)}
    assert s["a"].tool_history == [("pytest", 60.0)] and s["a"].ctx_tokens == 336
    assert all(x.session_id != "a" for x in rep.states_at(70.0))
    assert all(x.session_id != "b" for x in rep.states_at(201.0))

def test_demand_truth():
    rep = FleetReplayer(_rows())
    d = rep.demand_truth(10.0, [30.0, 60.0], block_size=16)
    # within (10,40]: b's first call at 30 -> exogenous (b not active at t=10)
    assert d["kv_blocks"]["background"].tolist() == [50, 50]
    assert d["endogenous_kv_blocks"]["background"].tolist() == [0, 0]
    # within (10,70]: a's second call at 63 (isl 320 -> 20 blocks), endogenous
    assert d["kv_blocks"]["interactive"].tolist() == [0, 20]
    assert d["endogenous_kv_blocks"]["interactive"].tolist() == [0, 20]
    assert d["prefill_tokens"]["interactive"].tolist() == [0, 320]

def test_one_shot_session_ends():
    t = TraceTable.from_rows([TraceRow(session_id="q", cls="interactive", tenant="t", turn_index=0, t_request=0.0,
                                       t_first_token=1.0, t_last_token=3.0, isl=16, osl=1, tool_name=None, source="test")])
    rep = FleetReplayer(t)
    assert rep.states_at(2.0)[0].phase == "llm_running" and rep.states_at(3.5) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/board/test_replay.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/board/__init__.py
```

```python
# src/atfm/board/state.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

Phase = Literal["tool_running", "llm_pending", "llm_running", "ended"]

@dataclass
class SessionState:
    session_id: str
    cls: str
    tenant: str
    parent_session_id: str | None
    phase: Phase
    turn_index: int
    tool_name: str | None = None
    backend_id: str | None = None
    t_tool_start: float | None = None
    progress: list[dict] = field(default_factory=list)
    data: list[dict] = field(default_factory=list)
    ctx_tokens: int = 0
    tool_history: list[tuple[str, float]] = field(default_factory=list)
    t_phase_start: float = 0.0

    def elapsed(self, now: float) -> float:
        if self.phase == "tool_running" and self.t_tool_start is not None:
            return max(0.0, now - self.t_tool_start)
        return max(0.0, now - self.t_phase_start)
```

```python
# src/atfm/board/replay.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from atfm.schema.trace import TraceTable
from .state import SessionState

class _Session:
    __slots__ = ("sid", "cls", "tenant", "parent", "rows", "t_start", "t_end")
    def __init__(self, sid: str, g: pd.DataFrame):
        self.sid = sid
        self.cls = g["class"].iloc[0]
        self.tenant = g["tenant"].iloc[0]
        self.parent = g["parent_session_id"].iloc[0]
        self.rows = g.to_dict("records")
        self.t_start = float(self.rows[0]["t_request"])
        last = self.rows[-1]
        if last["tool_name"] is None or _isnan(last["t_tool_end"]):
            self.t_end = _last_llm_time(last)
        else:
            self.t_end = float(last["t_tool_end"])

def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))

def _last_llm_time(row: dict) -> float:
    for k in ("t_last_token", "t_first_token", "t_request"):
        if not _isnan(row[k]):
            return float(row[k])
    return float(row["t_request"])

class FleetReplayer:
    def __init__(self, table: TraceTable):
        self.table = table
        self.sessions = [_Session(sid, g) for sid, g in table.sessions()]
        df = table.df
        self._t_req = df["t_request"].to_numpy(float)
        order = np.argsort(self._t_req, kind="stable")
        self._t_req_sorted = self._t_req[order]
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
        ctx = 0
        for i, r in enumerate(s.rows):
            if t < r["t_request"]:
                break
            ctx = int(r["isl"]) + int(r["osl"])
            t_llm_end = _last_llm_time(r)
            base = dict(session_id=s.sid, cls=s.cls, tenant=s.tenant, parent_session_id=s.parent,
                        turn_index=int(r["turn_index"]), ctx_tokens=ctx)
            if t < t_llm_end:
                return SessionState(phase="llm_running", t_phase_start=float(r["t_request"]),
                                    tool_history=list(history), **base)
            has_tool = r["tool_name"] is not None and not _isnan(r["t_tool_start"])
            if has_tool:
                ts, te = float(r["t_tool_start"]), float(r["t_tool_end"])
                if t < te:
                    prog = [e for e in (r["progress_events"] or []) if e["t"] <= t]
                    data = [e for e in (r["data_events"] or []) if e["t"] <= t]
                    return SessionState(phase="tool_running", tool_name=r["tool_name"], backend_id=r["backend_id"],
                                        t_tool_start=min(ts, t), progress=prog, data=data, t_phase_start=ts,
                                        tool_history=list(history), **base)
                history.append((r["tool_name"], max(0.0, te - ts)))
                nxt = s.rows[i + 1] if i + 1 < len(s.rows) else None
                if nxt is None or t < nxt["t_request"]:
                    return SessionState(phase="llm_pending", t_phase_start=te, tool_history=list(history), **base)
            else:
                return None
        return None

    def demand_truth(self, t: float, horizons: list[float], block_size: int = 16) -> dict:
        H = len(horizons)
        classes = ("interactive", "background")
        kv = {c: np.zeros(H) for c in classes}
        pf = {c: np.zeros(H) for c in classes}
        endo = {c: np.zeros(H) for c in classes}
        lo = np.searchsorted(self._t_req_sorted, t, side="right")
        hi = np.searchsorted(self._t_req_sorted, t + max(horizons), side="right")
        for j in range(lo, hi):
            tr, isl, c, sid = self._t_req_sorted[j], self._isl_sorted[j], self._cls_sorted[j], self._sid_sorted[j]
            blocks = math.ceil(isl / block_size)
            active = self._start[sid] <= t < self._end[sid]
            for k, h in enumerate(horizons):
                if tr <= t + h:
                    kv[c][k] += blocks
                    pf[c][k] += isl
                    if active:
                        endo[c][k] += blocks
        return {"kv_blocks": kv, "prefill_tokens": pf, "endogenous_kv_blocks": endo}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/board/test_replay.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/board tests/board
git commit -m "feat: fleet replayer with session states and demand ground truth"
```

---

### Task 6: Predictor interface and rungs B0, B1, B2, M1

**Files:**
- Create: `src/atfm/board/predictors/__init__.py`, `base.py`, `series.py`, `duration.py`
- Test: `tests/board/test_predictors.py`

**Interfaces:**
- Produces:
  - `SessionPredictor` ABC: `name: str`; `fit(train: TraceTable) -> None`; `resumption(s: SessionState, now: float, n: int, rng) -> np.ndarray` (n draws, seconds, `np.inf` allowed); `next_call_isl(s, n, rng) -> np.ndarray` (int draws); `spawn(s, horizon, n, rng) -> np.ndarray` (int draws).
  - `SeriesPredictor` ABC: `name`; `predict(history: np.ndarray, n: int, rng) -> np.ndarray` where `history` is the past demand series for one (target, class, horizon) at the tick interval; returns n samples for the next value.
  - `ConstantSeries` (B0): returns `history[-1]` repeated n times (0 if empty).
  - `KalmanSeries` (B1): local linear trend Kalman filter with fixed variances `q_level=1.0, q_trend=0.1, r=10.0` (scaled by history variance), returns Gaussian samples from the one-step-ahead predictive distribution clipped at 0.
  - `DurationModel`: per-tool empirical durations from training rows (`t_tool_end - t_tool_start` for rows with a tool), with `sample(tool, n, rng)` (unconditional) and `sample_conditional(tool, elapsed, n, rng)` (from durations `> elapsed`; if fewer than 5 such durations exist, sample from a log-normal fitted to that tool's durations, truncated at `elapsed` by rejection with a fallback of `elapsed * lognormal(0, 0.5)` tail extrapolation), plus `no_return_prob(tool) -> float` = fraction of training rows with that tool that were the last row of their session, and `overhead(tool) -> float` = median gap between `t_tool_end` and the next row's `t_request`. Unknown tools use the pooled distribution over all tools.
  - `HistoryPredictor` (B2): `resumption = max(D - elapsed, 0) + overhead` with D unconditional; `next_call_isl` = `ctx_tokens + delta` where delta is drawn from the empirical per-tool increase in isl between consecutive rows; `spawn` = Poisson with rate `spawn_rate(tool) * min(horizon, E[D])/E[D]` where `spawn_rate(tool)` = mean `spawned_children` for that tool.
  - `SurvivalPredictor` (M1): identical to B2 except `resumption = D_cond - elapsed + overhead` with `D_cond` from `sample_conditional`, and with probability `no_return_prob` the sample is `np.inf`.
  - For sessions in `llm_running` or `llm_pending`, every session predictor returns resumption samples of 0 (the call is due now).

- [ ] **Step 1: Write the failing tests**

```python
# tests/board/test_predictors.py
import numpy as np
from atfm.schema.trace import TraceRow, TraceTable
from atfm.board.state import SessionState
from atfm.board.predictors import ConstantSeries, KalmanSeries, DurationModel, HistoryPredictor, SurvivalPredictor

def _train():
    rows = []
    t = 0.0
    for k in range(40):
        d = 10.0 + k  # pytest durations 10..49
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=0, t_request=t,
                             t_first_token=t + 1, t_last_token=t + 2, isl=100, osl=10, tool_name="pytest",
                             t_tool_start=t + 2, t_tool_end=t + 2 + d, source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=1, t_request=t + 2 + d + 1.0,
                             t_first_token=t + 4 + d, t_last_token=t + 5 + d, isl=150, osl=10, tool_name=None, source="test"))
        t += 1000.0
    return TraceTable.from_rows(rows)

def _state(elapsed, phase="tool_running"):
    return SessionState(session_id="x", cls="background", tenant="t", parent_session_id=None, phase=phase,
                        turn_index=0, tool_name="pytest", backend_id="ci", t_tool_start=100.0 - elapsed,
                        ctx_tokens=110, t_phase_start=100.0 - elapsed)

def test_series_constant_and_kalman():
    rng = np.random.default_rng(0)
    assert np.all(ConstantSeries().predict(np.array([3.0, 5.0]), 4, rng) == 5.0)
    assert np.all(ConstantSeries().predict(np.array([]), 4, rng) == 0.0)
    k = KalmanSeries()
    s = k.predict(np.arange(1.0, 51.0), 2000, rng)
    assert 45.0 < s.mean() < 56.0 and s.min() >= 0.0

def test_duration_model_conditional():
    dm = DurationModel().fit(_train())
    rng = np.random.default_rng(1)
    unc = dm.sample("pytest", 2000, rng)
    assert 25.0 < unc.mean() < 35.0
    cond = dm.sample_conditional("pytest", 40.0, 2000, rng)
    assert cond.min() > 40.0 and cond.max() <= 49.0 + 1e-9
    tail = dm.sample_conditional("pytest", 500.0, 200, rng)
    assert tail.min() > 500.0 and np.isfinite(tail).all()
    assert dm.no_return_prob("pytest") == 0.0
    assert abs(dm.overhead("pytest") - 1.0) < 1e-9
    assert dm.sample("unknown_tool", 10, rng).shape == (10,)

def test_b2_vs_m1():
    tr = _train()
    b2, m1 = HistoryPredictor(), SurvivalPredictor()
    b2.fit(tr); m1.fit(tr)
    rng = np.random.default_rng(2)
    s = _state(elapsed=45.0)
    r2 = b2.resumption(s, 100.0, 2000, rng)
    r1 = m1.resumption(s, 100.0, 2000, rng)
    # B2 ignores elapsed: most draws are already "due" (D < 45 -> 0 + overhead)
    assert (r2 <= 1.0 + 1e-9).mean() > 0.8
    # M1 conditions: all draws in (0, 4] + overhead
    assert r1.min() > 1.0 and r1.max() <= 5.0 + 1e-9
    assert np.all(m1.resumption(_state(0.0, phase="llm_pending"), 100.0, 5, rng) == 0.0)
    isl = m1.next_call_isl(s, 100, rng)
    assert isl.min() >= 110 and isl.dtype.kind == "i"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/board/test_predictors.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/board/predictors/__init__.py
from .base import SessionPredictor, SeriesPredictor
from .series import ConstantSeries, KalmanSeries
from .duration import DurationModel, HistoryPredictor, SurvivalPredictor
__all__ = ["SessionPredictor", "SeriesPredictor", "ConstantSeries", "KalmanSeries",
           "DurationModel", "HistoryPredictor", "SurvivalPredictor"]
```

```python
# src/atfm/board/predictors/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from atfm.schema.trace import TraceTable
from atfm.board.state import SessionState

class SessionPredictor(ABC):
    name: str = "base"
    @abstractmethod
    def fit(self, train: TraceTable) -> "SessionPredictor": ...
    @abstractmethod
    def resumption(self, s: SessionState, now: float, n: int, rng: np.random.Generator) -> np.ndarray: ...
    @abstractmethod
    def next_call_isl(self, s: SessionState, n: int, rng: np.random.Generator) -> np.ndarray: ...
    @abstractmethod
    def spawn(self, s: SessionState, horizon: float, n: int, rng: np.random.Generator) -> np.ndarray: ...

class SeriesPredictor(ABC):
    name: str = "series"
    @abstractmethod
    def predict(self, history: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray: ...
```

```python
# src/atfm/board/predictors/series.py
from __future__ import annotations
import numpy as np
from .base import SeriesPredictor

class ConstantSeries(SeriesPredictor):
    name = "B0_constant"
    def predict(self, history: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        last = float(history[-1]) if len(history) else 0.0
        return np.full(n, last)

class KalmanSeries(SeriesPredictor):
    """Local linear trend model; one-step-ahead predictive Gaussian, clipped at 0."""
    name = "B1_kalman"
    def __init__(self, q_level: float = 1.0, q_trend: float = 0.1, r: float = 10.0, min_points: int = 5):
        self.q_level, self.q_trend, self.r, self.min_points = q_level, q_trend, r, min_points

    def predict(self, history: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        y = np.asarray(history, float)
        if len(y) < self.min_points:
            last = float(y[-1]) if len(y) else 0.0
            return np.full(n, last)
        scale = max(float(np.var(y)), 1e-6)
        F = np.array([[1.0, 1.0], [0.0, 1.0]]); Hm = np.array([[1.0, 0.0]])
        Q = np.diag([self.q_level, self.q_trend]) * scale * 0.01; R = np.array([[self.r]]) * scale * 0.1
        x = np.array([y[0], 0.0]); P = np.eye(2) * scale
        for obs in y:
            x = F @ x; P = F @ P @ F.T + Q
            S = Hm @ P @ Hm.T + R; K = P @ Hm.T / S
            x = x + (K * (obs - Hm @ x)).ravel(); P = (np.eye(2) - K @ Hm) @ P
        x = F @ x; P = F @ P @ F.T + Q
        mean = float(x[0]); var = float((Hm @ P @ Hm.T + R)[0, 0])
        return np.clip(rng.normal(mean, np.sqrt(var), size=n), 0.0, None)
```

```python
# src/atfm/board/predictors/duration.py
from __future__ import annotations
from collections import defaultdict
import numpy as np
from atfm.schema.trace import TraceTable
from atfm.board.state import SessionState
from .base import SessionPredictor

POOLED = "__pooled__"

class DurationModel:
    def __init__(self, min_conditional: int = 5):
        self.min_conditional = min_conditional
        self.durations: dict[str, np.ndarray] = {}
        self.lognorm: dict[str, tuple[float, float]] = {}
        self._no_return: dict[str, float] = {}
        self._overhead: dict[str, float] = {}
        self._isl_delta: dict[str, np.ndarray] = {}
        self._spawn_rate: dict[str, float] = {}

    def fit(self, train: TraceTable) -> "DurationModel":
        durs, last, gaps, deltas, spawns = (defaultdict(list) for _ in range(5))
        for sid, g in train.sessions():
            rows = g.to_dict("records")
            for i, r in enumerate(rows):
                tool = r["tool_name"]
                if tool is None or r["t_tool_start"] is None or np.isnan(r["t_tool_start"]):
                    continue
                d = max(float(r["t_tool_end"] - r["t_tool_start"]), 1e-3)
                durs[tool].append(d); durs[POOLED].append(d)
                is_last = i + 1 >= len(rows)
                last[tool].append(1.0 if is_last else 0.0); last[POOLED].append(1.0 if is_last else 0.0)
                spawns[tool].append(float(r["spawned_children"])); spawns[POOLED].append(float(r["spawned_children"]))
                if not is_last:
                    nxt = rows[i + 1]
                    gap = max(0.0, float(nxt["t_request"] - r["t_tool_end"]))
                    gaps[tool].append(gap); gaps[POOLED].append(gap)
                    dl = max(0, int(nxt["isl"]) - (int(r["isl"]) + int(r["osl"])))
                    deltas[tool].append(dl); deltas[POOLED].append(dl)
        for tool, ds in durs.items():
            arr = np.sort(np.asarray(ds, float))
            self.durations[tool] = arr
            logs = np.log(arr)
            self.lognorm[tool] = (float(logs.mean()), float(max(logs.std(), 0.1)))
            self._no_return[tool] = float(np.mean(last[tool]))
            self._overhead[tool] = float(np.median(gaps[tool])) if gaps[tool] else 0.0
            self._isl_delta[tool] = np.asarray(deltas[tool] or [0], int)
            self._spawn_rate[tool] = float(np.mean(spawns[tool]))
        if POOLED not in self.durations:
            self.durations[POOLED] = np.array([1.0]); self.lognorm[POOLED] = (0.0, 0.5)
            self._no_return[POOLED] = 0.0; self._overhead[POOLED] = 0.0
            self._isl_delta[POOLED] = np.array([0]); self._spawn_rate[POOLED] = 0.0
        return self

    def _key(self, tool: str | None) -> str:
        return tool if tool in self.durations else POOLED

    def sample(self, tool: str | None, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.durations[self._key(tool)], size=n, replace=True)

    def sample_conditional(self, tool: str | None, elapsed: float, n: int, rng: np.random.Generator) -> np.ndarray:
        k = self._key(tool)
        arr = self.durations[k]
        tail = arr[np.searchsorted(arr, elapsed, side="right"):]
        if len(tail) >= self.min_conditional:
            return rng.choice(tail, size=n, replace=True)
        mu, sigma = self.lognorm[k]
        out = np.empty(n); filled = 0
        for _ in range(20):
            cand = rng.lognormal(mu, sigma, size=4 * n)
            cand = cand[cand > elapsed]
            take = min(len(cand), n - filled)
            out[filled:filled + take] = cand[:take]; filled += take
            if filled >= n:
                break
        if filled < n:
            out[filled:] = elapsed * rng.lognormal(0.0, 0.5, size=n - filled) + elapsed
        return out

    def mean(self, tool: str | None) -> float:
        return float(self.durations[self._key(tool)].mean())

    def no_return_prob(self, tool: str | None) -> float:
        return self._no_return[self._key(tool)]

    def overhead(self, tool: str | None) -> float:
        return self._overhead[self._key(tool)]

    def isl_delta(self, tool: str | None, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self._isl_delta[self._key(tool)], size=n, replace=True)

    def spawn_rate(self, tool: str | None) -> float:
        return self._spawn_rate[self._key(tool)]

class HistoryPredictor(SessionPredictor):
    """B2: per-tool historical durations, elapsed time ignored (Continuum-style TTL from start)."""
    name = "B2_history"
    def __init__(self):
        self.dm = DurationModel()

    def fit(self, train: TraceTable) -> "HistoryPredictor":
        self.dm.fit(train); return self

    def _draw_duration(self, s: SessionState, now: float, n: int, rng) -> np.ndarray:
        return self.dm.sample(s.tool_name, n, rng)

    def resumption(self, s: SessionState, now: float, n: int, rng: np.random.Generator) -> np.ndarray:
        if s.phase != "tool_running":
            return np.zeros(n)
        d = self._draw_duration(s, now, n, rng)
        r = np.maximum(d - s.elapsed(now), 0.0) + self.dm.overhead(s.tool_name)
        p = self.dm.no_return_prob(s.tool_name)
        if p > 0:
            r = np.where(rng.random(n) < p, np.inf, r)
        return r

    def next_call_isl(self, s: SessionState, n: int, rng: np.random.Generator) -> np.ndarray:
        return (s.ctx_tokens + self.dm.isl_delta(s.tool_name, n, rng)).astype(int)

    def spawn(self, s: SessionState, horizon: float, n: int, rng: np.random.Generator) -> np.ndarray:
        rate = self.dm.spawn_rate(s.tool_name)
        if rate <= 0:
            return np.zeros(n, int)
        frac = min(1.0, horizon / max(self.dm.mean(s.tool_name), 1e-6))
        return rng.poisson(rate * frac, size=n)

class SurvivalPredictor(HistoryPredictor):
    """M1: B2 plus conditioning on elapsed time (survival)."""
    name = "M1_survival"
    def _draw_duration(self, s: SessionState, now: float, n: int, rng) -> np.ndarray:
        return self.dm.sample_conditional(s.tool_name, s.elapsed(now), n, rng)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/board/test_predictors.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/board/predictors tests/board/test_predictors.py
git commit -m "feat: predictor interfaces with B0, B1, B2 and M1 rungs"
```

---

### Task 7: M2 progress filter and M3 backend factor

**Files:**
- Create: `src/atfm/board/predictors/progress.py`, `src/atfm/board/predictors/backend.py`
- Modify: `src/atfm/board/predictors/__init__.py` (export `ProgressPredictor`, `BackendPredictor`, `BackendFactor`)
- Test: `tests/board/test_progress.py`

**Interfaces:**
- Produces:
  - `rate_posterior(progress: list[dict], t_start: float, now: float, prior_shape=2.0, prior_rate_per_unit=None) -> tuple[float, float]`: Gamma posterior (shape, rate) over work rate in units per second, updated from consecutive progress increments `(Δcompleted, Δt)`; with no increments returns the prior.
  - `ProgressPredictor(SessionPredictor)` (M2): extends `SurvivalPredictor`. If the session has at least one progress event with `total` known and `completed < total`: draw rate `r ~ Gamma(shape, rate)`, remaining `= (total - completed_latest)/r + end_phase_residual` where the residual is drawn from the per-tool empirical distribution of `(t_tool_end - t_of_last_progress_event)` in training (0 if unavailable); resumption `= max(remaining - (now - t_latest_event), 0) + overhead`. If `completed >= total`, remaining is the residual only. Data events: if any `metric == "early_stop_prob"` the latest value `p` truncates: with probability p the resumption is `min(sample, overhead)`. Without usable progress events, behaves exactly as M1.
  - `BackendFactor`: keeps per-backend log-speed `z_b` with `update(backend_id, observed_ratio: float)` where `observed_ratio` = (actual elapsed / expected elapsed) from completed tools on that backend within the last window; exposes `draw(backend_id, n, rng) -> np.ndarray` multiplicative factors `exp(N(mu_b, sigma_b))` with `mu_b` the EWMA of log ratios (alpha 0.3) and `sigma_b` the EWMA std (floor 0.05); unknown backend returns ones.
  - `BackendPredictor` (M3): extends `ProgressPredictor`; `resumption` multiplies the remaining-time part by a factor drawn from `BackendFactor` for `s.backend_id`; the forecaster shares one draw per backend across sessions in the same Monte Carlo replicate by passing `factors: dict[str, np.ndarray]` through `resumption(..., factors=...)` (keyword-only, default None, then draws independently). `fit` learns nothing extra; `observe_completion(backend_id, tool, actual_duration)` updates the factor with `actual / dm.mean(tool)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/board/test_progress.py
import numpy as np
from atfm.schema.trace import TraceRow, TraceTable
from atfm.board.state import SessionState
from atfm.board.predictors import ProgressPredictor, BackendPredictor, BackendFactor
from atfm.board.predictors.progress import rate_posterior

def _train():
    rows = []
    t = 0.0
    for k in range(30):
        d = 100.0
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=0, t_request=t,
                             t_first_token=t + 1, t_last_token=t + 2, isl=100, osl=10, tool_name="pytest",
                             t_tool_start=t + 2, t_tool_end=t + 2 + d, backend_id="ci",
                             progress_events=[{"t": t + 2 + 10 * j, "completed": 10 * j, "total": 100, "phase": "run"} for j in range(1, 10)],
                             source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=1, t_request=t + 2 + d,
                             t_first_token=t + 3 + d, t_last_token=t + 4 + d, isl=150, osl=10, tool_name=None, source="test"))
        t += 1000.0
    return TraceTable.from_rows(rows)

def test_rate_posterior_concentrates():
    prog = [{"t": 10.0 * j, "completed": 5.0 * j, "total": 100.0, "phase": "run"} for j in range(1, 6)]
    shape, rate = rate_posterior(prog, t_start=0.0, now=50.0)
    assert abs(shape / rate - 0.5) < 0.1 and shape > 5

def test_m2_uses_progress_m1_fallback():
    tr = _train()
    m2 = ProgressPredictor().fit(tr)
    rng = np.random.default_rng(0)
    # slow session: at now=52 it has completed 20/100 at rate 0.4/s -> remaining ~ 200 s (M1 would say ~48 s)
    s = SessionState(session_id="x", cls="background", tenant="t", parent_session_id=None, phase="tool_running",
                     turn_index=0, tool_name="pytest", backend_id="ci", t_tool_start=0.0, ctx_tokens=110, t_phase_start=0.0,
                     progress=[{"t": 25.0, "completed": 10, "total": 100, "phase": "run"},
                               {"t": 50.0, "completed": 20, "total": 100, "phase": "run"}])
    r = m2.resumption(s, 52.0, 2000, rng)
    assert 120.0 < np.median(r) < 320.0
    s.progress = []
    r1 = m2.resumption(s, 52.0, 2000, rng)
    assert 40.0 < np.median(r1) < 60.0

def test_backend_factor_and_m3():
    bf = BackendFactor()
    assert np.all(bf.draw("nope", 3, np.random.default_rng(0)) == 1.0)
    for _ in range(10):
        bf.update("ci", 2.0)
    f = bf.draw("ci", 1000, np.random.default_rng(0))
    assert 1.7 < np.median(f) < 2.3
    m3 = BackendPredictor().fit(_train())
    for _ in range(10):
        m3.observe_completion("ci", "pytest", 200.0)
    s = SessionState(session_id="x", cls="background", tenant="t", parent_session_id=None, phase="tool_running",
                     turn_index=0, tool_name="pytest", backend_id="ci", t_tool_start=0.0, ctx_tokens=110, t_phase_start=0.0)
    r = m3.resumption(s, 10.0, 2000, np.random.default_rng(1))
    assert np.median(r) > 120.0
    shared = {"ci": np.full(5, 3.0)}
    r_shared = m3.resumption(s, 10.0, 5, np.random.default_rng(1), factors=shared)
    assert np.all(r_shared > 200.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/board/test_progress.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement**

```python
# src/atfm/board/predictors/progress.py
from __future__ import annotations
from collections import defaultdict
import numpy as np
from atfm.schema.trace import TraceTable
from atfm.board.state import SessionState
from .duration import SurvivalPredictor, POOLED

def rate_posterior(progress: list[dict], t_start: float, now: float, prior_shape: float = 2.0,
                   prior_rate_per_unit: float | None = None) -> tuple[float, float]:
    """Gamma(shape, rate) posterior over work rate (units/s). Increments are (dw, dt) pairs."""
    pts = [(t_start, 0.0)] + [(float(e["t"]), float(e["completed"])) for e in progress if e.get("completed") is not None]
    pts.sort()
    dw = np.diff([p[1] for p in pts]); dt = np.diff([p[0] for p in pts])
    mask = dt > 0
    dw, dt = dw[mask], dt[mask]
    if prior_rate_per_unit is None:
        prior_rate_per_unit = prior_shape * max(now - t_start, 1.0) / max(pts[-1][1], 1.0) if len(pts) > 1 else prior_shape
    shape = prior_shape + float(dw.sum())
    rate = prior_rate_per_unit + float(dt.sum())
    return shape, rate

class ProgressPredictor(SurvivalPredictor):
    """M2: M1 plus a Bayesian rate filter over live progress events."""
    name = "M2_progress"
    def __init__(self):
        super().__init__()
        self._residual: dict[str, np.ndarray] = {}

    def fit(self, train: TraceTable) -> "ProgressPredictor":
        super().fit(train)
        res = defaultdict(list)
        for _, g in train.sessions():
            for r in g.to_dict("records"):
                ev = r["progress_events"] or []
                if r["tool_name"] is None or not ev or r["t_tool_end"] is None or np.isnan(r["t_tool_end"]):
                    continue
                last_t = max(e["t"] for e in ev)
                res[r["tool_name"]].append(max(0.0, float(r["t_tool_end"]) - last_t))
                res[POOLED].append(max(0.0, float(r["t_tool_end"]) - last_t))
        self._residual = {k: np.asarray(v) for k, v in res.items()}
        return self

    def _residual_draw(self, tool: str | None, n: int, rng) -> np.ndarray:
        arr = self._residual.get(tool, self._residual.get(POOLED))
        if arr is None or len(arr) == 0:
            return np.zeros(n)
        return rng.choice(arr, size=n, replace=True)

    def _remaining_from_progress(self, s: SessionState, now: float, n: int, rng) -> np.ndarray | None:
        usable = [e for e in s.progress if e.get("total") not in (None, 0) and e.get("completed") is not None]
        if not usable:
            return None
        latest = max(usable, key=lambda e: e["t"])
        total, done, t_latest = float(latest["total"]), float(latest["completed"]), float(latest["t"])
        shape, rate = rate_posterior(usable, s.t_tool_start if s.t_tool_start is not None else s.t_phase_start, now)
        r = rng.gamma(shape, 1.0 / rate, size=n)
        work_left = max(total - done, 0.0)
        remaining = work_left / np.maximum(r, 1e-9) + self._residual_draw(s.tool_name, n, rng)
        remaining = np.maximum(remaining - (now - t_latest), 0.0)
        for e in s.data:
            if e.get("metric") == "early_stop_prob":
                p = float(e["value"])
                remaining = np.where(rng.random(n) < p, 0.0, remaining)
        return remaining

    def resumption(self, s: SessionState, now: float, n: int, rng: np.random.Generator, *,
                   factors: dict[str, np.ndarray] | None = None) -> np.ndarray:
        if s.phase != "tool_running":
            return np.zeros(n)
        rem = self._remaining_from_progress(s, now, n, rng)
        if rem is None:
            d = self._draw_duration(s, now, n, rng)
            rem = np.maximum(d - s.elapsed(now), 0.0)
        rem = self._scale(s, rem, n, rng, factors)
        r = rem + self.dm.overhead(s.tool_name)
        p = self.dm.no_return_prob(s.tool_name)
        if p > 0:
            r = np.where(rng.random(n) < p, np.inf, r)
        return r

    def _scale(self, s, rem, n, rng, factors):
        return rem
```

```python
# src/atfm/board/predictors/backend.py
from __future__ import annotations
import numpy as np
from .progress import ProgressPredictor

class BackendFactor:
    def __init__(self, alpha: float = 0.3, sigma_floor: float = 0.05):
        self.alpha, self.sigma_floor = alpha, sigma_floor
        self.mu: dict[str, float] = {}
        self.var: dict[str, float] = {}

    def update(self, backend_id: str, observed_ratio: float) -> None:
        z = float(np.log(max(observed_ratio, 1e-6)))
        if backend_id not in self.mu:
            self.mu[backend_id] = z; self.var[backend_id] = self.sigma_floor ** 2
            return
        mu = self.mu[backend_id]
        self.mu[backend_id] = (1 - self.alpha) * mu + self.alpha * z
        self.var[backend_id] = (1 - self.alpha) * self.var[backend_id] + self.alpha * (z - mu) ** 2

    def draw(self, backend_id: str | None, n: int, rng: np.random.Generator) -> np.ndarray:
        if backend_id not in self.mu:
            return np.ones(n)
        sigma = max(np.sqrt(self.var[backend_id]), self.sigma_floor)
        return np.exp(rng.normal(self.mu[backend_id], sigma, size=n))

class BackendPredictor(ProgressPredictor):
    """M3: M2 plus a shared latent speed factor per backend."""
    name = "M3_backend"
    def __init__(self):
        super().__init__()
        self.factor = BackendFactor()

    def observe_completion(self, backend_id: str | None, tool: str | None, actual_duration: float) -> None:
        if backend_id is None:
            return
        self.factor.update(backend_id, actual_duration / max(self.dm.mean(tool), 1e-6))

    def _scale(self, s, rem, n, rng, factors):
        if factors is not None and s.backend_id in factors:
            f = factors[s.backend_id]
        else:
            f = self.factor.draw(s.backend_id, n, rng)
        return rem * f
```

Update `__init__.py`:

```python
# src/atfm/board/predictors/__init__.py
from .base import SessionPredictor, SeriesPredictor
from .series import ConstantSeries, KalmanSeries
from .duration import DurationModel, HistoryPredictor, SurvivalPredictor
from .progress import ProgressPredictor, rate_posterior
from .backend import BackendPredictor, BackendFactor
__all__ = ["SessionPredictor", "SeriesPredictor", "ConstantSeries", "KalmanSeries", "DurationModel",
           "HistoryPredictor", "SurvivalPredictor", "ProgressPredictor", "rate_posterior",
           "BackendPredictor", "BackendFactor"]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/board -v`
Expected: PASS (all board tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/board/predictors tests/board/test_progress.py
git commit -m "feat: M2 progress filter and M3 backend factor predictors"
```

---

### Task 8: Monte Carlo fleet forecaster

**Files:**
- Create: `src/atfm/board/forecaster.py`
- Test: `tests/board/test_forecaster.py`

**Interfaces:**
- Consumes: `SessionPredictor`, `SeriesPredictor`, `SessionState`, `ForecastSnapshot`, `BackendPredictor` (for shared factors).
- Produces:
  - `ExogenousModel` with `fit(train: TraceTable)` learning per-class first-call isl distribution; `update(t, new_session_starts: list[tuple[float, str]])`; `rate(cls, now) -> float` sessions/s over the trailing `window_s=1800`; `draw(cls, horizon, n, rng) -> tuple[np.ndarray, np.ndarray]` (kv_blocks, prefill_tokens) sums for arrivals within the horizon.
  - `SessionForecaster(predictor: SessionPredictor, exo: ExogenousModel, horizons, n=512, block_size=16, spawn_depth=2)` with `forecast(t: float, states: list[SessionState], rng) -> ForecastSnapshot`. Children: for each session and each replicate, draw `k = predictor.spawn(...)`; each child contributes a first call with isl drawn from `exo` first-call distribution at a uniform time within the parent's resumption-to-horizon interval, recursively to `spawn_depth`. If `predictor` is a `BackendPredictor`, one factor draw per backend per replicate is shared via `factors=`.
  - `SeriesForecaster(series: SeriesPredictor, horizons, n=512)` with `observe(t, truth: dict)` (stores per-(target, class, horizon) history) and `forecast(t, states, rng) -> ForecastSnapshot` ignoring `states`.
  - Both set `endogenous_fraction[cls]` = mean over replicates of (in-flight contribution / total), 0 when total is 0.

- [ ] **Step 1: Write the failing tests**

```python
# tests/board/test_forecaster.py
import numpy as np
from atfm.schema.trace import TraceRow, TraceTable
from atfm.board.state import SessionState
from atfm.board.predictors import SurvivalPredictor, ConstantSeries
from atfm.board.forecaster import SessionForecaster, SeriesForecaster, ExogenousModel

def _train():
    rows = []
    t = 0.0
    for k in range(20):
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=0, t_request=t,
                             t_first_token=t + 1, t_last_token=t + 2, isl=160, osl=16, tool_name="pytest",
                             t_tool_start=t + 2, t_tool_end=t + 62, source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=1, t_request=t + 62,
                             t_first_token=t + 63, t_last_token=t + 64, isl=320, osl=16, tool_name=None, source="test"))
        t += 500.0
    return TraceTable.from_rows(rows)

def _state(elapsed):
    return SessionState(session_id=f"x{elapsed}", cls="background", tenant="t", parent_session_id=None, phase="tool_running",
                        turn_index=0, tool_name="pytest", backend_id="ci", t_tool_start=100.0 - elapsed, ctx_tokens=176, t_phase_start=100.0 - elapsed)

def test_session_forecaster_shapes_and_empty():
    tr = _train()
    fc = SessionForecaster(SurvivalPredictor().fit(tr), ExogenousModel().fit(tr), horizons=[10.0, 120.0], n=64)
    snap = fc.forecast(100.0, [], np.random.default_rng(0))
    assert snap.samples["kv_blocks"]["background"].shape == (2, 64)
    assert np.all(snap.samples["kv_blocks"]["background"] == 0) and snap.endogenous_fraction["background"].tolist() == [0.0, 0.0]
    assert np.isfinite(snap.quantiles("kv_blocks", "background", 0.9)).all()

def test_session_forecaster_counts_due_sessions():
    tr = _train()
    fc = SessionForecaster(SurvivalPredictor().fit(tr), ExogenousModel().fit(tr), horizons=[10.0, 120.0], n=256)
    states = [_state(59.0), _state(59.5), _state(1.0)]   # two nearly done (60 s tools), one just started
    snap = fc.forecast(100.0, states, np.random.default_rng(0))
    q50 = snap.quantiles("kv_blocks", "background", 0.5)
    # next isl ~ 320 -> 20 blocks each; at 10 s two are due, at 120 s all three
    assert 35 <= q50[0] <= 45 and 55 <= q50[1] <= 65
    assert snap.endogenous_fraction["background"][1] > 0.99

def test_series_forecaster():
    sf = SeriesForecaster(ConstantSeries(), horizons=[10.0], n=8)
    sf.observe(0.0, {"kv_blocks": {"interactive": np.array([5.0]), "background": np.array([0.0])},
                     "prefill_tokens": {"interactive": np.array([80.0]), "background": np.array([0.0])}})
    snap = sf.forecast(10.0, [], np.random.default_rng(0))
    assert np.all(snap.samples["kv_blocks"]["interactive"] == 5.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/board/test_forecaster.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/board/forecaster.py
from __future__ import annotations
from collections import defaultdict, deque
import math
import numpy as np
from atfm.schema.trace import TraceTable
from atfm.schema.forecast import ForecastSnapshot
from atfm.board.state import SessionState
from atfm.board.predictors.base import SessionPredictor, SeriesPredictor
from atfm.board.predictors.backend import BackendPredictor

CLASSES = ("interactive", "background")
TARGETS = ("kv_blocks", "prefill_tokens")

class ExogenousModel:
    def __init__(self, window_s: float = 1800.0, block_size: int = 16):
        self.window_s, self.block_size = window_s, block_size
        self.first_isl: dict[str, np.ndarray] = {c: np.array([256]) for c in CLASSES}
        self._starts: dict[str, deque] = {c: deque() for c in CLASSES}

    def fit(self, train: TraceTable) -> "ExogenousModel":
        firsts = train.df[train.df["turn_index"] == 0]
        for c in CLASSES:
            arr = firsts.loc[firsts["class"] == c, "isl"].to_numpy(int)
            if len(arr):
                self.first_isl[c] = arr
        return self

    def update(self, t: float, new_session_starts: list[tuple[float, str]]) -> None:
        for ts, c in new_session_starts:
            self._starts[c].append(ts)
        for c in CLASSES:
            while self._starts[c] and self._starts[c][0] < t - self.window_s:
                self._starts[c].popleft()

    def rate(self, cls: str, now: float) -> float:
        return len(self._starts[cls]) / self.window_s

    def draw(self, cls: str, horizon: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        k = rng.poisson(self.rate(cls, 0.0) * horizon, size=n)
        kv = np.zeros(n); pf = np.zeros(n)
        for i in np.nonzero(k)[0]:
            isl = rng.choice(self.first_isl[cls], size=k[i], replace=True)
            kv[i] = np.ceil(isl / self.block_size).sum(); pf[i] = isl.sum()
        return kv, pf

    def first_call_isl(self, cls: str, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.first_isl[cls], size=n, replace=True)

def _empty(horizons, n):
    return {t: {c: np.zeros((len(horizons), n)) for c in CLASSES} for t in TARGETS}

class SessionForecaster:
    def __init__(self, predictor: SessionPredictor, exo: ExogenousModel, horizons: list[float], n: int = 512,
                 block_size: int = 16, spawn_depth: int = 2):
        self.predictor, self.exo, self.horizons, self.n = predictor, exo, list(horizons), n
        self.block_size, self.spawn_depth = block_size, spawn_depth

    def forecast(self, t: float, states: list[SessionState], rng: np.random.Generator) -> ForecastSnapshot:
        H, n = len(self.horizons), self.n
        hz = np.asarray(self.horizons)[:, None]
        samples = _empty(self.horizons, n)
        endo = {c: np.zeros((H, n)) for c in CLASSES}
        factors = None
        if isinstance(self.predictor, BackendPredictor):
            backends = {s.backend_id for s in states if s.backend_id is not None}
            factors = {b: self.predictor.factor.draw(b, n, rng) for b in backends}
        for s in states:
            kw = {"factors": factors} if factors is not None else {}
            R = self.predictor.resumption(s, t, n, rng, **kw)
            isl = self.predictor.next_call_isl(s, n, rng)
            due = (R[None, :] <= hz)                       # (H, n)
            kv = np.ceil(isl / self.block_size)
            samples["kv_blocks"][s.cls] += due * kv[None, :]
            samples["prefill_tokens"][s.cls] += due * isl[None, :]
            endo[s.cls] += due * kv[None, :]
            self._children(s, R, samples, rng, depth=1)
        for c in CLASSES:
            for k, h in enumerate(self.horizons):
                kv, pf = self.exo.draw(c, h, n, rng)
                samples["kv_blocks"][c][k] += kv
                samples["prefill_tokens"][c][k] += pf
        endo_frac = {}
        for c in CLASSES:
            tot = samples["kv_blocks"][c]
            frac = np.where(tot > 0, endo[c] / np.maximum(tot, 1e-12), 0.0)
            endo_frac[c] = frac.mean(axis=1)
        return ForecastSnapshot(t=t, horizons=self.horizons, model_id=self.predictor.name,
                                samples=samples, endogenous_fraction=endo_frac)

    def _children(self, s: SessionState, R: np.ndarray, samples, rng, depth: int) -> None:
        if depth > self.spawn_depth:
            return
        hmax = self.horizons[-1]
        k = self.predictor.spawn(s, hmax, len(R), rng)
        if not k.any():
            return
        hz = np.asarray(self.horizons)[:, None]
        for i in np.nonzero(k)[0]:
            if not np.isfinite(R[i]) or R[i] > hmax:
                continue
            arrival = R[i] + rng.uniform(0.0, max(hmax - R[i], 1e-9), size=k[i])
            isl = self.exo.first_call_isl(s.cls, k[i], rng)
            due = (arrival[None, :] <= hz)
            samples["kv_blocks"][s.cls][:, i] += (due * np.ceil(isl / self.block_size)[None, :]).sum(axis=1)
            samples["prefill_tokens"][s.cls][:, i] += (due * isl[None, :]).sum(axis=1)

class SeriesForecaster:
    def __init__(self, series: SeriesPredictor, horizons: list[float], n: int = 512, max_history: int = 500):
        self.series, self.horizons, self.n = series, list(horizons), n
        self.history: dict[tuple[str, str, int], deque] = defaultdict(lambda: deque(maxlen=max_history))

    def observe(self, t: float, truth: dict) -> None:
        for tgt in TARGETS:
            for c in CLASSES:
                for k in range(len(self.horizons)):
                    self.history[(tgt, c, k)].append(float(truth[tgt][c][k]))

    def forecast(self, t: float, states: list[SessionState], rng: np.random.Generator) -> ForecastSnapshot:
        samples = _empty(self.horizons, self.n)
        for tgt in TARGETS:
            for c in CLASSES:
                for k in range(len(self.horizons)):
                    hist = np.asarray(self.history[(tgt, c, k)])
                    samples[tgt][c][k] = self.series.predict(hist, self.n, rng)
        endo = {c: np.zeros(len(self.horizons)) for c in CLASSES}
        return ForecastSnapshot(t=t, horizons=self.horizons, model_id=self.series.name, samples=samples,
                                endogenous_fraction=endo)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/board/test_forecaster.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/board/forecaster.py tests/board/test_forecaster.py
git commit -m "feat: Monte Carlo fleet forecaster with exogenous arrivals and fan-out"
```

---

### Task 9: Forecast metrics

**Files:**
- Create: `src/atfm/eval/__init__.py`, `src/atfm/eval/forecast.py`
- Test: `tests/eval/test_forecast_metrics.py`

**Interfaces:**
- Produces (all vectorized over the last axis of samples `(..., n)` against truth `(...)`):
  - `crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray`: `E|X - y| - 0.5 E|X - X'|` estimated from samples (uses sorted-sample formula for the second term).
  - `pinball(samples, truth, q: float) -> np.ndarray`: pinball loss of the sample quantile at level q.
  - `coverage(samples, truth, level: float) -> np.ndarray` (bool): truth within the central `level` interval.
  - `surge_events(q_series: np.ndarray, truth_series: np.ndarray, capacity: float, tick_s: float) -> dict` with `lead_times` (seconds between the first forecast crossing and the first truth crossing for each truth surge), `false_alarms` (forecast crossings with no truth crossing within the horizon window `horizon_s`), and `missed` counts. Surges are maximal runs above capacity.
  - `score_run(records: list[dict]) -> pd.DataFrame`: each record is `{"t", "model", "target", "class", "h_index", "h", "samples": np.ndarray(n), "truth": float, "perturbed": bool}`; returns a DataFrame with columns `model, target, class, h, crps, pinball90, pinball95, cov80, cov90, n` aggregated over t, plus the same with `perturbed=True` only (suffix `_pert`), plus `endogenous_fraction` mean if present in records.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_forecast_metrics.py
import numpy as np
from atfm.eval.forecast import crps, pinball, coverage, surge_events, score_run

def test_crps_point_forecast_is_abs_error():
    s = np.full((3, 50), 4.0); y = np.array([4.0, 6.0, 1.0])
    assert np.allclose(crps(s, y), [0.0, 2.0, 3.0])

def test_crps_matches_gaussian_closed_form():
    rng = np.random.default_rng(0)
    s = rng.normal(0.0, 1.0, size=(1, 200000)); y = np.array([0.0])
    expected = 1.0 * (1 / np.sqrt(np.pi))  # CRPS of N(0,1) at 0 = sigma*(1/sqrt(pi)) with z=0: 2*phi(0) - 1/sqrt(pi) = 0.2339
    assert abs(crps(s, y)[0] - (2 * 0.3989422804 - 1 / np.sqrt(np.pi))) < 0.01

def test_pinball_asymmetry():
    s = np.full((1, 100), 10.0)
    under = pinball(s, np.array([19.0]), 0.9)   # truth above forecast by 9 -> 0.9*9
    over = pinball(s, np.array([1.0]), 0.9)     # truth below by 9 -> 0.1*9
    assert abs(under[0] - 8.1) < 1e-9 and abs(over[0] - 0.9) < 1e-9 and under[0] / over[0] == 9.0

def test_coverage():
    s = np.arange(100.0)[None, :].repeat(2, 0)
    assert coverage(s, np.array([50.0, 200.0]), 0.8).tolist() == [True, False]

def test_surge_events():
    truth = np.array([0, 0, 5, 5, 5, 0, 0, 0, 0, 0], float)
    q90 = np.array([0, 3, 5, 5, 0, 0, 0, 3, 3, 0], float)
    ev = surge_events(q90, truth, capacity=2.0, tick_s=10.0, horizon_s=30.0)
    assert ev["lead_times"] == [10.0] and ev["false_alarms"] == 1 and ev["missed"] == 0

def test_score_run_aggregates():
    recs = []
    for t in range(4):
        recs.append({"t": float(t), "model": "m", "target": "kv_blocks", "class": "background", "h_index": 0, "h": 10.0,
                     "samples": np.full(8, 5.0), "truth": 5.0 + t, "perturbed": t >= 2, "endogenous_fraction": 0.5})
    df = score_run(recs)
    row = df.iloc[0]
    assert row["n"] == 4 and abs(row["crps"] - 1.5) < 1e-9 and abs(row["crps_pert"] - 2.5) < 1e-9
    assert abs(row["endogenous_fraction"] - 0.5) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/eval/__init__.py
```

```python
# src/atfm/eval/forecast.py
from __future__ import annotations
import numpy as np
import pandas as pd

def crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    s = np.asarray(samples, float); y = np.asarray(truth, float)[..., None]
    s = np.where(np.isfinite(s), s, np.nanmax(np.where(np.isfinite(s), s, -np.inf), axis=-1, keepdims=True) * 0 + 1e12)
    n = s.shape[-1]
    term1 = np.abs(s - y).mean(axis=-1)
    srt = np.sort(s, axis=-1)
    idx = np.arange(1, n + 1)
    term2 = (2.0 / (n * n)) * ((2 * idx - n - 1) * srt).sum(axis=-1)   # = E|X - X'|
    return term1 - 0.5 * term2

def pinball(samples: np.ndarray, truth: np.ndarray, q: float) -> np.ndarray:
    pred = np.quantile(np.asarray(samples, float), q, axis=-1)
    y = np.asarray(truth, float)
    diff = y - pred
    return np.where(diff >= 0, q * diff, (q - 1) * diff)

def coverage(samples: np.ndarray, truth: np.ndarray, level: float) -> np.ndarray:
    lo = np.quantile(samples, (1 - level) / 2, axis=-1); hi = np.quantile(samples, 1 - (1 - level) / 2, axis=-1)
    y = np.asarray(truth, float)
    return (y >= lo) & (y <= hi)

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out = []; start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out

def surge_events(q_series: np.ndarray, truth_series: np.ndarray, capacity: float, tick_s: float,
                 horizon_s: float = 300.0) -> dict:
    fq = np.asarray(q_series) > capacity; ft = np.asarray(truth_series) > capacity
    truth_runs = _runs(ft); fc_runs = _runs(fq)
    lead, missed = [], 0
    w = int(round(horizon_s / tick_s))
    for (ts, te) in truth_runs:
        cands = [fs for (fs, fe) in fc_runs if ts - w <= fs <= ts]
        if cands:
            lead.append(float((ts - min(cands)) * tick_s))
        else:
            missed += 1
    false_alarms = 0
    for (fs, fe) in fc_runs:
        if not any(fs - w <= ts <= fe + w for (ts, te) in truth_runs):
            false_alarms += 1
    return {"lead_times": lead, "false_alarms": false_alarms, "missed": missed}

def score_run(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        s = r["samples"][None, :]; y = np.array([r["truth"]])
        rows.append({"model": r["model"], "target": r["target"], "class": r["class"], "h": r["h"],
                     "perturbed": bool(r.get("perturbed", False)),
                     "crps": float(crps(s, y)[0]), "pinball90": float(pinball(s, y, 0.9)[0]),
                     "pinball95": float(pinball(s, y, 0.95)[0]),
                     "cov80": float(coverage(s, y, 0.8)[0]), "cov90": float(coverage(s, y, 0.9)[0]),
                     "endogenous_fraction": r.get("endogenous_fraction", np.nan)})
    df = pd.DataFrame(rows)
    keys = ["model", "target", "class", "h"]
    metrics = ["crps", "pinball90", "pinball95", "cov80", "cov90", "endogenous_fraction"]
    agg = df.groupby(keys)[metrics].mean()
    agg["n"] = df.groupby(keys).size()
    pert = df[df["perturbed"]].groupby(keys)[metrics[:5]].mean().add_suffix("_pert")
    return agg.join(pert, how="left").reset_index()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/eval -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/eval tests/eval
git commit -m "feat: forecast metrics CRPS, pinball, coverage, surge lead time"
```

---

### Task 10: Synthetic fleet generator

**Files:**
- Create: `src/atfm/traces/synthetic.py`
- Test: `tests/traces/test_synthetic.py`

**Interfaces:**
- Produces pydantic models `ToolSpec(name, weight, log_mu, log_sigma, signal: Literal["strong","weak","none"], backend_id, spawn_prob=0.0, progress_every_s=10.0)`, `ClassSpec(cls, rate_per_hour, turns_mean, isl0, isl_growth, osl_mean, tools: list[ToolSpec], think_log_mu=None, think_log_sigma=None, deadline_s=None)`, `Perturbation(backend_id, t_start, t_end, factor)`, `WorkloadSpec(duration_s, classes: list[ClassSpec], perturbations: list[Perturbation] = [], tenants=4, llm_prefill_tps=20000.0, llm_decode_tps=60.0, seed=0)`, and `generate(spec: WorkloadSpec) -> TraceTable`.
- Generation rules: sessions arrive as Poisson per class over `[0, duration_s)`; turns per session `1 + Poisson(turns_mean - 1)`; per turn `isl = isl0 + isl_growth * turn + previous osl`, `osl ~ 1 + Poisson(osl_mean)`; LLM time = `isl / prefill_tps + osl / decode_tps`; tool chosen by weight; duration `lognormal(log_mu, log_sigma)` multiplied by the product of active perturbation factors on its backend at the tool's start (row `perturbation_flag=True` when any factor applies); `strong` tools emit progress events every `progress_every_s` with `completed = elapsed/duration * total (total = 100)` under the true (perturbed) rate; `weak` tools emit one event at 90% of the duration with `phase="end"`; `none` emit nothing; interactive sessions end a turn without a tool with probability 0.5 and then think for `lognormal(think_log_mu, think_log_sigma)` (rows use `tool_name="__think__"`, `backend_id="human"`); a tool with `spawn_prob` spawns one child session (same class, `parent_session_id` set, one to three turns) with probability `spawn_prob`, starting at the parent's tool end; `tenant = f"t{k % tenants}"`.
- `tail_share(table: TraceTable, threshold_s=60.0) -> tuple[float, float]` returns (share of tool calls over threshold, share of tool time over threshold).

- [ ] **Step 1: Write the failing tests**

```python
# tests/traces/test_synthetic.py
import numpy as np
from atfm.traces.synthetic import WorkloadSpec, ClassSpec, ToolSpec, Perturbation, generate, tail_share

def _spec(seed=0, long_weight=0.1, perturb=False):
    tools = [ToolSpec(name="bash", weight=1 - long_weight, log_mu=np.log(3.0), log_sigma=0.5, signal="none", backend_id="local"),
             ToolSpec(name="pytest", weight=long_weight, log_mu=np.log(300.0), log_sigma=0.6, signal="strong", backend_id="ci", spawn_prob=0.1)]
    perts = [Perturbation(backend_id="ci", t_start=1000.0, t_end=2000.0, factor=2.0)] if perturb else []
    return WorkloadSpec(duration_s=3600.0, seed=seed, perturbations=perts, classes=[
        ClassSpec(cls="background", rate_per_hour=60.0, turns_mean=8, isl0=2000, isl_growth=500, osl_mean=200, tools=tools),
        ClassSpec(cls="interactive", rate_per_hour=30.0, turns_mean=5, isl0=4000, isl_growth=800, osl_mean=150, tools=tools[:1],
                  think_log_mu=np.log(20.0), think_log_sigma=0.8)])

def test_generate_deterministic_and_valid():
    a = generate(_spec()); b = generate(_spec())
    assert a.df.equals(b.df) and len(a) > 100
    df = a.df
    tools = df[df["tool_name"].notna() & (df["tool_name"] != "__think__")]
    assert (tools["t_tool_end"] > tools["t_tool_start"]).all() and (tools["t_tool_start"] >= tools["t_last_token"] - 1e-9).all()
    assert set(df["class"]) == {"interactive", "background"} and df["source"].iloc[0] == "synthetic"
    strong = tools[tools["tool_name"] == "pytest"]
    assert all(len(ev) >= 1 for ev in strong["progress_events"] if True) and any(len(ev) > 3 for ev in strong["progress_events"])
    assert df["parent_session_id"].notna().any()
    assert (df[df["tool_name"] == "__think__"]["class"] == "interactive").all()

def test_tail_share_knob():
    lo = tail_share(generate(_spec(long_weight=0.02)))[1]
    hi = tail_share(generate(_spec(long_weight=0.3)))[1]
    assert hi > lo

def test_perturbation_slows_backend():
    df = generate(_spec(perturb=True)).df
    pert = df[df["perturbation_flag"]]
    assert len(pert) > 0 and (pert["backend_id"] == "ci").all()
    dur = lambda d: (d["t_tool_end"] - d["t_tool_start"])
    ci = df[(df["tool_name"] == "pytest")]
    assert dur(ci[ci["perturbation_flag"]]).median() > dur(ci[~ci["perturbation_flag"]]).median()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/traces/test_synthetic.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/traces/synthetic.py
from __future__ import annotations
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field
from atfm.schema.trace import TraceRow, TraceTable, ProgressEvent

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
        weights = np.array([t.weight for t in cs.tools], float); weights /= weights.sum()
        t = t0; prev_osl = 0
        for turn in range(n_turns):
            isl = cs.isl0 + cs.isl_growth * turn + prev_osl
            osl = 1 + rng.poisson(cs.osl_mean)
            t_first = t + isl / spec.llm_prefill_tps
            t_last = t_first + osl / spec.llm_decode_tps
            row = dict(session_id=sid, parent_session_id=parent, cls=cs.cls, tenant=tenant, turn_index=turn,
                       t_request=t, t_first_token=t_first, t_last_token=t_last, isl=int(isl), osl=int(osl), source="synthetic")
            last = turn == n_turns - 1
            if last:
                self.rows.append(TraceRow(**row)); break
            thinks = cs.cls == "interactive" and cs.think_log_mu is not None and rng.random() < 0.5
            if thinks:
                d = rng.lognormal(cs.think_log_mu, cs.think_log_sigma or 0.5)
                row.update(tool_name="__think__", backend_id="human", t_tool_start=t_last, t_tool_end=t_last + d, tool_exit_status=0)
                self.rows.append(TraceRow(**row)); t = t_last + d; prev_osl = osl; continue
            ts = cs.tools[rng.choice(len(cs.tools), p=weights)]
            base = rng.lognormal(ts.log_mu, ts.log_sigma)
            f = _factor(spec, ts.backend_id, t_last)
            d = base * f
            events: list[ProgressEvent] = []
            if ts.signal == "strong":
                k = 1
                while k * ts.progress_every_s < d:
                    events.append(ProgressEvent(t=t_last + k * ts.progress_every_s, completed=100.0 * k * ts.progress_every_s / d, total=100.0, phase="run"))
                    k += 1
                if not events:
                    events.append(ProgressEvent(t=t_last + d * 0.5, completed=50.0, total=100.0, phase="run"))
            elif ts.signal == "weak":
                events.append(ProgressEvent(t=t_last + 0.9 * d, completed=90.0, total=100.0, phase="end"))
            spawned = 1 if rng.random() < ts.spawn_prob else 0
            row.update(tool_name=ts.name, backend_id=ts.backend_id, t_tool_start=t_last, t_tool_end=t_last + d,
                       tool_exit_status=0, progress_events=events, perturbation_flag=f != 1.0, spawned_children=spawned)
            self.rows.append(TraceRow(**row))
            if spawned:
                self.session(cs, t_last + d, parent=sid, max_turns=int(rng.integers(1, 4)))
            t = t_last + d + 0.2; prev_osl = osl

    def run(self) -> TraceTable:
        for cs in self.spec.classes:
            n = self.rng.poisson(cs.rate_per_hour * self.spec.duration_s / 3600.0)
            for t0 in np.sort(self.rng.uniform(0.0, self.spec.duration_s, size=n)):
                self.session(cs, float(t0))
        return TraceTable.from_rows(self.rows)

def generate(spec: WorkloadSpec) -> TraceTable:
    return _Gen(spec).run()

def tail_share(table: TraceTable, threshold_s: float = 60.0) -> tuple[float, float]:
    df = table.df
    tools = df[df["tool_name"].notna() & (df["tool_name"] != "__think__")]
    d = (tools["t_tool_end"] - tools["t_tool_start"]).to_numpy(float)
    if len(d) == 0:
        return 0.0, 0.0
    long = d > threshold_s
    return float(long.mean()), float(d[long].sum() / d.sum())
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/traces/test_synthetic.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/traces/synthetic.py tests/traces/test_synthetic.py
git commit -m "feat: synthetic fleet generator with progress signals, fan-out and perturbations"
```

---

### Task 11: H1 experiment runner

**Files:**
- Create: `src/atfm/experiments/__init__.py`, `src/atfm/experiments/h1.py`, `scripts/run_h1.py`, `experiments/h1_tracelab.yaml`, `experiments/h1_synthetic.yaml`
- Test: `tests/experiments/test_h1.py`

**Interfaces:**
- Produces:
  - `H1Config` pydantic: `name, source: Literal["tracelab","synthetic"], tracelab_parquet: str | None, synthetic: WorkloadSpec | None, overlay_rate_per_hour: float | None, overlay_duration_s: float, tick_s: float = 30.0, horizons: list[float] = [10,30,120,300,900], n_samples: int = 512, models: list[str] = ["B0","B1","B2","M1","M2","M3"], block_seconds: float = 7*86400, test_fraction: float = 0.3, seed: int = 0, capacity_quantile: float = 0.95, out_dir: str = "runs"`.
  - `run_h1(cfg: H1Config) -> pd.DataFrame`: builds train/test tables (TraceLab: split by time blocks then overlay each with `overlay_rate_per_hour`; synthetic: generate two seeds, one train and one test), fits each session predictor on train, walks test ticks from `t_min` to `t_max - max(horizon)`, at each tick computes truth via `FleetReplayer.demand_truth`, states via `states_at`, feeds `SeriesForecaster.observe` for B0/B1, forecasts with every model, records per-(model, target, class, h) samples and truth plus `perturbed` (any perturbation active at t; TraceLab: False) and `endogenous_fraction`. M3 gets `observe_completion` calls for tools that ended in `(t - tick_s, t]`. Writes `out_dir/<name>/{metrics.csv, surge.json, config.json}` and returns the metrics DataFrame from `score_run`. Surge: capacity = `capacity_quantile` of the truth total kv_blocks series at h=300 s; computes `surge_events` on q90 for each model at h=300 s.
  - `scripts/run_h1.py <config.yaml>` loads the YAML into `H1Config` and calls `run_h1`, printing the pinball90 table pivoted by model and horizon.
- Model registry: `{"B0": SeriesForecaster(ConstantSeries()), "B1": SeriesForecaster(KalmanSeries()), "B2": SessionForecaster(HistoryPredictor()), "M1": SessionForecaster(SurvivalPredictor()), "M2": SessionForecaster(ProgressPredictor()), "M3": SessionForecaster(BackendPredictor())}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/experiments/test_h1.py
import numpy as np
from atfm.experiments.h1 import H1Config, run_h1
from atfm.traces.synthetic import WorkloadSpec, ClassSpec, ToolSpec

def test_run_h1_synthetic_smoke(tmp_path):
    tools = [ToolSpec(name="bash", weight=0.8, log_mu=np.log(3.0), log_sigma=0.5, signal="none"),
             ToolSpec(name="pytest", weight=0.2, log_mu=np.log(120.0), log_sigma=0.5, signal="strong", backend_id="ci")]
    spec = WorkloadSpec(duration_s=1200.0, seed=1, classes=[
        ClassSpec(cls="background", rate_per_hour=120.0, turns_mean=6, isl0=2000, isl_growth=400, osl_mean=100, tools=tools)])
    cfg = H1Config(name="smoke", source="synthetic", synthetic=spec, overlay_duration_s=1200.0, tick_s=60.0,
                   horizons=[30.0, 120.0], n_samples=32, models=["B0", "M1", "M2"], out_dir=str(tmp_path))
    df = run_h1(cfg)
    assert set(df["model"]) == {"B0_constant", "M1_survival", "M2_progress"}
    assert (tmp_path / "smoke" / "metrics.csv").exists() and (tmp_path / "smoke" / "surge.json").exists()
    assert df["pinball90"].notna().all() and (df["n"] > 5).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/experiments/__init__.py
```

```python
# src/atfm/experiments/h1.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from atfm.schema.trace import TraceTable
from atfm.traces.tracelab import load_tracelab
from atfm.traces.transform import split_by_time_blocks, overlay_sessions
from atfm.traces.synthetic import WorkloadSpec, generate
from atfm.board.replay import FleetReplayer
from atfm.board.predictors import (ConstantSeries, KalmanSeries, HistoryPredictor, SurvivalPredictor,
                                   ProgressPredictor, BackendPredictor)
from atfm.board.forecaster import SessionForecaster, SeriesForecaster, ExogenousModel, CLASSES, TARGETS
from atfm.eval.forecast import score_run, surge_events

class H1Config(BaseModel):
    name: str
    source: Literal["tracelab", "synthetic"]
    tracelab_parquet: str | None = None
    synthetic: WorkloadSpec | None = None
    overlay_rate_per_hour: float | None = None
    overlay_duration_s: float = 4 * 3600.0
    tick_s: float = 30.0
    horizons: list[float] = Field(default_factory=lambda: [10.0, 30.0, 120.0, 300.0, 900.0])
    n_samples: int = 512
    models: list[str] = Field(default_factory=lambda: ["B0", "B1", "B2", "M1", "M2", "M3"])
    block_seconds: float = 7 * 86400.0
    test_fraction: float = 0.3
    seed: int = 0
    capacity_quantile: float = 0.95
    out_dir: str = "runs"

def _tables(cfg: H1Config) -> tuple[TraceTable, TraceTable]:
    if cfg.source == "synthetic":
        assert cfg.synthetic is not None
        train = generate(cfg.synthetic.model_copy(update={"seed": cfg.synthetic.seed + 1000}))
        test = generate(cfg.synthetic)
        return train, test
    assert cfg.tracelab_parquet is not None
    table = TraceTable.from_parquet(cfg.tracelab_parquet)
    train, test = split_by_time_blocks(table, cfg.block_seconds, cfg.test_fraction, cfg.seed)
    if cfg.overlay_rate_per_hour:
        train = overlay_sessions(train, cfg.overlay_rate_per_hour, cfg.overlay_duration_s, cfg.seed + 1)
        test = overlay_sessions(test, cfg.overlay_rate_per_hour, cfg.overlay_duration_s, cfg.seed + 2)
    return train, test

def _build(cfg: H1Config, train: TraceTable) -> dict:
    exo = ExogenousModel().fit(train)
    reg = {
        "B0": lambda: SeriesForecaster(ConstantSeries(), cfg.horizons, cfg.n_samples),
        "B1": lambda: SeriesForecaster(KalmanSeries(), cfg.horizons, cfg.n_samples),
        "B2": lambda: SessionForecaster(HistoryPredictor().fit(train), exo, cfg.horizons, cfg.n_samples),
        "M1": lambda: SessionForecaster(SurvivalPredictor().fit(train), exo, cfg.horizons, cfg.n_samples),
        "M2": lambda: SessionForecaster(ProgressPredictor().fit(train), exo, cfg.horizons, cfg.n_samples),
        "M3": lambda: SessionForecaster(BackendPredictor().fit(train), exo, cfg.horizons, cfg.n_samples),
    }
    return {m: reg[m]() for m in cfg.models}

def _perturbed_at(cfg: H1Config, t: float) -> bool:
    if cfg.source != "synthetic" or cfg.synthetic is None:
        return False
    return any(p.t_start <= t < p.t_end for p in cfg.synthetic.perturbations)

def run_h1(cfg: H1Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    train, test = _tables(cfg)
    models = _build(cfg, train)
    rep = FleetReplayer(test)
    t_min, t_max = test.time_range()
    ticks = np.arange(t_min, t_max - max(cfg.horizons), cfg.tick_s)
    df = test.df
    ended = df[df["t_tool_end"].notna()][["t_tool_end", "backend_id", "tool_name", "t_tool_start"]].sort_values("t_tool_end")
    ended_t = ended["t_tool_end"].to_numpy(float)
    starts = df[df["turn_index"] == 0][["t_request", "class"]].sort_values("t_request")
    starts_t = starts["t_request"].to_numpy(float)
    records = []; series = {m: {"truth": [], "q90": []} for m in models}
    exo_ptr = 0; end_ptr = 0
    for t in ticks:
        t = float(t)
        truth = rep.demand_truth(t, cfg.horizons)
        states = rep.states_at(t)
        new_ptr = int(np.searchsorted(starts_t, t, side="right"))
        new_starts = [(float(starts_t[i]), starts["class"].iloc[i]) for i in range(exo_ptr, new_ptr)]
        exo_ptr = new_ptr
        e_ptr = int(np.searchsorted(ended_t, t, side="right"))
        completions = ended.iloc[end_ptr:e_ptr]; end_ptr = e_ptr
        pert = _perturbed_at(cfg, t)
        for name, fc in models.items():
            if isinstance(fc, SessionForecaster):
                fc.exo.update(t, new_starts)
                if isinstance(fc.predictor, BackendPredictor):
                    for r in completions.itertuples():
                        fc.predictor.observe_completion(r.backend_id, r.tool_name, float(r.t_tool_end - r.t_tool_start))
            snap = fc.forecast(t, states, rng)
            for tgt in TARGETS:
                for c in CLASSES:
                    for k, h in enumerate(cfg.horizons):
                        records.append({"t": t, "model": snap.model_id, "target": tgt, "class": c, "h_index": k, "h": h,
                                        "samples": snap.samples[tgt][c][k], "truth": float(truth[tgt][c][k]),
                                        "perturbed": pert, "endogenous_fraction": float(snap.endogenous_fraction[c][k])})
            k300 = cfg.horizons.index(300.0) if 300.0 in cfg.horizons else len(cfg.horizons) - 1
            series[name]["truth"].append(float(sum(truth["kv_blocks"][c][k300] for c in CLASSES)))
            series[name]["q90"].append(float(np.quantile(snap.total("kv_blocks")[k300], 0.9)))
            if isinstance(fc, SeriesForecaster):
                fc.observe(t, truth)
    metrics = score_run(records)
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    surge = {}
    for name, fc in models.items():
        truth_s = np.asarray(series[name]["truth"]); q90 = np.asarray(series[name]["q90"])
        cap = float(np.quantile(truth_s, cfg.capacity_quantile)) if len(truth_s) else 0.0
        ev = surge_events(q90, truth_s, cap, cfg.tick_s, horizon_s=300.0)
        surge[name] = {"capacity": cap, **ev}
    (out / "surge.json").write_text(json.dumps(surge, indent=2))
    (out / "config.json").write_text(cfg.model_dump_json(indent=2))
    return metrics
```

```python
# scripts/run_h1.py
import sys, yaml
import pandas as pd
from atfm.experiments.h1 import H1Config, run_h1

def main(path: str) -> None:
    cfg = H1Config(**yaml.safe_load(open(path)))
    df = run_h1(cfg)
    kv = df[(df["target"] == "kv_blocks")]
    print(kv.pivot_table(index=["class", "model"], columns="h", values="pinball90").round(2).to_string())

if __name__ == "__main__":
    main(sys.argv[1])
```

```yaml
# experiments/h1_tracelab.yaml
name: h1_tracelab_r200
source: tracelab
tracelab_parquet: data/tracelab/tracelab.parquet
overlay_rate_per_hour: 200
overlay_duration_s: 14400
tick_s: 30
n_samples: 256
models: [B0, B1, B2, M1, M2]
seed: 0
```

```yaml
# experiments/h1_synthetic.yaml
name: h1_synth_mixed
source: synthetic
overlay_duration_s: 7200
tick_s: 30
n_samples: 256
synthetic:
  duration_s: 7200
  seed: 0
  perturbations:
    - {backend_id: ci, t_start: 3000, t_end: 4200, factor: 2.0}
  classes:
    - cls: background
      rate_per_hour: 120
      turns_mean: 10
      isl0: 3000
      isl_growth: 600
      osl_mean: 200
      tools:
        - {name: bash, weight: 0.7, log_mu: 1.1, log_sigma: 0.6, signal: none, backend_id: local}
        - {name: pytest, weight: 0.2, log_mu: 5.3, log_sigma: 0.7, signal: strong, backend_id: ci, spawn_prob: 0.05}
        - {name: build, weight: 0.1, log_mu: 6.2, log_sigma: 0.5, signal: weak, backend_id: ci}
    - cls: interactive
      rate_per_hour: 60
      turns_mean: 6
      isl0: 4000
      isl_growth: 800
      osl_mean: 150
      think_log_mu: 3.0
      think_log_sigma: 0.8
      tools:
        - {name: bash, weight: 0.9, log_mu: 1.1, log_sigma: 0.6, signal: none, backend_id: local}
        - {name: pytest, weight: 0.1, log_mu: 4.6, log_sigma: 0.7, signal: strong, backend_id: ci}
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/experiments -v`
Expected: PASS

- [ ] **Step 5: Run the synthetic experiment end to end**

Run: `uv run python scripts/run_h1.py experiments/h1_synthetic.yaml`
Expected: a pinball90 table with rows per class and model, M1 below B2 and M2 below M1 for background at h=120 and 300.

- [ ] **Step 6: Run the TraceLab experiment**

Run: `uv run python scripts/run_h1.py experiments/h1_tracelab.yaml`
Expected: a table; record the result in `docs/research/2026-09-22-h1-first-results.md` with the numbers and the run directory.

- [ ] **Step 7: Commit**

```bash
git add src/atfm/experiments scripts experiments tests/experiments docs/research
git commit -m "feat: H1 forecastability experiment runner with TraceLab and synthetic configs"
```

---

## Self-review

- Spec coverage: schema 3.1 to 3.4 (Task 2), TraceLab adapter and synthetic generator from 8 and D4 (Tasks 3, 10), overlay and time-block splits from 9 and detail.md 8.4 (Task 4), session registry 5.1 (Task 5), predictor interface and ladder 5.2 (Tasks 6, 7), fleet forecaster 5.3 including children and exogenous (Task 8), forecast metrics 9 (Task 9), experiment definitions in YAML with run directories (Task 11). Directives 3.5, controllers, proxy, sidecar, simulator and serving metrics are out of scope for this plan by design (delivery order items 2 and 3).
- Type consistency: `SessionState` fields used in Tasks 6 to 8 match Task 5; `resumption(..., factors=)` keyword is defined on `ProgressPredictor` and used by `SessionForecaster` only for `BackendPredictor`; `ForecastSnapshot.total` used in Task 11 is defined in Task 2; `TARGETS`/`CLASSES` exported from `forecaster.py` and imported in Task 11.
- Review focus: item 1 tested in `test_duration_model_conditional` (tail at elapsed 500 s); item 2 in `test_parallel_tools_critical_path`; item 3 in `test_three_round_session` and `test_one_shot_session_ends`; item 4 in `test_session_forecaster_shapes_and_empty`; item 5 in `test_crps_point_forecast_is_abs_error` and `test_pinball_asymmetry`.
