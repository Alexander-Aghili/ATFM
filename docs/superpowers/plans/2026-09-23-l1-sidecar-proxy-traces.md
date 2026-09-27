# L1: Sidecar, Bus, Proxy, Live Board, Long-Tool Trace Collection and H1b. Implementation Plan

> **Historical implementation plan.** Embedded code and task checklists are a design record,
> not the current source of setup instructions. See [implementation status](../../status.md),
> [operations](../../operations.md), and [core development](../../development/core.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the laptop-runnable plumbing of ATFM (event schema and bus, tool-runtime sidecar with progress parsers and launch gate, mini-SWE-agent adapters, harness proxy with a global window and priority tiers in front of Dynamo Mocker, live demand board), use it to collect real long-tool traces with progress events, and report signal coverage and an honest M2-versus-M1 forecast result (hypothesis H1b).

**Architecture:** The sidecar wraps every tool subprocess: bytes go back to the harness unchanged while a parser chain turns the same stream into progress and data events on the bus. The proxy is an OpenAI-compatible ASGI app that classifies each call, computes the index and tier, holds it in a global window or per controller directive, injects `nvext.agent_hints`, forwards it, and emits call events. The live board rebuilds session states from bus events and reuses the L0 predictors and forecaster. A scripted driver runs real tools (pytest, builds, pipelines) inside Docker through the sidecar adapters while making its LLM calls through the proxy to Mocker, producing a canonical trace table for the H1 runner.

**Tech Stack:** Python 3.12, uv, pydantic v2, FastAPI + uvicorn, httpx, numpy, pandas, pytest, pytest-asyncio, optional extras: `mini-swe-agent` (harness), `ai-dynamo` (Mocker + frontend), Docker CLI for tool containers.

**Spec:** `docs/superpowers/specs/2026-09-22-atfm-architecture-design.md` v1.1, sections 3.2, 4, 5.1, 5.4, 6.2 (hold semantics only), 7, 9 (H1b), 10, 13 item 2. Platform facts: `docs/research/2026-09-22-platform-acquisition.md`.

## Global Constraints

- Python 3.12 exactly; `uv`; package `atfm`, src layout; every stochastic function takes `rng`.
- Timestamps are float seconds since the Unix epoch, UTC; durations float seconds.
- Sidecar contract (D9): the tool's bytes reach the harness unchanged; the sidecar never blocks on the bus; every parser exception is caught per line; the sidecar and gate are fail-open.
- Proxy contract (D10): if the board or gate is unreachable within 50 ms, forward with default hints; holds have a hard cap (`max_hold_s`, default 600); the request body is forwarded unchanged apart from `nvext` and headers.
- Priority encoding (spec 4.3): `strict_priority` 2 = interactive under slack threshold, 1 = interactive, 0 = background; `priority` = coarse bucket 0..3 of the index within the tier.
- Global admission window `W` (spec 4.2); no per-worker windows.
- Class values exactly `"interactive"` or `"background"`; header names `x-atfm-session`, `x-atfm-class`, `x-atfm-tenant`, `x-atfm-deadline` (epoch seconds), `x-atfm-parent`.
- Optional dependencies live in extras: `harness` (mini-swe-agent), `dynamo` (ai-dynamo), `serve` (fastapi, uvicorn, httpx). Core tests never require the `harness` or `dynamo` extras; tests that need Docker or Dynamo are skipped unless `ATFM_DOCKER=1` / `ATFM_DYNAMO=1`.
- `data/` and `runs/` are git-ignored; tests build fixtures inline.

## Review Focus

1. A tool that writes binary or invalid UTF-8 to stdout: the harness must receive exactly the bytes the tool produced, and the parser chain must not raise. Test in Task 2.
2. A tool that exceeds its timeout: the process must be killed (no orphan), the result must carry `returncode == -1` and the partial output, and a `tool.end` event must still be emitted. Test in Task 2.
3. The proxy with the window full when an interactive call with a near deadline arrives behind three background calls: the interactive call must be released next, before any background call, and its hint must carry `strict_priority == 2`. Test in Task 4.
4. A hold directive with a release time beyond `max_hold_s`: the proxy must cap the hold at `max_hold_s` and log the cap, never hold longer. Test in Task 4.
5. Bus events arriving out of order or for unknown sessions at the live board: a `tool.progress` before its `tool.start`, or events for a session never announced, must create the session and not raise. Test in Task 5.

---

### Task 1: Event schema, bus, and serving extras

**Files:**
- Create: `src/atfm/schema/events.py`, `src/atfm/bus/__init__.py`, `src/atfm/bus/memory.py`, `src/atfm/bus/jsonl.py`
- Modify: `pyproject.toml` (extras), `src/atfm/schema/__init__.py` (exports)
- Test: `tests/schema/test_events.py`, `tests/bus/test_bus.py`

**Interfaces:**
- Produces: pydantic models `SessionStart, LlmRequest, LlmFirstToken, LlmDone, ToolStart, ToolProgress, ToolData, ToolEnd, WorkerMetrics, SpawnRequest`; union `Event`; `parse_event(d: dict) -> Event`; `event_to_dict(e: Event) -> dict` (by alias); `Bus` protocol with `publish(e: Event) -> None` and `drain() -> list[Event]`; `InMemoryBus()`; `JsonlBus(path)` with `publish` appending one JSON line and `read_events(path) -> list[Event]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/schema/test_events.py
from atfm.schema.events import parse_event, event_to_dict, ToolProgress, LlmRequest

def test_roundtrip_by_kind():
    d = {"kind": "tool.progress", "t": 1.5, "session_id": "s", "call_id": "c", "completed": 3, "total": 10, "phase": "run"}
    e = parse_event(d)
    assert isinstance(e, ToolProgress) and e.completed == 3
    assert event_to_dict(e) == d

def test_class_alias_and_defaults():
    e = parse_event({"kind": "session.start", "t": 0.0, "session_id": "s", "tenant": "t", "class": "background"})
    assert e.cls == "background" and e.parent_session_id is None and e.deadline is None
    r = parse_event({"kind": "llm.request", "t": 2.0, "session_id": "s", "turn_index": 0, "request_id": "r", "isl": 100})
    assert isinstance(r, LlmRequest) and r.hints == {} and r.held_s == 0.0

def test_unknown_kind_rejected():
    import pytest
    with pytest.raises(ValueError):
        parse_event({"kind": "nope", "t": 0.0})
```

```python
# tests/bus/test_bus.py
from atfm.bus import InMemoryBus, JsonlBus, read_events
from atfm.schema.events import ToolStart, ToolEnd

def _ev():
    return [ToolStart(t=1.0, session_id="s", turn_index=0, call_id="c", tool_name="pytest", backend_id="local"),
            ToolEnd(t=5.0, session_id="s", call_id="c", exit_status=0, output_chars=12)]

def test_memory_bus_drain_clears():
    b = InMemoryBus()
    for e in _ev():
        b.publish(e)
    assert [e.kind for e in b.drain()] == ["tool.start", "tool.end"] and b.drain() == []

def test_jsonl_bus_roundtrip(tmp_path):
    p = tmp_path / "ev.jsonl"
    b = JsonlBus(p)
    for e in _ev():
        b.publish(e)
    back = read_events(p)
    assert [e.kind for e in back] == ["tool.start", "tool.end"] and back[1].output_chars == 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/schema/test_events.py tests/bus -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Add extras and implement**

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
serve = ["fastapi>=0.115", "uvicorn>=0.30", "httpx>=0.27"]
harness = ["mini-swe-agent>=2.4"]
dynamo = ["ai-dynamo>=1.5"]
dev = ["pytest>=8", "pytest-cov>=5", "pytest-asyncio>=0.24", "httpx>=0.27", "fastapi>=0.115"]
```

```python
# src/atfm/schema/events.py
from __future__ import annotations
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

class _E(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    t: float

class SessionStart(_E):
    kind: Literal["session.start"] = "session.start"
    session_id: str
    tenant: str
    cls: Literal["interactive", "background"] = Field(alias="class")
    parent_session_id: str | None = None
    deadline: float | None = None

class LlmRequest(_E):
    kind: Literal["llm.request"] = "llm.request"
    session_id: str
    turn_index: int
    request_id: str
    isl: int
    predicted_osl: int | None = None
    hints: dict = Field(default_factory=dict)
    held_s: float = 0.0

class LlmFirstToken(_E):
    kind: Literal["llm.first_token"] = "llm.first_token"
    session_id: str
    request_id: str

class LlmDone(_E):
    kind: Literal["llm.done"] = "llm.done"
    session_id: str
    request_id: str
    osl: int = 0
    worker_id: str | None = None
    prefix_hit_tokens: int = 0
    status: int = 200

class ToolStart(_E):
    kind: Literal["tool.start"] = "tool.start"
    session_id: str
    turn_index: int
    call_id: str
    tool_name: str
    backend_id: str = "local"
    args_hash: str | None = None

class ToolProgress(_E):
    kind: Literal["tool.progress"] = "tool.progress"
    session_id: str
    call_id: str
    completed: float
    total: float | None = None
    phase: str | None = None

class ToolData(_E):
    kind: Literal["tool.data"] = "tool.data"
    session_id: str
    call_id: str
    metric: str
    value: float

class ToolEnd(_E):
    kind: Literal["tool.end"] = "tool.end"
    session_id: str
    call_id: str
    exit_status: int
    output_chars: int = 0

class SpawnRequest(_E):
    kind: Literal["spawn.request"] = "spawn.request"
    parent_session_id: str
    child_session_id: str

class WorkerMetrics(_E):
    kind: Literal["worker.metrics"] = "worker.metrics"
    worker_id: str
    kv_blocks_used: int
    kv_blocks_total: int
    queue_depth: int = 0
    tier_blocks: dict[str, int] = Field(default_factory=dict)

Event = Annotated[Union[SessionStart, LlmRequest, LlmFirstToken, LlmDone, ToolStart, ToolProgress, ToolData,
                        ToolEnd, SpawnRequest, WorkerMetrics], Field(discriminator="kind")]
_adapter = TypeAdapter(Event)

def parse_event(d: dict) -> Event:
    return _adapter.validate_python(d)

def event_to_dict(e: Event) -> dict:
    return e.model_dump(by_alias=True, exclude_none=False, exclude_defaults=False) | {}
```

Note: `event_to_dict` must produce exactly the input dict in the round-trip test; `exclude_none=False` keeps `None` fields that were present. The test dict has no optional-absent fields, so full dump equals input.

```python
# src/atfm/bus/__init__.py
from .memory import InMemoryBus
from .jsonl import JsonlBus, read_events
from typing import Protocol
from atfm.schema.events import Event

class Bus(Protocol):
    def publish(self, e: Event) -> None: ...
    def drain(self) -> list[Event]: ...

__all__ = ["Bus", "InMemoryBus", "JsonlBus", "read_events"]
```

```python
# src/atfm/bus/memory.py
from __future__ import annotations
from collections import deque
from threading import Lock
from atfm.schema.events import Event

class InMemoryBus:
    def __init__(self):
        self._q: deque = deque()
        self._lock = Lock()

    def publish(self, e: Event) -> None:
        with self._lock:
            self._q.append(e)

    def drain(self) -> list[Event]:
        with self._lock:
            out = list(self._q)
            self._q.clear()
        return out
```

```python
# src/atfm/bus/jsonl.py
from __future__ import annotations
import json
from pathlib import Path
from threading import Lock
from atfm.schema.events import Event, event_to_dict, parse_event

class JsonlBus:
    """Append-only event log; publish never raises (write failures are counted)."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self.errors = 0

    def publish(self, e: Event) -> None:
        try:
            line = json.dumps(event_to_dict(e), separators=(",", ":"))
            with self._lock, open(self.path, "a") as f:
                f.write(line + "\n")
        except Exception:
            self.errors += 1

    def drain(self) -> list[Event]:
        return read_events(self.path)

def read_events(path: str | Path) -> list[Event]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(parse_event(json.loads(line)))
    return out
```

Add to `src/atfm/schema/__init__.py`: `from .events import Event, parse_event, event_to_dict` and extend `__all__`.

- [ ] **Step 4: Sync and run tests**

Run: `uv sync --extra dev --extra serve && uv run pytest tests/schema tests/bus -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/atfm/schema src/atfm/bus tests/schema tests/bus
git commit -m "feat: event schema, in-memory and JSONL buses"
```

---

### Task 2: Sidecar core with streaming and progress parsers

**Files:**
- Create: `src/atfm/sidecar/__init__.py`, `src/atfm/sidecar/parsers.py`, `src/atfm/sidecar/core.py`
- Test: `tests/sidecar/test_parsers.py`, `tests/sidecar/test_core.py`

**Interfaces:**
- Produces:
  - `Parser` protocol: `feed(line: str, t: float) -> dict | None` returning `{"completed": float, "total": float | None, "phase": str | None}` or `None`; `PytestParser`, `PercentParser`, `CounterParser`, `RateParser(window_s=5.0)` (emits `{"metric": "lines_per_s", "value": v}` dicts through `feed_data(line, t) -> dict | None`).
  - `default_parsers() -> list[Parser]`.
  - `ToolContext(session_id, turn_index, tool_name, backend_id="local", args_hash=None)`.
  - `ToolResult(call_id, output: bytes, returncode: int, duration_s: float, timed_out: bool)`.
  - `run_tool(cmd: str | list[str], ctx: ToolContext, bus, *, cwd=None, env=None, timeout=None, shell=None, parsers=None, clock=time.time) -> ToolResult`. `shell` defaults to `True` when `cmd` is a string. Emits `tool.start`, deduplicated `tool.progress` (only when `completed` changes), `tool.data`, `tool.end` (exit_status = returncode, or -1 on timeout). Output bytes are exactly the subprocess's combined stdout+stderr in arrival order (stderr merged into stdout).
  - `classify_tool(command: str) -> str`: `pytest` if "pytest" in command; `build` if any of ("make", "cmake", "ninja", "cargo build", "npm run build", "gradle") in it; `install` for "pip install"/"npm install"/"apt-get"; `clone` for "git clone"; else the first word of the last `&&`-separated segment.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sidecar/test_parsers.py
from atfm.sidecar.parsers import PytestParser, PercentParser, CounterParser, RateParser
from atfm.sidecar.core import classify_tool

def test_pytest_parser_counts_results():
    p = PytestParser()
    assert p.feed("collected 3 items", 0.0) == {"completed": 0, "total": 3, "phase": "collect"}
    assert p.feed("tests/test_a.py::test_x PASSED                    [ 33%]", 1.0) == {"completed": 1, "total": 3, "phase": "run"}
    assert p.feed("tests/test_a.py::test_y FAILED                    [ 66%]", 2.0) == {"completed": 2, "total": 3, "phase": "run"}
    assert p.feed("some unrelated line", 2.5) is None
    assert p.feed("=========== 1 failed, 1 passed in 2.10s ===========", 3.0) == {"completed": 3, "total": 3, "phase": "done"}

def test_percent_and_counter_parsers():
    assert PercentParser().feed("[ 45%] Building C object x.c.o", 0.0) == {"completed": 45.0, "total": 100.0, "phase": "run"}
    assert CounterParser().feed("[12/345] compiling foo", 0.0) == {"completed": 12.0, "total": 345.0, "phase": "run"}
    assert CounterParser().feed("processed 10/100 rows", 0.0) == {"completed": 10.0, "total": 100.0, "phase": "run"}
    assert CounterParser().feed("version 1/2", 0.0) is None          # total must be at least 3
    assert CounterParser().feed("k 500/100", 0.0) is None            # completed may not exceed total

def test_rate_parser_emits_every_window():
    r = RateParser(window_s=5.0)
    assert r.feed_data("a", 0.0) is None
    for i in range(9):
        assert r.feed_data("x", 1.0 + i * 0.5) is None
    d = r.feed_data("y", 5.5)
    assert d["metric"] == "lines_per_s" and abs(d["value"] - 11 / 5.5) < 1e-9

def test_classify_tool():
    assert classify_tool("cd /w && pytest -q tests") == "pytest"
    assert classify_tool("cmake --build build -j4") == "build"
    assert classify_tool("pip install -e .") == "install"
    assert classify_tool("git clone https://x/y") == "clone"
    assert classify_tool("cd /w && ls -la") == "ls"
```

```python
# tests/sidecar/test_core.py
import os, sys, time
from atfm.bus import InMemoryBus
from atfm.sidecar.core import run_tool, ToolContext

CTX = ToolContext(session_id="s1", turn_index=0, tool_name="pytest", backend_id="local")

def test_output_is_byte_exact_and_events_emitted():
    bus = InMemoryBus()
    script = "import sys; sys.stdout.buffer.write(b'collected 2 items\\n'); sys.stdout.buffer.write(b'\\xff\\xfe raw bytes\\n'); sys.stdout.buffer.write(b'a::t PASSED [ 50%]\\n'); sys.stderr.write('warn\\n'); sys.stdout.buffer.write(b'b::t PASSED [100%]\\n')"
    res = run_tool([sys.executable, "-u", "-c", script], CTX, bus, shell=False)
    assert res.returncode == 0 and res.timed_out is False
    assert b"\xff\xfe raw bytes\n" in res.output and b"warn\n" in res.output and res.output.startswith(b"collected 2 items\n")
    kinds = [e.kind for e in bus.drain()]
    assert kinds[0] == "tool.start" and kinds[-1] == "tool.end" and kinds.count("tool.progress") == 3  # 0/2, 1/2, 2/2

def test_progress_is_deduplicated():
    bus = InMemoryBus()
    res = run_tool("printf '[ 10%%] a\\n[ 10%%] b\\n[ 20%%] c\\n'", CTX, bus)
    assert res.returncode == 0
    prog = [e for e in bus.drain() if e.kind == "tool.progress"]
    assert [e.completed for e in prog] == [10.0, 20.0]

def test_timeout_kills_and_reports():
    bus = InMemoryBus()
    t0 = time.time()
    res = run_tool([sys.executable, "-u", "-c", "import time,sys; print('start', flush=True); time.sleep(30)"], CTX, bus, shell=False, timeout=1.0)
    assert res.timed_out and res.returncode == -1 and b"start" in res.output and time.time() - t0 < 5
    end = [e for e in bus.drain() if e.kind == "tool.end"][0]
    assert end.exit_status == -1

def test_parser_exception_does_not_break_tool():
    class Bad:
        def feed(self, line, t):
            raise RuntimeError("boom")
    bus = InMemoryBus()
    res = run_tool("echo hello", CTX, bus, parsers=[Bad()])
    assert res.returncode == 0 and res.output == b"hello\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sidecar -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/sidecar/__init__.py
from .core import run_tool, ToolContext, ToolResult, classify_tool
from .parsers import default_parsers
__all__ = ["run_tool", "ToolContext", "ToolResult", "classify_tool", "default_parsers"]
```

```python
# src/atfm/sidecar/parsers.py
from __future__ import annotations
import re
from typing import Protocol

class Parser(Protocol):
    def feed(self, line: str, t: float) -> dict | None: ...

_COLLECTED = re.compile(r"collected (\d+) items?")
_RESULT = re.compile(r"::\S.*\b(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")
_SUMMARY = re.compile(r"^=+ .*(passed|failed|error|skipped|no tests ran).* in [\d.]+s")
_PERCENT = re.compile(r"\[\s*(\d{1,3})%\]")
_COUNTER = re.compile(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?![\d.])")

class PytestParser:
    def __init__(self):
        self.total: int | None = None
        self.done = 0

    def feed(self, line: str, t: float) -> dict | None:
        m = _COLLECTED.search(line)
        if m:
            self.total = int(m.group(1))
            self.done = 0
            return {"completed": 0, "total": self.total, "phase": "collect"}
        if _RESULT.search(line):
            self.done += 1
            return {"completed": self.done, "total": self.total, "phase": "run"}
        if _SUMMARY.search(line):
            total = self.total if self.total is not None else self.done
            return {"completed": total, "total": total, "phase": "done"}
        return None

class PercentParser:
    def feed(self, line: str, t: float) -> dict | None:
        m = _PERCENT.search(line)
        if not m:
            return None
        return {"completed": float(m.group(1)), "total": 100.0, "phase": "run"}

class CounterParser:
    def feed(self, line: str, t: float) -> dict | None:
        m = _COUNTER.search(line)
        if not m:
            return None
        k, n = float(m.group(1)), float(m.group(2))
        if n < 3 or k > n:
            return None
        return {"completed": k, "total": n, "phase": "run"}

class RateParser:
    """Weak signal: output line rate over a window; emitted as a data event."""
    def __init__(self, window_s: float = 5.0):
        self.window_s = window_s
        self.t0: float | None = None
        self.n = 0

    def feed(self, line: str, t: float) -> dict | None:
        return None

    def feed_data(self, line: str, t: float) -> dict | None:
        if self.t0 is None:
            self.t0 = t
        self.n += 1
        if t - self.t0 >= self.window_s:
            v = self.n / (t - self.t0)
            self.t0, self.n = t, 0
            return {"metric": "lines_per_s", "value": v}
        return None

def default_parsers() -> list:
    return [PytestParser(), PercentParser(), CounterParser(), RateParser()]
```

```python
# src/atfm/sidecar/core.py
from __future__ import annotations
import os, signal, subprocess, time, uuid
from dataclasses import dataclass
from atfm.schema.events import ToolStart, ToolProgress, ToolData, ToolEnd
from .parsers import default_parsers

@dataclass
class ToolContext:
    session_id: str
    turn_index: int
    tool_name: str
    backend_id: str = "local"
    args_hash: str | None = None

@dataclass
class ToolResult:
    call_id: str
    output: bytes
    returncode: int
    duration_s: float
    timed_out: bool = False

_BUILD = ("make", "cmake", "ninja", "cargo build", "npm run build", "gradle")
_INSTALL = ("pip install", "npm install", "apt-get", "uv sync", "uv pip")

def classify_tool(command: str) -> str:
    c = command.strip()
    if "pytest" in c:
        return "pytest"
    if any(b in c for b in _BUILD):
        return "build"
    if any(i in c for i in _INSTALL):
        return "install"
    if "git clone" in c:
        return "clone"
    last = c.split("&&")[-1].strip()
    return last.split()[0] if last else "sh"

def _safe_publish(bus, e) -> None:
    try:
        bus.publish(e)
    except Exception:
        pass

def run_tool(cmd, ctx: ToolContext, bus, *, cwd=None, env=None, timeout: float | None = None, shell=None,
             parsers=None, clock=time.time) -> ToolResult:
    """Run a tool subprocess; return its bytes unchanged; emit progress events from the same stream."""
    if shell is None:
        shell = isinstance(cmd, str)
    parsers = default_parsers() if parsers is None else parsers
    call_id = uuid.uuid4().hex[:16]
    t_start = clock()
    _safe_publish(bus, ToolStart(t=t_start, session_id=ctx.session_id, turn_index=ctx.turn_index, call_id=call_id,
                                 tool_name=ctx.tool_name, backend_id=ctx.backend_id, args_hash=ctx.args_hash))
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    chunks: list[bytes] = []
    last_completed: float | None = None
    timed_out = False
    deadline = None if timeout is None else t_start + timeout
    assert proc.stdout is not None
    try:
        while True:
            if deadline is not None and clock() > deadline:
                timed_out = True
                break
            line = proc.stdout.readline()
            if not line:
                break
            chunks.append(line)
            now = clock()
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            for p in parsers:
                try:
                    prog = p.feed(text, now)
                    if prog is not None and prog.get("completed") != last_completed:
                        last_completed = prog["completed"]
                        _safe_publish(bus, ToolProgress(t=now, session_id=ctx.session_id, call_id=call_id,
                                                        completed=float(prog["completed"]),
                                                        total=None if prog.get("total") is None else float(prog["total"]),
                                                        phase=prog.get("phase")))
                        break
                    fd = getattr(p, "feed_data", None)
                    if fd is not None:
                        d = fd(text, now)
                        if d is not None:
                            _safe_publish(bus, ToolData(t=now, session_id=ctx.session_id, call_id=call_id,
                                                        metric=d["metric"], value=float(d["value"])))
                except Exception:
                    continue
    finally:
        if timed_out:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            try:
                rest = proc.stdout.read()
                if rest:
                    chunks.append(rest)
            except Exception:
                pass
        proc.wait()
    output = b"".join(chunks)
    rc = -1 if timed_out else int(proc.returncode)
    t_end = clock()
    _safe_publish(bus, ToolEnd(t=t_end, session_id=ctx.session_id, call_id=call_id, exit_status=rc, output_chars=len(output)))
    return ToolResult(call_id=call_id, output=output, returncode=rc, duration_s=t_end - t_start, timed_out=timed_out)
```

Timeout detail: `readline()` blocks while the child sleeps silently, so the deadline check above never fires for a silent process. Use a reader thread instead:

```python
# replace the read loop in run_tool with this thread-based version
import threading, queue as _queue

    q: "_queue.Queue[bytes | None]" = _queue.Queue()
    def _reader():
        try:
            for line in iter(proc.stdout.readline, b""):
                q.put(line)
        finally:
            q.put(None)
    threading.Thread(target=_reader, daemon=True).start()
    try:
        while True:
            remaining = None if deadline is None else max(0.0, deadline - clock())
            try:
                line = q.get(timeout=remaining if remaining is not None else 0.5)
            except _queue.Empty:
                if deadline is not None and clock() > deadline:
                    timed_out = True
                    break
                continue
            if line is None:
                break
            chunks.append(line)
            # ... parser handling exactly as above ...
```

Write the final function with the thread-based loop (the first listing shows the parser handling; the second shows the loop; combine them, keeping the `finally` block).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/sidecar -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/sidecar tests/sidecar
git commit -m "feat: sidecar core with byte-exact streaming and progress parsers"
```

---

### Task 3: Launch gate client and mini-SWE-agent adapters

**Files:**
- Create: `src/atfm/sidecar/gate.py`, `src/atfm/sidecar/minisweagent.py`
- Test: `tests/sidecar/test_adapters.py`

**Interfaces:**
- Consumes: `run_tool`, `ToolContext`, `classify_tool`, buses.
- Produces:
  - `gate_allowed_at(url: str | None, session_id: str, kind: str, *, timeout_s: float = 0.2) -> float | None`: POSTs `{"session_id", "kind"}` to `f"{url}/gate"`, returns `allowed_at` epoch seconds or `None` (now) on any failure or when `url` is None. Uses `urllib.request` (no httpx dependency in the harness process).
  - `SidecarConfig(session_id, tenant="t0", cls="background", bus=None, gate_url=None, deferrable=None, backend_map: dict[str, str] | None = None, max_gate_wait_s=600.0, clock=time.time)`; `deferrable` defaults to `cls == "background"`; `backend_for(tool_name) -> str` uses `backend_map` (default `{"pytest": "ci", "build": "ci", "install": "pkg", "clone": "git"}`, else `"local"`).
  - `SidecarMixin.sidecar_execute(command: str, cwd: str, timeout: float | None, argv_builder) -> dict`: increments `turn_index`, classifies the tool, consults the gate when deferrable and sleeps until `allowed_at` (capped by `max_gate_wait_s`), runs `run_tool(argv_builder(command), ...)`, returns mini-SWE-agent's result dict `{"output": str, "returncode": int, "exception_info": str}` with output decoded as UTF-8 with replacement (mini-SWE-agent itself decodes with `text=True`, so replacement is the closest match to its behaviour).
  - `SidecarLocalEnvironment(LocalEnvironment)` and `SidecarDockerEnvironment(DockerEnvironment)`: construct with `sidecar=SidecarConfig(...)` plus the base class kwargs; override `execute`. Local runs `["bash", "-lc", command]` in `cwd`; Docker runs `[executable, "exec", "-w", cwd, *("-e", f"{k}={v}") for forwarded env..., container_id, "bash", "-lc", command]`. Both call `self._check_finished(output)` after building the dict, exactly as the base classes do. Importing this module without `mini-swe-agent` installed raises `ImportError` with a message naming the `harness` extra.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sidecar/test_adapters.py
import json, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from atfm.bus import InMemoryBus
from atfm.sidecar.gate import gate_allowed_at
from atfm.sidecar.minisweagent import SidecarConfig, SidecarMixin

class _Gate(BaseHTTPRequestHandler):
    allowed_at = None
    def do_POST(self):
        n = int(self.headers.get("content-length", 0)); body = json.loads(self.rfile.read(n))
        self.send_response(200); self.send_header("content-type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({"allowed_at": _Gate.allowed_at, "echo": body}).encode())
    def log_message(self, *a): pass

@pytest.fixture
def gate_server():
    srv = HTTPServer(("127.0.0.1", 0), _Gate)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()

def test_gate_fail_open_and_value(gate_server):
    assert gate_allowed_at(None, "s", "tool") is None
    assert gate_allowed_at("http://127.0.0.1:1", "s", "tool", timeout_s=0.2) is None   # nothing listening
    _Gate.allowed_at = 123.0
    assert gate_allowed_at(gate_server, "s", "tool") == 123.0

class _Env(SidecarMixin):
    def __init__(self, cfg):
        self.sidecar = cfg
    def _check_finished(self, output):
        pass

def test_mixin_runs_tool_and_publishes(gate_server):
    bus = InMemoryBus()
    _Gate.allowed_at = time.time() + 0.3
    cfg = SidecarConfig(session_id="s9", cls="background", bus=bus, gate_url=gate_server)
    env = _Env(cfg)
    t0 = time.time()
    out = env.sidecar_execute("printf 'collected 1 items\\nx::t PASSED [100%%]\\n'", cwd="/tmp", timeout=10,
                              argv_builder=lambda c: ["bash", "-lc", c])
    assert out["returncode"] == 0 and out["output"].startswith("collected 1 items") and out["exception_info"] == ""
    assert time.time() - t0 >= 0.25                       # waited for the gate
    ev = bus.drain()
    assert ev[0].kind == "tool.start" and ev[0].tool_name == "sh" or ev[0].tool_name == "printf"
    assert any(e.kind == "tool.progress" for e in ev) and ev[-1].kind == "tool.end"
    assert env.sidecar.turn_index == 1

def test_interactive_skips_gate():
    bus = InMemoryBus()
    cfg = SidecarConfig(session_id="i1", cls="interactive", bus=bus, gate_url="http://127.0.0.1:1")
    t0 = time.time()
    out = _Env(cfg).sidecar_execute("echo hi", cwd="/tmp", timeout=5, argv_builder=lambda c: ["bash", "-lc", c])
    assert out["output"] == "hi\n" and time.time() - t0 < 1.0

def test_minisweagent_classes_importable_or_skipped():
    pytest.importorskip("minisweagent")
    from atfm.sidecar.minisweagent import SidecarLocalEnvironment
    env = SidecarLocalEnvironment(sidecar=SidecarConfig(session_id="l1", bus=InMemoryBus()))
    out = env.execute({"command": "echo local"}, cwd="/tmp")
    assert out["output"] == "local\n" and out["returncode"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sidecar/test_adapters.py -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/sidecar/gate.py
from __future__ import annotations
import json, urllib.request

def gate_allowed_at(url: str | None, session_id: str, kind: str, *, timeout_s: float = 0.2) -> float | None:
    """Ask the proxy's launch gate. Fail-open: any error means 'allowed now' (None)."""
    if not url:
        return None
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/gate", data=json.dumps({"session_id": session_id, "kind": kind}).encode(),
                                     headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = json.loads(r.read().decode())
        v = body.get("allowed_at")
        return None if v is None else float(v)
    except Exception:
        return None
```

```python
# src/atfm/sidecar/minisweagent.py
from __future__ import annotations
import os, time
from dataclasses import dataclass, field
from typing import Callable
from .core import ToolContext, classify_tool, run_tool
from .gate import gate_allowed_at

DEFAULT_BACKENDS = {"pytest": "ci", "build": "ci", "install": "pkg", "clone": "git"}

@dataclass
class SidecarConfig:
    session_id: str
    tenant: str = "t0"
    cls: str = "background"
    bus: object = None
    gate_url: str | None = None
    deferrable: bool | None = None
    backend_map: dict[str, str] | None = None
    max_gate_wait_s: float = 600.0
    clock: Callable[[], float] = time.time
    turn_index: int = field(default=0)

    def __post_init__(self):
        if self.deferrable is None:
            self.deferrable = self.cls == "background"
        if self.backend_map is None:
            self.backend_map = dict(DEFAULT_BACKENDS)
        if self.bus is None:
            from atfm.bus import InMemoryBus
            self.bus = InMemoryBus()

    def backend_for(self, tool_name: str) -> str:
        return self.backend_map.get(tool_name, "local")

class SidecarMixin:
    """Provides sidecar_execute for mini-SWE-agent environment subclasses (attribute `sidecar`)."""
    sidecar: SidecarConfig

    def sidecar_execute(self, command: str, cwd: str, timeout: float | None, argv_builder) -> dict:
        cfg = self.sidecar
        tool = classify_tool(command)
        if cfg.deferrable:
            allowed = gate_allowed_at(cfg.gate_url, cfg.session_id, "tool")
            if allowed is not None:
                wait = min(max(0.0, allowed - cfg.clock()), cfg.max_gate_wait_s)
                if wait > 0:
                    time.sleep(wait)
        ctx = ToolContext(session_id=cfg.session_id, turn_index=cfg.turn_index, tool_name=tool,
                          backend_id=cfg.backend_for(tool))
        cfg.turn_index += 1
        res = run_tool(argv_builder(command), ctx, cfg.bus, cwd=cwd or None, timeout=timeout, shell=False,
                       clock=cfg.clock)
        output = {"output": res.output.decode("utf-8", errors="replace"), "returncode": res.returncode,
                  "exception_info": "" if not res.timed_out else f"timeout after {timeout}s"}
        self._check_finished(output)
        return output

try:
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.environments.docker import DockerEnvironment
except ImportError as _e:  # pragma: no cover
    LocalEnvironment = DockerEnvironment = None
    _import_error = _e
else:
    _import_error = None

if LocalEnvironment is not None:
    class SidecarLocalEnvironment(SidecarMixin, LocalEnvironment):
        def __init__(self, *, sidecar: SidecarConfig, **kwargs):
            super().__init__(**kwargs)
            self.sidecar = sidecar

        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
            command = action.get("command", "")
            cwd = cwd or self.config.cwd or os.getcwd()
            return self.sidecar_execute(command, cwd, timeout or self.config.timeout, lambda c: ["bash", "-lc", c])

    class SidecarDockerEnvironment(SidecarMixin, DockerEnvironment):
        def __init__(self, *, sidecar: SidecarConfig, **kwargs):
            super().__init__(**kwargs)
            self.sidecar = sidecar

        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
            command = action.get("command", "")
            cwd = cwd or self.config.cwd
            assert self.container_id, "Container not started"
            def argv(c):
                cmd = [self.config.executable, "exec", "-w", cwd]
                for key in self.config.forward_env:
                    if (value := os.getenv(key)) is not None:
                        cmd.extend(["-e", f"{key}={value}"])
                return cmd + [self.container_id, "bash", "-lc", c]
            return self.sidecar_execute(command, cwd, timeout or self.config.timeout, argv)
else:
    def __getattr__(name):  # pragma: no cover
        if name in ("SidecarLocalEnvironment", "SidecarDockerEnvironment"):
            raise ImportError(f"{name} needs mini-swe-agent: uv sync --extra harness ({_import_error})")
        raise AttributeError(name)
```

- [ ] **Step 4: Run tests**

Run: `uv sync --extra dev --extra serve --extra harness && uv run pytest tests/sidecar -q`
Expected: PASS (all sidecar tests, the last one runs because the harness extra is installed)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/sidecar tests/sidecar uv.lock
git commit -m "feat: launch gate client and mini-SWE-agent sidecar environments"
```

---

### Task 4: Harness proxy

**Files:**
- Create: `src/atfm/proxy/__init__.py`, `src/atfm/proxy/config.py`, `src/atfm/proxy/index.py`, `src/atfm/proxy/queue.py`, `src/atfm/proxy/app.py`, `scripts/run_proxy.py`
- Test: `tests/proxy/test_index.py`, `tests/proxy/test_queue.py`, `tests/proxy/test_app.py`

**Interfaces:**
- Produces:
  - `ProxyConfig(upstream_url: str, window: int = 8, w_interactive: float = 10.0, w_background: float = 1.0, beta: float = 0.0, slack_threshold_s: float = 5.0, prefill_tps: float = 20000.0, decode_tps: float = 60.0, default_osl: int = 256, max_hold_s: float = 600.0, board_timeout_s: float = 0.05, events_path: str | None = None, trace_path: str | None = None)`.
  - `CallMeta(session_id, cls, tenant, deadline, parent, turn_index, isl, predicted_osl, t_arrival)`; `estimate_isl(body: dict) -> int` = total characters of all message contents divided by 4, at least 1.
  - `compute_index(meta, cfg, e_service_s: float, e_tool_next_s: float) -> float` = `w(cls) * (1 + beta * e_tool_next_s) / e_service_s`; `service_time(meta, cfg) -> float` = `isl/prefill_tps + predicted_osl/decode_tps`; `tier(meta, cfg, now, e_service_s) -> int` (2 slack, 1 interactive, 0 background); `priority_bucket(index, tier_indices: list[float]) -> int` = rank quartile within the tier's current indices (0..3, 3 = highest).
  - `HoldQueue(window: int, clock)`: async; `submit(entry: Entry) -> None`, `await entry.released.wait()`, `release_one()` picks the highest `(tier, index, -t_arrival)` among entries whose `not_before <= now`, if `in_flight < window`; `complete()` decrements `in_flight` and triggers `release_one()`; `set_directive(session_id, release_not_before, reason)`; `stats() -> dict(queued, in_flight, held)`; holds are capped: `not_before = min(release_not_before, t_arrival + max_hold_s)`.
  - `create_app(cfg: ProxyConfig, *, upstream_client: httpx.AsyncClient | None = None, bus=None, predictor=None, clock=time.time) -> FastAPI` with routes `POST /v1/chat/completions` (JSON and SSE streaming passthrough), `POST /gate`, `POST /directives`, `GET /state`, `GET /healthz`. Hints injected as `body["nvext"]["agent_hints"] = {"priority": bucket, "strict_priority": tier, "osl": predicted_osl}`; header `x-dynamo-session-id` set to the session id. Events: `llm.request` (with `hints`, `held_s`), `llm.first_token`, `llm.done` (osl from `usage.completion_tokens` when present, `status`). Trace rows appended to `trace_path` as JSON lines with the TraceRow fields the proxy knows (session, class, tenant, turn, t_request, t_first_token, t_last_token, isl, osl).
  - Default hints when no predictor: `e_tool_next_s = 0`, `e_service_s` from `service_time`. With a predictor object exposing `expected_tool_next(session_id) -> float` and `expected_service(session_id, isl, osl) -> float`, those are used, each call bounded by `board_timeout_s` (run in a thread with a timeout; on timeout fall back to defaults).

- [ ] **Step 1: Write the failing tests**

```python
# tests/proxy/test_index.py
from atfm.proxy.config import ProxyConfig
from atfm.proxy.index import CallMeta, estimate_isl, service_time, compute_index, tier, priority_bucket

def _meta(cls="background", deadline=None, isl=2000, osl=100):
    return CallMeta(session_id="s", cls=cls, tenant="t", deadline=deadline, parent=None, turn_index=0,
                    isl=isl, predicted_osl=osl, t_arrival=1000.0)

def test_estimate_isl_from_messages():
    assert estimate_isl({"messages": [{"role": "user", "content": "a" * 400}, {"role": "system", "content": "b" * 40}]}) == 110
    assert estimate_isl({"messages": []}) == 1

def test_service_time_and_index():
    cfg = ProxyConfig(upstream_url="http://u", beta=0.5)
    m = _meta(isl=20000, osl=60)
    assert abs(service_time(m, cfg) - 2.0) < 1e-9
    assert abs(compute_index(m, cfg, e_service_s=2.0, e_tool_next_s=4.0) - 1.0 * 3.0 / 2.0) < 1e-9
    assert compute_index(_meta("interactive", isl=20000, osl=60), cfg, 2.0, 0.0) == 5.0

def test_tiers():
    cfg = ProxyConfig(upstream_url="http://u", slack_threshold_s=5.0)
    assert tier(_meta("background"), cfg, now=1000.0, e_service_s=1.0) == 0
    assert tier(_meta("interactive"), cfg, now=1000.0, e_service_s=1.0) == 1
    assert tier(_meta("interactive", deadline=1004.0), cfg, now=1000.0, e_service_s=1.0) == 2

def test_priority_bucket_quartiles():
    assert priority_bucket(10.0, [1.0, 2.0, 3.0, 10.0]) == 3
    assert priority_bucket(1.0, [1.0, 2.0, 3.0, 10.0]) == 0
    assert priority_bucket(5.0, []) == 3
```

```python
# tests/proxy/test_queue.py
import asyncio, pytest
from atfm.proxy.queue import HoldQueue, Entry

def _entry(sid, tier, index, t):
    return Entry(session_id=sid, tier=tier, index=index, t_arrival=t)

@pytest.mark.asyncio
async def test_window_and_priority_order():
    now = [100.0]
    q = HoldQueue(window=1, clock=lambda: now[0], max_hold_s=600.0)
    a, b, c, d = _entry("a", 0, 1.0, 1.0), _entry("b", 0, 2.0, 2.0), _entry("c", 0, 3.0, 3.0), _entry("d", 2, 0.5, 4.0)
    for e in (a, b, c, d):
        q.submit(e)
    await asyncio.wait_for(a.released.wait(), 1.0)       # first arrival takes the single slot
    assert not b.released.is_set() and not d.released.is_set()
    q.complete()                                          # slot frees: interactive-under-slack d goes next despite lowest index
    await asyncio.wait_for(d.released.wait(), 1.0)
    assert not c.released.is_set()
    q.complete()
    await asyncio.wait_for(c.released.wait(), 1.0)        # then highest index among background
    assert not b.released.is_set()

@pytest.mark.asyncio
async def test_directive_hold_and_cap():
    now = [100.0]
    q = HoldQueue(window=4, clock=lambda: now[0], max_hold_s=10.0)
    q.set_directive("h", release_not_before=100000.0, reason="gdp")
    e = _entry("h", 0, 1.0, 100.0)
    q.submit(e)
    assert not e.released.is_set() and q.stats()["held"] == 1
    assert e.not_before == 110.0                          # capped at t_arrival + max_hold_s
    now[0] = 111.0
    q.tick()
    await asyncio.wait_for(e.released.wait(), 1.0)
```

```python
# tests/proxy/test_app.py
import asyncio, json, time
import httpx, pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from atfm.bus import InMemoryBus
from atfm.proxy.config import ProxyConfig
from atfm.proxy.app import create_app

def _upstream(delay_s=0.0, fail=False):
    up = FastAPI()
    up.state.seen = []
    @up.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        up.state.seen.append({"body": body, "headers": dict(req.headers)})
        await asyncio.sleep(delay_s)
        if fail:
            return JSONResponse({"error": "boom"}, status_code=500)
        return JSONResponse({"id": "x", "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 7}})
    return up

def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")

def _cfg(**kw):
    return ProxyConfig(upstream_url="http://up", **kw)

def _headers(sid, cls="background", deadline=None):
    h = {"x-atfm-session": sid, "x-atfm-class": cls, "x-atfm-tenant": "t1"}
    if deadline is not None:
        h["x-atfm-deadline"] = str(deadline)
    return h

BODY = {"model": "m", "messages": [{"role": "user", "content": "hello " * 50}], "max_tokens": 32}

@pytest.mark.asyncio
async def test_hints_injected_body_otherwise_unchanged():
    up = _upstream(); bus = InMemoryBus()
    app = create_app(_cfg(window=4), upstream_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=up), base_url="http://up"), bus=bus)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=BODY, headers=_headers("s1", "interactive"))
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "ok"
    sent = up.state.seen[0]
    assert sent["body"]["messages"] == BODY["messages"] and sent["body"]["max_tokens"] == 32
    assert sent["body"]["nvext"]["agent_hints"] == {"priority": 3, "strict_priority": 1, "osl": 32}
    assert sent["headers"]["x-dynamo-session-id"] == "s1"
    kinds = [e.kind for e in bus.drain()]
    assert kinds == ["session.start", "llm.request", "llm.first_token", "llm.done"]

@pytest.mark.asyncio
async def test_window_orders_interactive_slack_first():
    up = _upstream(delay_s=0.3)
    app = create_app(_cfg(window=1, slack_threshold_s=5.0), upstream_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=up), base_url="http://up"))
    async with _client(app) as c:
        first = asyncio.create_task(c.post("/v1/chat/completions", json=BODY, headers=_headers("bg0")))
        await asyncio.sleep(0.05)
        bgs = [asyncio.create_task(c.post("/v1/chat/completions", json=BODY, headers=_headers(f"bg{i}"))) for i in (1, 2, 3)]
        await asyncio.sleep(0.05)
        it = asyncio.create_task(c.post("/v1/chat/completions", json=BODY, headers=_headers("int", "interactive", deadline=time.time() + 2)))
        await asyncio.gather(first, *bgs, it)
    order = [s["headers"]["x-atfm-session"] for s in up.state.seen]
    assert order[0] == "bg0" and order[1] == "int"
    assert up.state.seen[1]["body"]["nvext"]["agent_hints"]["strict_priority"] == 2

@pytest.mark.asyncio
async def test_directive_hold_capped_and_upstream_error_passthrough():
    up = _upstream(fail=True); bus = InMemoryBus()
    app = create_app(_cfg(window=4, max_hold_s=0.2), upstream_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=up), base_url="http://up"), bus=bus)
    async with _client(app) as c:
        r = await c.post("/directives", json={"session_id": "h1", "release_not_before": time.time() + 60, "reason": "gdp"})
        assert r.status_code == 200
        t0 = time.time()
        r = await c.post("/v1/chat/completions", json=BODY, headers=_headers("h1"))
        assert 0.15 <= time.time() - t0 < 2.0             # held for max_hold_s, not 60 s
        assert r.status_code == 500
        st = (await c.get("/state")).json()
        assert st["caps"] == 1
    ev = bus.drain()
    req = [e for e in ev if e.kind == "llm.request"][0]
    assert req.held_s >= 0.15
    assert [e for e in ev if e.kind == "llm.done"][0].status == 500

@pytest.mark.asyncio
async def test_gate_endpoint_reports_directive():
    app = create_app(_cfg(window=4))
    async with _client(app) as c:
        assert (await c.post("/gate", json={"session_id": "z", "kind": "tool"})).json()["allowed_at"] is None
        await c.post("/directives", json={"session_id": "z", "release_not_before": 5000.0, "reason": "gdp"})
        assert (await c.post("/gate", json={"session_id": "z", "kind": "tool"})).json()["allowed_at"] == 5000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/proxy -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/proxy/__init__.py
```

```python
# src/atfm/proxy/config.py
from __future__ import annotations
from pydantic import BaseModel

class ProxyConfig(BaseModel):
    upstream_url: str
    window: int = 8
    w_interactive: float = 10.0
    w_background: float = 1.0
    beta: float = 0.0
    slack_threshold_s: float = 5.0
    prefill_tps: float = 20000.0
    decode_tps: float = 60.0
    default_osl: int = 256
    max_hold_s: float = 600.0
    board_timeout_s: float = 0.05
    events_path: str | None = None
    trace_path: str | None = None
```

```python
# src/atfm/proxy/index.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import ProxyConfig

@dataclass
class CallMeta:
    session_id: str
    cls: str
    tenant: str
    deadline: float | None
    parent: str | None
    turn_index: int
    isl: int
    predicted_osl: int
    t_arrival: float

def estimate_isl(body: dict) -> int:
    chars = 0
    for m in body.get("messages", []) or []:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            chars += sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
    return max(1, chars // 4)

def service_time(meta: CallMeta, cfg: ProxyConfig) -> float:
    return meta.isl / cfg.prefill_tps + meta.predicted_osl / cfg.decode_tps

def weight(cls: str, cfg: ProxyConfig) -> float:
    return cfg.w_interactive if cls == "interactive" else cfg.w_background

def compute_index(meta: CallMeta, cfg: ProxyConfig, e_service_s: float, e_tool_next_s: float) -> float:
    return weight(meta.cls, cfg) * (1.0 + cfg.beta * e_tool_next_s) / max(e_service_s, 1e-6)

def tier(meta: CallMeta, cfg: ProxyConfig, now: float, e_service_s: float) -> int:
    if meta.cls != "interactive":
        return 0
    if meta.deadline is not None and (meta.deadline - now - e_service_s) < cfg.slack_threshold_s:
        return 2
    return 1

def priority_bucket(index: float, tier_indices: list[float]) -> int:
    if not tier_indices:
        return 3
    rank = float(np.mean(np.asarray(tier_indices) <= index))
    return min(3, int(rank * 4)) if rank < 1.0 else 3
```

```python
# src/atfm/proxy/queue.py
from __future__ import annotations
import asyncio, time
from dataclasses import dataclass, field

@dataclass
class Entry:
    session_id: str
    tier: int
    index: float
    t_arrival: float
    not_before: float = 0.0
    released: asyncio.Event = field(default_factory=asyncio.Event)
    t_release: float | None = None

class HoldQueue:
    """Global admission window with tier-then-index ordering and per-session hold directives."""

    def __init__(self, window: int, clock=time.time, max_hold_s: float = 600.0):
        self.window, self.clock, self.max_hold_s = window, clock, max_hold_s
        self.in_flight = 0
        self.pending: list[Entry] = []
        self.directives: dict[str, tuple[float, str]] = {}
        self.caps = 0
        self._timer: asyncio.TimerHandle | None = None

    def set_directive(self, session_id: str, release_not_before: float, reason: str) -> None:
        self.directives[session_id] = (release_not_before, reason)

    def directive_for(self, session_id: str) -> float | None:
        d = self.directives.get(session_id)
        return None if d is None else d[0]

    def submit(self, e: Entry) -> None:
        d = self.directive_for(e.session_id)
        if d is not None and d > e.t_arrival:
            cap = e.t_arrival + self.max_hold_s
            if d > cap:
                self.caps += 1
            e.not_before = min(d, cap)
        self.pending.append(e)
        self.tick()

    def complete(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        self.tick()

    def tick(self) -> None:
        now = self.clock()
        while self.in_flight < self.window:
            eligible = [e for e in self.pending if e.not_before <= now]
            if not eligible:
                break
            best = max(eligible, key=lambda e: (e.tier, e.index, -e.t_arrival))
            self.pending.remove(best)
            self.in_flight += 1
            best.t_release = now
            best.released.set()
        self._arm_timer(now)

    def _arm_timer(self, now: float) -> None:
        future = [e.not_before for e in self.pending if e.not_before > now]
        if not future:
            return
        delay = max(0.0, min(future) - now)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._timer is not None:
            self._timer.cancel()
        self._timer = loop.call_later(delay + 1e-3, self.tick)

    def stats(self) -> dict:
        now = self.clock()
        return {"queued": len(self.pending), "in_flight": self.in_flight,
                "held": sum(1 for e in self.pending if e.not_before > now), "caps": self.caps}

    def tier_indices(self, tier: int) -> list[float]:
        return [e.index for e in self.pending if e.tier == tier]
```

```python
# src/atfm/proxy/app.py
from __future__ import annotations
import asyncio, json, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from atfm.bus import InMemoryBus, JsonlBus
from atfm.schema.events import LlmDone, LlmFirstToken, LlmRequest, SessionStart
from .config import ProxyConfig
from .index import CallMeta, compute_index, estimate_isl, priority_bucket, service_time, tier
from .queue import Entry, HoldQueue

def _meta_from(req: Request, body: dict, cfg: ProxyConfig, now: float, turn_index: int) -> CallMeta:
    h = req.headers
    sid = h.get("x-atfm-session") or f"anon-{uuid.uuid4().hex[:8]}"
    cls = h.get("x-atfm-class", "background")
    cls = cls if cls in ("interactive", "background") else "background"
    dl = h.get("x-atfm-deadline")
    osl = int(body.get("max_tokens") or cfg.default_osl)
    return CallMeta(session_id=sid, cls=cls, tenant=h.get("x-atfm-tenant", "t0"),
                    deadline=float(dl) if dl else None, parent=h.get("x-atfm-parent"), turn_index=turn_index,
                    isl=estimate_isl(body), predicted_osl=osl, t_arrival=now)

def create_app(cfg: ProxyConfig, *, upstream_client: httpx.AsyncClient | None = None, bus=None, predictor=None,
               clock=time.time) -> FastAPI:
    app = FastAPI()
    st = app.state
    st.cfg = cfg
    st.bus = bus if bus is not None else (JsonlBus(cfg.events_path) if cfg.events_path else InMemoryBus())
    st.client = upstream_client or httpx.AsyncClient(base_url=cfg.upstream_url, timeout=httpx.Timeout(600.0))
    st.queue = HoldQueue(cfg.window, clock=clock, max_hold_s=cfg.max_hold_s)
    st.predictor = predictor
    st.pool = ThreadPoolExecutor(max_workers=4)
    st.turns: dict[str, int] = {}
    st.known: set[str] = set()
    st.trace = open(cfg.trace_path, "a") if cfg.trace_path else None

    def predict(meta: CallMeta) -> tuple[float, float]:
        e_service = service_time(meta, cfg)
        e_tool = 0.0
        if st.predictor is None:
            return e_service, e_tool
        def _call():
            return (float(st.predictor.expected_service(meta.session_id, meta.isl, meta.predicted_osl)),
                    float(st.predictor.expected_tool_next(meta.session_id)))
        fut = st.pool.submit(_call)
        try:
            return fut.result(timeout=cfg.board_timeout_s)
        except Exception:
            return e_service, e_tool

    def emit(e) -> None:
        try:
            st.bus.publish(e)
        except Exception:
            pass

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/state")
    async def state():
        return st.queue.stats()

    @app.post("/directives")
    async def directives(req: Request):
        d = await req.json()
        st.queue.set_directive(d["session_id"], float(d["release_not_before"]), d.get("reason", ""))
        st.queue.tick()
        return {"ok": True}

    @app.post("/gate")
    async def gate(req: Request):
        d = await req.json()
        return {"allowed_at": st.queue.directive_for(d["session_id"])}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        now = clock()
        turn = st.turns.get(req.headers.get("x-atfm-session", ""), 0)
        meta = _meta_from(req, body, cfg, now, turn)
        st.turns[meta.session_id] = turn + 1
        if meta.session_id not in st.known:
            st.known.add(meta.session_id)
            emit(SessionStart(t=now, session_id=meta.session_id, tenant=meta.tenant, cls=meta.cls,
                              parent_session_id=meta.parent, deadline=meta.deadline))
        e_service, e_tool = predict(meta)
        idx = compute_index(meta, cfg, e_service, e_tool)
        tr = tier(meta, cfg, now, e_service)
        entry = Entry(session_id=meta.session_id, tier=tr, index=idx, t_arrival=now)
        st.queue.submit(entry)
        await entry.released.wait()
        t_rel = clock()
        bucket = priority_bucket(idx, st.queue.tier_indices(tr) + [idx])
        hints = {"priority": bucket, "strict_priority": tr, "osl": meta.predicted_osl}
        body = dict(body)
        body["nvext"] = dict(body.get("nvext") or {})
        body["nvext"]["agent_hints"] = hints
        rid = uuid.uuid4().hex[:16]
        emit(LlmRequest(t=t_rel, session_id=meta.session_id, turn_index=meta.turn_index, request_id=rid,
                        isl=meta.isl, predicted_osl=meta.predicted_osl, hints=hints, held_s=t_rel - now))
        headers = {"content-type": "application/json", "x-dynamo-session-id": meta.session_id}
        stream = bool(body.get("stream"))
        t_first = None
        try:
            if not stream:
                r = await st.client.post("/v1/chat/completions", json=body, headers=headers)
                t_first = clock()
                osl = 0
                try:
                    osl = int(r.json().get("usage", {}).get("completion_tokens", 0))
                except Exception:
                    pass
                _finish(meta, rid, now, t_rel, t_first, clock(), osl, r.status_code)
                return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
            upstream = await st.client.send(st.client.build_request("POST", "/v1/chat/completions", json=body, headers=headers), stream=True)
            async def gen():
                nonlocal t_first
                osl = 0
                try:
                    async for chunk in upstream.aiter_bytes():
                        if t_first is None:
                            t_first = clock()
                            emit(LlmFirstToken(t=t_first, session_id=meta.session_id, request_id=rid))
                        osl += chunk.count(b"data:")
                        yield chunk
                finally:
                    await upstream.aclose()
                    _finish(meta, rid, now, t_rel, t_first, clock(), osl, upstream.status_code, first_emitted=True)
            return StreamingResponse(gen(), status_code=upstream.status_code, media_type=upstream.headers.get("content-type", "text/event-stream"))
        except httpx.HTTPError as e:
            _finish(meta, rid, now, t_rel, None, clock(), 0, 502)
            return JSONResponse({"error": f"upstream unavailable: {e}"}, status_code=502)

    def _finish(meta: CallMeta, rid: str, t_arr: float, t_rel: float, t_first, t_last: float, osl: int, status: int,
                first_emitted: bool = False) -> None:
        st.queue.complete()
        if t_first is not None and not first_emitted:
            emit(LlmFirstToken(t=t_first, session_id=meta.session_id, request_id=rid))
        emit(LlmDone(t=t_last, session_id=meta.session_id, request_id=rid, osl=osl, status=status))
        if st.trace is not None:
            row = {"session_id": meta.session_id, "parent_session_id": meta.parent, "class": meta.cls, "tenant": meta.tenant,
                   "turn_index": meta.turn_index, "t_request": t_arr, "t_release": t_rel, "t_first_token": t_first,
                   "t_last_token": t_last, "isl": meta.isl, "osl": osl, "status": status}
            st.trace.write(json.dumps(row) + "\n")
            st.trace.flush()

    return app
```

```python
# scripts/run_proxy.py
import argparse, uvicorn
from atfm.proxy.app import create_app
from atfm.proxy.config import ProxyConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--events", default="runs/proxy/events.jsonl")
    ap.add_argument("--trace", default="runs/proxy/calls.jsonl")
    a = ap.parse_args()
    cfg = ProxyConfig(upstream_url=a.upstream, window=a.window, events_path=a.events, trace_path=a.trace)
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=a.port, log_level="warning")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/proxy -q`
Expected: PASS (9 tests). If `pytest-asyncio` needs configuration, add `asyncio_mode = "auto"` under `[tool.pytest.ini_options]`.

- [ ] **Step 5: Commit**

```bash
git add src/atfm/proxy scripts/run_proxy.py tests/proxy pyproject.toml
git commit -m "feat: harness proxy with global window, priority tiers, hints, events and gate"
```

---

### Task 5: Live board and sidecar-log trace adapter

**Files:**
- Create: `src/atfm/board/live.py`, `src/atfm/traces/sidecar.py`, `scripts/run_board.py`
- Test: `tests/board/test_live.py`, `tests/traces/test_sidecar_adapter.py`

**Interfaces:**
- Consumes: `SessionState`, `SessionForecaster`, `ExogenousModel`, `SessionPredictor`, events, buses, `TraceTable`, `TraceRow`.
- Produces:
  - `SessionRegistry` with `apply(e: Event) -> None`, `states(now: float) -> list[SessionState]` (sessions inactive for more than `expire_s`, default 7200, are dropped), `new_starts_since(t: float) -> list[tuple[float, str]]`. Transitions: `session.start` creates (phase `llm_pending`); `llm.request` -> `llm_running` with `t_phase_start`, `turn_index`, `ctx_tokens = isl + predicted_osl`; `llm.done` -> `llm_pending` (`t_phase_start = t`); `tool.start` -> `tool_running` (`tool_name, backend_id, t_tool_start = t`, clears progress/data); `tool.progress` / `tool.data` append dicts `{t, completed, total, phase}` / `{t, metric, value}`; `tool.end` -> `llm_pending` and appends `(tool_name, t - t_tool_start)` to `tool_history`. Unknown sessions are created on first sight with class `background`, tenant `unknown`. Events out of order (progress before start) attach to the session's current tool if any, else create a placeholder tool `unknown`.
  - `LiveBoard(registry, forecaster: SessionForecaster, tick_s=5.0)` with `step(now, rng) -> ForecastSnapshot` (updates `forecaster.exo` with new starts, calls `forecaster.forecast(now, registry.states(now), rng)`), `expected_tool_next(session_id) -> float` (mean tool duration from the predictor's `DurationModel` for the session's most recent tool, else pooled mean) and `expected_service(session_id, isl, osl) -> float` (uses `forecaster` prefill/decode rates if provided; default 20000 tps / 60 tps).
  - `run_board(events_path, snapshots_path, predictor, train_table, tick_s, horizons, n, once=False)` in `scripts/run_board.py`: tails the JSONL bus, applies events, writes one snapshot JSON line per tick `{t, model_id, horizons, q50, q90 per target/class, endogenous_fraction}`.
  - `events_to_trace_table(events: list[Event]) -> TraceTable` in `atfm.traces.sidecar`: one row per `llm.request` with `t_request`, `t_first_token`, `t_last_token` from its `llm.first_token` / `llm.done`, `isl`, `osl`; the row's tool phase is the first `tool.start` in that session after the request and before the next request (`t_tool_start`, `t_tool_end`, `tool_name`, `backend_id`, progress and data events, `tool_exit_status`); `source = "sidecar"`; class and tenant from `session.start`. Sessions with tool events but no LLM events (tool-only collection) get synthetic LLM rows: `t_request = t_tool_start - 0.01`, `t_first_token = t_last_token = t_tool_start`, `isl = 1`, `osl = 0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/board/test_live.py
import numpy as np
from atfm.schema.events import parse_event
from atfm.board.live import SessionRegistry, LiveBoard
from atfm.board.forecaster import SessionForecaster, ExogenousModel
from atfm.board.predictors import SurvivalPredictor
from atfm.schema.trace import TraceRow, TraceTable

def _ev(**d):
    return parse_event(d)

def test_registry_transitions_and_unknown_sessions():
    r = SessionRegistry()
    r.apply(_ev(kind="tool.progress", t=5.0, session_id="ghost", call_id="c0", completed=1, total=3))   # out of order
    s = {x.session_id: x for x in r.states(5.0)}["ghost"]
    assert s.phase == "tool_running" and s.tool_name == "unknown" and s.progress[0]["completed"] == 1
    r.apply(_ev(kind="session.start", t=0.0, session_id="a", tenant="t", **{"class": "background"}))
    r.apply(_ev(kind="llm.request", t=1.0, session_id="a", turn_index=0, request_id="r1", isl=1000, predicted_osl=100))
    assert {x.session_id: x for x in r.states(1.5)}["a"].phase == "llm_running"
    r.apply(_ev(kind="llm.done", t=2.0, session_id="a", request_id="r1", osl=80))
    r.apply(_ev(kind="tool.start", t=2.1, session_id="a", turn_index=0, call_id="c1", tool_name="pytest", backend_id="ci"))
    r.apply(_ev(kind="tool.progress", t=12.1, session_id="a", call_id="c1", completed=10, total=100, phase="run"))
    s = {x.session_id: x for x in r.states(15.0)}["a"]
    assert s.phase == "tool_running" and s.elapsed(15.0) == 12.9 and s.progress[-1]["total"] == 100 and s.ctx_tokens == 1100
    r.apply(_ev(kind="tool.end", t=62.1, session_id="a", call_id="c1", exit_status=0))
    s = {x.session_id: x for x in r.states(63.0)}["a"]
    assert s.phase == "llm_pending" and s.tool_history == [("pytest", 60.0)]
    assert r.new_starts_since(0.0) == [(0.0, "background")]
    assert all(x.session_id != "ghost" for x in r.states(5.0 + 7201.0))

def _train():
    rows = []
    for k in range(10):
        t = k * 500.0
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=0, t_request=t, t_first_token=t + 1,
                             t_last_token=t + 2, isl=160, osl=16, tool_name="pytest", t_tool_start=t + 2, t_tool_end=t + 62, source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=1, t_request=t + 62, t_first_token=t + 63,
                             t_last_token=t + 64, isl=320, osl=16, tool_name=None, source="test"))
    return TraceTable.from_rows(rows)

def test_live_board_step_and_predictions():
    tr = _train()
    pred = SurvivalPredictor().fit(tr)
    board = LiveBoard(SessionRegistry(), SessionForecaster(pred, ExogenousModel().fit(tr), horizons=[10.0, 120.0], n=32))
    board.registry.apply(_ev(kind="session.start", t=0.0, session_id="a", tenant="t", **{"class": "background"}))
    board.registry.apply(_ev(kind="tool.start", t=1.0, session_id="a", turn_index=0, call_id="c", tool_name="pytest", backend_id="ci"))
    snap = board.step(59.0, np.random.default_rng(0))
    assert snap.samples["kv_blocks"]["background"].shape == (2, 32)
    assert abs(board.expected_tool_next("a") - 60.0) < 1e-6 and board.expected_service("a", 20000, 60) == 2.0
```

```python
# tests/traces/test_sidecar_adapter.py
from atfm.schema.events import parse_event
from atfm.traces.sidecar import events_to_trace_table

def _e(**d):
    return parse_event(d)

def test_events_to_trace_table_pairs_calls_and_tools():
    ev = [
        _e(kind="session.start", t=0.0, session_id="a", tenant="t1", **{"class": "background"}),
        _e(kind="llm.request", t=1.0, session_id="a", turn_index=0, request_id="r1", isl=1000),
        _e(kind="llm.first_token", t=1.5, session_id="a", request_id="r1"),
        _e(kind="llm.done", t=2.0, session_id="a", request_id="r1", osl=50),
        _e(kind="tool.start", t=2.2, session_id="a", turn_index=0, call_id="c1", tool_name="pytest", backend_id="ci"),
        _e(kind="tool.progress", t=12.2, session_id="a", call_id="c1", completed=5, total=20, phase="run"),
        _e(kind="tool.data", t=15.0, session_id="a", call_id="c1", metric="lines_per_s", value=3.0),
        _e(kind="tool.end", t=42.2, session_id="a", call_id="c1", exit_status=1),
        _e(kind="llm.request", t=42.5, session_id="a", turn_index=1, request_id="r2", isl=1200),
        _e(kind="llm.done", t=44.0, session_id="a", request_id="r2", osl=10),
        _e(kind="tool.start", t=100.0, session_id="b", turn_index=0, call_id="c9", tool_name="build", backend_id="ci"),
        _e(kind="tool.end", t=160.0, session_id="b", call_id="c9", exit_status=0),
    ]
    t = events_to_trace_table(ev)
    df = t.df
    a = df[df.session_id == "a"].sort_values("turn_index")
    assert len(a) == 2 and a.iloc[0]["tool_name"] == "pytest" and a.iloc[0]["t_tool_end"] == 42.2
    assert a.iloc[0]["progress_events"][0]["completed"] == 5 and a.iloc[0]["data_events"][0]["metric"] == "lines_per_s"
    assert a.iloc[0]["tool_exit_status"] == 1 and a.iloc[0]["osl"] == 50 and a.iloc[0]["class"] == "background"
    assert a.iloc[1]["tool_name"] is None
    b = df[df.session_id == "b"]
    assert len(b) == 1 and b.iloc[0]["isl"] == 1 and b.iloc[0]["t_tool_start"] == 100.0 and b.iloc[0]["source"] == "sidecar"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/board/test_live.py tests/traces/test_sidecar_adapter.py -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/board/live.py
from __future__ import annotations
import numpy as np
from atfm.board.forecaster import SessionForecaster
from atfm.board.state import SessionState
from atfm.schema.events import Event
from atfm.schema.forecast import ForecastSnapshot

class SessionRegistry:
    def __init__(self, expire_s: float = 7200.0):
        self.expire_s = expire_s
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
        k = e.kind
        if k == "session.start":
            s = self._get(e.session_id, e.t)
            s.cls, s.tenant, s.parent_session_id = e.cls, e.tenant, e.parent_session_id
            self._starts.append((e.t, e.cls))
        elif k == "llm.request":
            s = self._get(e.session_id, e.t)
            s.phase, s.t_phase_start, s.turn_index = "llm_running", e.t, e.turn_index
            s.ctx_tokens = int(e.isl) + int(e.predicted_osl or 0)
        elif k == "llm.done":
            s = self._get(e.session_id, e.t)
            s.phase, s.t_phase_start = "llm_pending", e.t
        elif k == "tool.start":
            s = self._get(e.session_id, e.t)
            s.phase, s.tool_name, s.backend_id, s.t_tool_start, s.t_phase_start = "tool_running", e.tool_name, e.backend_id, e.t, e.t
            s.progress, s.data = [], []
        elif k in ("tool.progress", "tool.data"):
            s = self._get(e.session_id, e.t)
            if s.phase != "tool_running":
                s.phase, s.tool_name, s.backend_id, s.t_tool_start, s.t_phase_start = "tool_running", "unknown", "local", e.t, e.t
                s.progress, s.data = [], []
            if k == "tool.progress":
                s.progress.append({"t": e.t, "completed": e.completed, "total": e.total, "phase": e.phase})
            else:
                s.data.append({"t": e.t, "metric": e.metric, "value": e.value})
        elif k == "tool.end":
            s = self._get(e.session_id, e.t)
            if s.phase == "tool_running" and s.t_tool_start is not None:
                s.tool_history.append((s.tool_name or "unknown", max(0.0, e.t - s.t_tool_start)))
            s.phase, s.t_phase_start = "llm_pending", e.t
        elif k == "spawn.request":
            self._get(e.child_session_id, e.t).parent_session_id = e.parent_session_id

    def states(self, now: float) -> list[SessionState]:
        for sid in [s for s, t in self._last.items() if now - t > self.expire_s]:
            self._s.pop(sid, None); self._last.pop(sid, None)
        return list(self._s.values())

    def new_starts_since(self, t: float) -> list[tuple[float, str]]:
        return [x for x in self._starts if x[0] >= t]

class LiveBoard:
    def __init__(self, registry: SessionRegistry, forecaster: SessionForecaster, tick_s: float = 5.0,
                 prefill_tps: float = 20000.0, decode_tps: float = 60.0):
        self.registry, self.forecaster, self.tick_s = registry, forecaster, tick_s
        self.prefill_tps, self.decode_tps = prefill_tps, decode_tps
        self._starts_ptr = 0.0

    def step(self, now: float, rng: np.random.Generator) -> ForecastSnapshot:
        starts = [s for s in self.registry.new_starts_since(self._starts_ptr) if s[0] < now]
        self._starts_ptr = now
        self.forecaster.exo.update(now, starts)
        return self.forecaster.forecast(now, self.registry.states(now), rng)

    def expected_tool_next(self, session_id: str) -> float:
        dm = getattr(self.forecaster.predictor, "dm", None)
        if dm is None:
            return 0.0
        s = self.registry._s.get(session_id)
        tool = None
        if s is not None:
            tool = s.tool_name if s.phase == "tool_running" else (s.tool_history[-1][0] if s.tool_history else None)
        return dm.mean(tool)

    def expected_service(self, session_id: str, isl: int, osl: int) -> float:
        return isl / self.prefill_tps + osl / self.decode_tps
```

```python
# src/atfm/traces/sidecar.py
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
            meta.setdefault(e.child_session_id, {"class": "background", "tenant": "unknown", "parent": None})["parent"] = e.parent_session_id
        else:
            by_sess[e.session_id].append(e)
    rows: list[TraceRow] = []
    for sid, evs in by_sess.items():
        m = meta.get(sid, {"class": "background", "tenant": "unknown", "parent": None})
        calls = [e for e in evs if e.kind == "llm.request"]
        tools = _tools(evs)
        if not calls:  # tool-only session: synthesize one LLM row per tool phase
            for i, tl in enumerate(tools):
                rows.append(_row(sid, m, i, tl["t_start"] - 0.01, tl["t_start"], tl["t_start"], 1, 0, tl))
            continue
        for i, c in enumerate(calls):
            first = next((e.t for e in evs if e.kind == "llm.first_token" and e.request_id == c.request_id), None)
            done = next((e for e in evs if e.kind == "llm.done" and e.request_id == c.request_id), None)
            t_last = done.t if done else first
            nxt = calls[i + 1].t if i + 1 < len(calls) else float("inf")
            tl = next((t for t in tools if c.t <= t["t_start"] < nxt), None)
            rows.append(_row(sid, m, c.turn_index, c.t, first, t_last, c.isl, done.osl if done else 0, tl))
    return TraceTable.from_rows(rows)

def _tools(evs: list) -> list[dict]:
    out: dict[str, dict] = {}
    for e in evs:
        if e.kind == "tool.start":
            out[e.call_id] = {"t_start": e.t, "t_end": None, "name": e.tool_name, "backend": e.backend_id,
                              "progress": [], "data": [], "exit": None}
        elif e.call_id in out:
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
        kw.update(tool_name=tl["name"], backend_id=tl["backend"], t_tool_start=tl["t_start"],
                  t_tool_end=tl["t_end"] if tl["t_end"] is not None else tl["t_start"],
                  tool_exit_status=tl["exit"], progress_events=tl["progress"], data_events=tl["data"])
    return TraceRow(**kw)
```

```python
# scripts/run_board.py
import argparse, json, time
import numpy as np
from atfm.board.forecaster import ExogenousModel, SessionForecaster, CLASSES, TARGETS
from atfm.board.live import LiveBoard, SessionRegistry
from atfm.board.predictors import ProgressPredictor
from atfm.bus import read_events
from atfm.schema.trace import TraceTable

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--train", required=True, help="parquet trace table to fit the predictor on")
    ap.add_argument("--tick", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    train = TraceTable.from_parquet(a.train)
    fc = SessionForecaster(ProgressPredictor().fit(train), ExogenousModel().fit(train), horizons=[10.0, 30.0, 120.0, 300.0, 900.0], n=256)
    board = LiveBoard(SessionRegistry(), fc, tick_s=a.tick)
    rng = np.random.default_rng(0)
    seen = 0
    with open(a.snapshots, "a") as out:
        while True:
            ev = read_events(a.events)
            for e in ev[seen:]:
                board.registry.apply(e)
            seen = len(ev)
            now = time.time()
            snap = board.step(now, rng)
            rec = {"t": now, "model_id": snap.model_id, "horizons": snap.horizons,
                   "q50": {t: {c: snap.quantiles(t, c, 0.5).tolist() for c in CLASSES} for t in TARGETS},
                   "q90": {t: {c: snap.quantiles(t, c, 0.9).tolist() for c in CLASSES} for t in TARGETS},
                   "endogenous_fraction": {c: snap.endogenous_fraction[c].tolist() for c in CLASSES}}
            out.write(json.dumps(rec) + "\n"); out.flush()
            if a.once:
                break
            time.sleep(a.tick)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/board/test_live.py tests/traces/test_sidecar_adapter.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/atfm/board/live.py src/atfm/traces/sidecar.py scripts/run_board.py tests/board/test_live.py tests/traces/test_sidecar_adapter.py
git commit -m "feat: live demand board from bus events and sidecar-log trace adapter"
```

---

### Task 6: Local Dynamo stack (Mocker + frontend) and gated integration test

**Files:**
- Create: `src/atfm/dynamo/__init__.py`, `src/atfm/dynamo/local.py`, `scripts/dynamo_local.py`
- Test: `tests/dynamo/test_local.py`

**Interfaces:**
- Produces: `LocalDynamo(model="Qwen/Qwen3-0.6B", workers=2, port=8000, blocks=4096, speedup=10.0, log_dir="runs/dynamo")` with `start(timeout_s=120) -> None` (launches `python -m dynamo.mocker --model-path <model> --discovery-backend file --num-workers N --num-gpu-blocks-override B --speedup-ratio S` and `python -m dynamo.frontend --discovery-backend file --http-port <port> --router-mode kv`, each with `start_new_session=True`, logs to files, then polls `GET /v1/models` until the model is listed), `stop() -> None` (SIGTERM the process groups, wait 10 s, SIGKILL), context manager, `chat(messages, max_tokens=8, hints=None, session_id=None) -> dict` helper using `urllib`. Both processes are found by `sys.executable -m ...` so they use the project venv (needs the `dynamo` extra).
- CLI `scripts/dynamo_local.py up|down|smoke` writing the PIDs to `runs/dynamo/pids.json`. `smoke` starts, sends one hinted chat request, prints the response and the last frontend log lines, stops.
- Test `tests/dynamo/test_local.py` is skipped unless `ATFM_DYNAMO=1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/dynamo/test_local.py
import os, pytest
pytestmark = pytest.mark.skipif(os.environ.get("ATFM_DYNAMO") != "1", reason="set ATFM_DYNAMO=1 with the dynamo extra installed")

def test_mocker_frontend_serves_hinted_request(tmp_path):
    from atfm.dynamo.local import LocalDynamo
    with LocalDynamo(port=8790, log_dir=str(tmp_path)) as d:
        r = d.chat([{"role": "user", "content": "hello"}], max_tokens=8,
                   hints={"priority": 3, "strict_priority": 1, "osl": 8}, session_id="smoke-1")
        assert "choices" in r and r["choices"][0]["message"]["content"]
```

- [ ] **Step 2: Run test to verify it is skipped, then fails when enabled**

Run: `uv run pytest tests/dynamo -q` (expected: 1 skipped) then `uv sync --extra dev --extra serve --extra harness --extra dynamo && ATFM_DYNAMO=1 uv run pytest tests/dynamo -q`
Expected: FAIL with ModuleNotFoundError for `atfm.dynamo.local`

- [ ] **Step 3: Implement**

```python
# src/atfm/dynamo/__init__.py
```

```python
# src/atfm/dynamo/local.py
from __future__ import annotations
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

class LocalDynamo:
    def __init__(self, model: str = "Qwen/Qwen3-0.6B", workers: int = 2, port: int = 8000, blocks: int = 4096,
                 speedup: float = 10.0, log_dir: str = "runs/dynamo"):
        self.model, self.workers, self.port, self.blocks, self.speedup = model, workers, port, blocks, speedup
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.procs: list[subprocess.Popen] = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _spawn(self, name: str, args: list[str]) -> subprocess.Popen:
        log = open(self.log_dir / f"{name}.log", "a")
        p = subprocess.Popen([sys.executable, "-m", *args], stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, "DYN_HTTP_PORT": str(self.port)})
        self.procs.append(p)
        return p

    def start(self, timeout_s: float = 120.0) -> None:
        self._spawn("mocker", ["dynamo.mocker", "--model-path", self.model, "--discovery-backend", "file",
                               "--num-workers", str(self.workers), "--num-gpu-blocks-override", str(self.blocks),
                               "--speedup-ratio", str(self.speedup)])
        self._spawn("frontend", ["dynamo.frontend", "--discovery-backend", "file", "--http-port", str(self.port),
                                 "--router-mode", "kv"])
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if any(p.poll() is not None for p in self.procs):
                self.stop()
                raise RuntimeError(f"a dynamo process exited early; see {self.log_dir}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=2) as r:
                    models = json.loads(r.read().decode())
                if any(self.model in str(m) for m in models.get("data", [])):
                    return
            except Exception:
                pass
            time.sleep(2.0)
        self.stop()
        raise TimeoutError(f"dynamo frontend did not list {self.model} within {timeout_s}s; see {self.log_dir}")

    def stop(self) -> None:
        for p in self.procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                pass
        deadline = time.time() + 10
        for p in self.procs:
            while p.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.procs = []

    def chat(self, messages: list[dict], max_tokens: int = 8, hints: dict | None = None, session_id: str | None = None) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if hints:
            body["nvext"] = {"agent_hints": hints}
        headers = {"content-type": "application/json"}
        if session_id:
            headers["x-dynamo-session-id"] = session_id
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *a):
        self.stop()
```

```python
# scripts/dynamo_local.py
import json, sys
from pathlib import Path
from atfm.dynamo.local import LocalDynamo

PIDS = Path("runs/dynamo/pids.json")

def main(cmd: str) -> None:
    if cmd == "up":
        d = LocalDynamo(); d.start()
        PIDS.write_text(json.dumps([p.pid for p in d.procs]))
        print(f"up at {d.base_url}; pids {PIDS.read_text()}")
    elif cmd == "down":
        import os, signal
        for pid in json.loads(PIDS.read_text()) if PIDS.exists() else []:
            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                pass
        print("down")
    elif cmd == "smoke":
        with LocalDynamo(port=8790) as d:
            print(json.dumps(d.chat([{"role": "user", "content": "hello"}], hints={"priority": 3, "strict_priority": 1, "osl": 8}, session_id="smoke"), indent=1)[:800])
            print((d.log_dir / "frontend.log").read_text()[-1500:])
    else:
        raise SystemExit("usage: dynamo_local.py up|down|smoke")

if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 4: Run the gated test**

Run: `ATFM_DYNAMO=1 uv run pytest tests/dynamo -q`
Expected: PASS. If the frontend never lists the model, read `runs/dynamo/frontend.log` and `mocker.log`: the file discovery backend may need both processes to share a root directory via an environment variable named in `python -m dynamo.frontend --help`; set it in `_spawn`'s env and retry. Record the exact variable in the ledger.

- [ ] **Step 5: Commit**

```bash
git add src/atfm/dynamo scripts/dynamo_local.py tests/dynamo uv.lock pyproject.toml
git commit -m "feat: local Dynamo Mocker + frontend launcher with gated integration test"
```

---

### Task 7: Scripted trace-collection driver with real long tools

**Files:**
- Create: `src/atfm/collect/__init__.py`, `src/atfm/collect/jobs.py`, `src/atfm/collect/driver.py`, `scripts/collect_traces.py`, `experiments/l1_jobs.yaml`, `scripts/tools/pipeline.py`
- Test: `tests/collect/test_driver.py`

**Interfaces:**
- Produces:
  - `JobSpec(name, cls, tenant, image: str | None, repeat: int = 1, setup: list[str] = [], turns: list[TurnSpec], timeout_s: float = 1800.0, deadline_s: float | None = None)`; `TurnSpec(cmd: str, prompt: str = "continue", max_tokens: int = 16)`; `CollectionSpec(proxy_url: str | None, model: str, events_path: str, jobs: list[JobSpec], seed: int = 0, concurrency: int = 1)`.
  - `run_collection(spec: CollectionSpec, env_factory=None, llm=None, clock=time.time) -> dict` returning `{"sessions": n, "tools": m, "errors": k}`. For each job and repeat: session id `f"{job.name}-{k}-{uuid4[:6]}"`; environment from `env_factory(job, sidecar_cfg)` (default: `SidecarDockerEnvironment(image=job.image, timeout=job.timeout_s, cwd="/w")` when `image` is set, else `SidecarLocalEnvironment(timeout=..., cwd=tempdir)`); runs `setup` commands through the environment (they emit tool events too, class `install`/`clone`); for each turn: `llm(session_id, cls, tenant, deadline, prompt, max_tokens)` then `env.execute({"command": cmd})`; a final `llm(...)` call. `llm` defaults to `proxy_chat(proxy_url, model)` which POSTs to the proxy with the `x-atfm-*` headers; when `proxy_url` is None it is `no_llm` which emits synthetic `llm.request`/`llm.done` pairs with zero duration to the bus so the trace adapter still pairs tools with calls. Jobs run in `concurrency` threads.
  - `scripts/tools/pipeline.py --rows N --rate R --signal strong|weak|none`: prints `processed k/N rows` every second (strong), only `stage k/4` at quarter boundaries (weak), or nothing until the end (none); rate is rows per second with 20% noise; exits 0.
  - `experiments/l1_jobs.yaml` with these jobs (all Docker, `python:3.12-slim` unless stated): `pytest-attrs` (clone `python-attrs/attrs`, `pip install -e .[tests]`, turn: `cd /w/attrs && python -m pytest -q -p no:cacheprovider tests`), `pytest-more-itertools` (clone `more-itertools/more-itertools`, turn: `cd /w/more-itertools && python -m pytest -v -p no:cacheprovider tests`), `build-cjson` (image `gcc:13`, setup `apt-get update -qq && apt-get install -y -qq cmake`, clone `DaveGamble/cJSON`, turn: `cd /w/cJSON && cmake -S . -B build >/dev/null && cmake --build build -j2`), `pipeline-strong` (turns: `python /w/pipeline.py --rows 600 --rate 10 --signal strong`), `pipeline-weak` (`--signal weak`), `pipeline-none` (`--signal none`), and one interactive job `interactive-short` (class interactive, turns: `ls -la /w`, `python -c "print(1)"`, deadline 30 s). `repeat: 3` for pytest and pipeline jobs, `2` for the build. The pipeline script is copied into the container by the setup command `mkdir -p /w && cat > /w/pipeline.py <<'EOF' ... EOF` generated by the driver from `scripts/tools/pipeline.py` (`JobSpec.setup` entries may be `{"file": "scripts/tools/pipeline.py", "to": "/w/pipeline.py"}` dicts, which the driver expands into a heredoc command).

- [ ] **Step 1: Write the failing test**

```python
# tests/collect/test_driver.py
import time
from atfm.bus import InMemoryBus
from atfm.collect.jobs import CollectionSpec, JobSpec, TurnSpec
from atfm.collect.driver import run_collection, no_llm
from atfm.sidecar.minisweagent import SidecarConfig, SidecarMixin
from atfm.traces.sidecar import events_to_trace_table

class _LocalEnv(SidecarMixin):
    def __init__(self, cfg):
        self.sidecar = cfg
    def execute(self, action, cwd="", *, timeout=None):
        return self.sidecar_execute(action["command"], "/tmp", timeout, lambda c: ["bash", "-lc", c])
    def _check_finished(self, output):
        pass

def test_collection_records_sessions_and_tools(tmp_path):
    bus = InMemoryBus()
    spec = CollectionSpec(proxy_url=None, model="m", events_path=str(tmp_path / "ev.jsonl"), jobs=[
        JobSpec(name="fake-pytest", cls="background", tenant="t1", image=None, repeat=2,
                setup=["echo setup"],
                turns=[TurnSpec(cmd="printf 'collected 2 items\\na::t PASSED [ 50%%]\\nb::t PASSED [100%%]\\n'"),
                       TurnSpec(cmd="echo done")])])
    out = run_collection(spec, env_factory=lambda job, cfg: _LocalEnv(cfg), llm=no_llm(bus), bus=bus)
    assert out["sessions"] == 2 and out["tools"] == 6 and out["errors"] == 0
    t = events_to_trace_table(bus.drain())
    df = t.df
    assert df.session_id.nunique() == 2 and (df["class"] == "background").all()
    assert df[df.tool_name == "pytest"]["progress_events"].apply(len).min() >= 2
    assert set(df.tool_name.dropna()) >= {"pytest", "echo"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/collect -q`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# src/atfm/collect/__init__.py
```

```python
# src/atfm/collect/jobs.py
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

class TurnSpec(BaseModel):
    cmd: str
    prompt: str = "continue"
    max_tokens: int = 16

class JobSpec(BaseModel):
    name: str
    cls: Literal["interactive", "background"] = Field(alias="class", default="background")
    model_config = {"populate_by_name": True}
    tenant: str = "t0"
    image: str | None = None
    repeat: int = 1
    setup: list[str | dict] = Field(default_factory=list)
    turns: list[TurnSpec]
    timeout_s: float = 1800.0
    deadline_s: float | None = None

class CollectionSpec(BaseModel):
    proxy_url: str | None = None
    model: str = "Qwen/Qwen3-0.6B"
    events_path: str = "runs/collect/events.jsonl"
    jobs: list[JobSpec]
    seed: int = 0
    concurrency: int = 1
```

```python
# src/atfm/collect/driver.py
from __future__ import annotations
import json, tempfile, time, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from atfm.bus import JsonlBus
from atfm.schema.events import LlmDone, LlmRequest, SessionStart
from atfm.sidecar.minisweagent import SidecarConfig
from .jobs import CollectionSpec, JobSpec

def no_llm(bus):
    """Synthetic zero-duration LLM calls so tool phases still pair with calls in the trace table."""
    turns: dict[str, int] = {}
    started: set[str] = set()
    def call(session_id, cls, tenant, deadline, prompt, max_tokens):
        t = time.time()
        if session_id not in started:
            started.add(session_id)
            bus.publish(SessionStart(t=t, session_id=session_id, tenant=tenant, cls=cls, deadline=deadline))
        k = turns.get(session_id, 0); turns[session_id] = k + 1
        rid = uuid.uuid4().hex[:12]
        bus.publish(LlmRequest(t=t, session_id=session_id, turn_index=k, request_id=rid, isl=max(1, len(prompt) // 4)))
        bus.publish(LlmDone(t=t + 1e-3, session_id=session_id, request_id=rid, osl=0))
        return {"synthetic": True}
    return call

def proxy_chat(proxy_url: str, model: str):
    def call(session_id, cls, tenant, deadline, prompt, max_tokens):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        headers = {"content-type": "application/json", "x-atfm-session": session_id, "x-atfm-class": cls, "x-atfm-tenant": tenant}
        if deadline is not None:
            headers["x-atfm-deadline"] = str(deadline)
        req = urllib.request.Request(f"{proxy_url.rstrip('/')}/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode())
    return call

def _default_env_factory(job: JobSpec, cfg: SidecarConfig):
    from atfm.sidecar.minisweagent import SidecarDockerEnvironment, SidecarLocalEnvironment
    if job.image:
        return SidecarDockerEnvironment(sidecar=cfg, image=job.image, timeout=int(job.timeout_s), cwd="/w",
                                        run_args=["--rm"])
    return SidecarLocalEnvironment(sidecar=cfg, timeout=int(job.timeout_s), cwd=tempfile.mkdtemp(prefix="atfm-"))

def _expand_setup(item) -> str:
    if isinstance(item, str):
        return item
    src = Path(item["file"]).read_text()
    return f"mkdir -p {Path(item['to']).parent} && cat > {item['to']} <<'ATFM_EOF'\n{src}\nATFM_EOF"

def _run_session(job: JobSpec, k: int, spec: CollectionSpec, bus, env_factory, llm, clock) -> dict:
    sid = f"{job.name}-{k}-{uuid.uuid4().hex[:6]}"
    deadline = None if job.deadline_s is None else clock() + job.deadline_s
    cfg = SidecarConfig(session_id=sid, tenant=job.tenant, cls=job.cls, bus=bus, gate_url=spec.proxy_url, clock=clock)
    env = env_factory(job, cfg)
    tools = errors = 0
    try:
        for s in job.setup:
            out = env.execute({"command": _expand_setup(s)}); tools += 1
            if out["returncode"] != 0:
                errors += 1
        for turn in job.turns:
            llm(sid, job.cls, job.tenant, deadline, turn.prompt, turn.max_tokens)
            out = env.execute({"command": turn.cmd}); tools += 1
            if out["returncode"] not in (0, 1):   # test failures (1) are legitimate outcomes
                errors += 1
        llm(sid, job.cls, job.tenant, deadline, "final", 8)
    except Exception:
        errors += 1
    finally:
        cleanup = getattr(env, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception:
                pass
    return {"tools": tools, "errors": errors}

def run_collection(spec: CollectionSpec, env_factory=None, llm=None, bus=None, clock=time.time) -> dict:
    bus = bus if bus is not None else JsonlBus(spec.events_path)
    env_factory = env_factory or _default_env_factory
    llm = llm or (proxy_chat(spec.proxy_url, spec.model) if spec.proxy_url else no_llm(bus))
    work = [(job, k) for job in spec.jobs for k in range(job.repeat)]
    totals = {"sessions": 0, "tools": 0, "errors": 0}
    with ThreadPoolExecutor(max_workers=max(1, spec.concurrency)) as pool:
        for r in pool.map(lambda jk: _run_session(jk[0], jk[1], spec, bus, env_factory, llm, clock), work):
            totals["sessions"] += 1; totals["tools"] += r["tools"]; totals["errors"] += r["errors"]
    return totals
```

```python
# scripts/tools/pipeline.py
import argparse, random, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=600)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--signal", choices=["strong", "weak", "none"], default="strong")
    a = ap.parse_args()
    done = 0; last = time.time(); stage = 0
    while done < a.rows:
        time.sleep(1.0)
        done = min(a.rows, done + int(a.rate * random.uniform(0.8, 1.2)))
        if a.signal == "strong":
            print(f"processed {done}/{a.rows} rows", flush=True)
        elif a.signal == "weak":
            q = int(4 * done / a.rows)
            if q > stage:
                stage = q; print(f"stage {stage}/4", flush=True)
    print("finished", flush=True)

if __name__ == "__main__":
    main()
```

```python
# scripts/collect_traces.py
import sys, yaml
from atfm.collect.driver import run_collection
from atfm.collect.jobs import CollectionSpec

if __name__ == "__main__":
    spec = CollectionSpec(**yaml.safe_load(open(sys.argv[1])))
    print(run_collection(spec))
```

```yaml
# experiments/l1_jobs.yaml
proxy_url: null
model: Qwen/Qwen3-0.6B
events_path: runs/collect/l1_events.jsonl
concurrency: 2
jobs:
  - name: pytest-attrs
    class: background
    tenant: t1
    image: python:3.12-slim
    repeat: 3
    setup:
      - "apt-get update -qq >/dev/null && apt-get install -y -qq git >/dev/null"
      - "git clone -q --depth 1 https://github.com/python-attrs/attrs /w/attrs"
      - "pip install -q -e '/w/attrs[tests]' >/dev/null 2>&1 || pip install -q -e /w/attrs pytest hypothesis >/dev/null 2>&1"
    turns:
      - {cmd: "cd /w/attrs && python -m pytest -q -p no:cacheprovider tests"}
  - name: pytest-more-itertools
    class: background
    tenant: t2
    image: python:3.12-slim
    repeat: 3
    setup:
      - "apt-get update -qq >/dev/null && apt-get install -y -qq git >/dev/null"
      - "git clone -q --depth 1 https://github.com/more-itertools/more-itertools /w/more-itertools"
      - "pip install -q pytest >/dev/null 2>&1"
    turns:
      - {cmd: "cd /w/more-itertools && python -m pytest -v -p no:cacheprovider tests"}
  - name: build-cjson
    class: background
    tenant: t1
    image: gcc:13
    repeat: 2
    setup:
      - "apt-get update -qq >/dev/null && apt-get install -y -qq cmake git >/dev/null"
      - "git clone -q --depth 1 https://github.com/DaveGamble/cJSON /w/cJSON"
    turns:
      - {cmd: "cd /w/cJSON && cmake -S . -B build >/dev/null && cmake --build build -j2"}
  - name: pipeline-strong
    class: background
    tenant: t3
    image: python:3.12-slim
    repeat: 3
    setup:
      - {file: scripts/tools/pipeline.py, to: /w/pipeline.py}
    turns:
      - {cmd: "python /w/pipeline.py --rows 600 --rate 10 --signal strong"}
  - name: pipeline-weak
    class: background
    tenant: t3
    image: python:3.12-slim
    repeat: 3
    setup:
      - {file: scripts/tools/pipeline.py, to: /w/pipeline.py}
    turns:
      - {cmd: "python /w/pipeline.py --rows 400 --rate 10 --signal weak"}
  - name: pipeline-none
    class: background
    tenant: t3
    image: python:3.12-slim
    repeat: 3
    setup:
      - {file: scripts/tools/pipeline.py, to: /w/pipeline.py}
    turns:
      - {cmd: "python /w/pipeline.py --rows 300 --rate 10 --signal none"}
  - name: interactive-short
    class: interactive
    tenant: t4
    image: python:3.12-slim
    repeat: 3
    deadline_s: 30
    turns:
      - {cmd: "ls -la /"}
      - {cmd: "python -c 'print(sum(range(10**6)))'"}
      - {cmd: "python -c 'import time; time.sleep(3); print(1)'"}
```

- [ ] **Step 4: Run the unit test, then a short real Docker collection**

Run: `uv run pytest tests/collect -q`
Expected: PASS

Then, with Docker available: `ATFM_DOCKER=1 uv run python scripts/collect_traces.py experiments/l1_jobs.yaml`
Expected: prints `{'sessions': 20, 'tools': ..., 'errors': 0}` after 15 to 40 minutes (first pulls of `python:3.12-slim` and `gcc:13` plus apt installs). If a job errors, read the tool output by re-running its command in the image by hand, fix the YAML, and rerun only that job (edit `repeat` of the others to 0 temporarily). Event log lands in `runs/collect/l1_events.jsonl`.

- [ ] **Step 5: Commit**

```bash
git add src/atfm/collect scripts/collect_traces.py scripts/tools/pipeline.py experiments/l1_jobs.yaml tests/collect
git commit -m "feat: scripted trace-collection driver with real long tools through the sidecar"
```

---

### Task 8: Coverage report, H1b experiment and results note

**Files:**
- Create: `src/atfm/eval/coverage.py`, `experiments/h1b_sidecar.yaml`, `docs/research/2026-09-23-l1-results.md`
- Modify: `src/atfm/experiments/h1.py` (source `"sidecar"`), `tests/experiments/test_h1.py`
- Test: `tests/eval/test_coverage.py`

**Interfaces:**
- Produces:
  - `signal_class(row) -> Literal["strong", "weak", "none"]`: strong if at least two progress events with a known `total`; weak if exactly one progress event with `total`, or any `data_events`, or a progress event without `total`; none otherwise. Think rows (`__think__`) and rows without a tool are excluded.
  - `coverage_report(table: TraceTable) -> pd.DataFrame` with columns `tool_name, calls, tool_time_s, strong_time_share, weak_time_share, none_time_share` plus a final `ALL` row; `coverage_markdown(df) -> str`.
  - `H1Config.source` gains `"sidecar"` with `sidecar_events: str | None`; `_tables` for sidecar: `events_to_trace_table(read_events(path))`, split by time blocks of `block_seconds` (config, default 600 s for sidecar runs) with `test_fraction`; no overlay (the collection is already a fleet); `overlay_duration_s` is ignored and ticks cover the test table's own span (`_ticks(t_min, t_max, t_max - t_min, ...)`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_coverage.py
from atfm.schema.trace import TraceRow, TraceTable
from atfm.eval.coverage import signal_class, coverage_report

def _row(tool, prog, data=(), dur=100.0, i=0):
    return TraceRow(session_id=f"s{i}", cls="background", tenant="t", turn_index=0, t_request=0.0, t_first_token=1.0, t_last_token=2.0,
                    isl=10, osl=1, tool_name=tool, t_tool_start=2.0, t_tool_end=2.0 + dur,
                    progress_events=[{"t": 3.0 + j, "completed": j, "total": 10 if tot else None, "phase": "run"} for j, tot in prog],
                    data_events=[{"t": 3.0, "metric": m, "value": 1.0} for m in data], source="test")

def test_signal_class():
    assert signal_class(_row("pytest", [(1, True), (2, True)]).model_dump(by_alias=True)) == "strong"
    assert signal_class(_row("build", [(9, True)]).model_dump(by_alias=True)) == "weak"
    assert signal_class(_row("bash", [], data=("lines_per_s",)).model_dump(by_alias=True)) == "weak"
    assert signal_class(_row("bash", []).model_dump(by_alias=True)) == "none"

def test_coverage_report_time_shares():
    t = TraceTable.from_rows([_row("pytest", [(1, True), (2, True)], dur=300.0, i=0), _row("bash", [], dur=100.0, i=1),
                              _row("bash", [(1, True)], dur=100.0, i=2)])
    df = coverage_report(t).set_index("tool_name")
    assert abs(df.loc["ALL", "strong_time_share"] - 0.6) < 1e-9 and abs(df.loc["bash", "weak_time_share"] - 0.5) < 1e-9
    assert df.loc["ALL", "calls"] == 3
```

Append to `tests/experiments/test_h1.py`:

```python
def test_h1_sidecar_source_runs(tmp_path):
    import json
    from atfm.bus import JsonlBus
    from atfm.schema.events import parse_event
    bus = JsonlBus(tmp_path / "ev.jsonl")
    t = 0.0
    for k in range(12):                                   # 12 tool-only sessions, 100 s pytest each with progress
        sid = f"s{k}"
        bus.publish(parse_event({"kind": "session.start", "t": t, "session_id": sid, "tenant": "t", "class": "background"}))
        bus.publish(parse_event({"kind": "tool.start", "t": t + 1, "session_id": sid, "turn_index": 0, "call_id": f"c{k}", "tool_name": "pytest", "backend_id": "ci"}))
        for j in range(1, 10):
            bus.publish(parse_event({"kind": "tool.progress", "t": t + 1 + 10 * j, "session_id": sid, "call_id": f"c{k}", "completed": 10 * j, "total": 100, "phase": "run"}))
        bus.publish(parse_event({"kind": "tool.end", "t": t + 101, "session_id": sid, "call_id": f"c{k}", "exit_status": 0}))
        t += 150.0
    cfg = H1Config(name="sc", source="sidecar", sidecar_events=str(tmp_path / "ev.jsonl"), block_seconds=600.0, test_fraction=0.5,
                   tick_s=30.0, horizons=[30.0, 120.0], n_samples=32, models=["M1", "M2"], out_dir=str(tmp_path))
    df = run_h1(cfg)
    assert set(df["model"]) == {"M1_survival", "M2_progress"} and (df["n"] > 3).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_coverage.py tests/experiments -q`
Expected: FAIL (ModuleNotFoundError for coverage; validation error for `source="sidecar"`)

- [ ] **Step 3: Implement**

```python
# src/atfm/eval/coverage.py
from __future__ import annotations
import pandas as pd
from atfm.schema.trace import TraceTable

def signal_class(row: dict) -> str:
    prog = row.get("progress_events") or []
    data = row.get("data_events") or []
    with_total = [p for p in prog if p.get("total") not in (None, 0)]
    if len(with_total) >= 2:
        return "strong"
    if with_total or data or prog:
        return "weak"
    return "none"

def coverage_report(table: TraceTable) -> pd.DataFrame:
    df = table.df
    tools = df[df["tool_name"].notna() & (df["tool_name"] != "__think__")].copy()
    tools["dur"] = (tools["t_tool_end"] - tools["t_tool_start"]).clip(lower=0.0)
    tools["sig"] = tools.apply(lambda r: signal_class(r.to_dict()), axis=1)
    rows = []
    groups = list(tools.groupby("tool_name")) + [("ALL", tools)]
    for name, g in groups:
        tot = float(g["dur"].sum())
        rows.append({"tool_name": name, "calls": int(len(g)), "tool_time_s": tot,
                     **{f"{s}_time_share": (float(g.loc[g["sig"] == s, "dur"].sum()) / tot if tot > 0 else 0.0)
                        for s in ("strong", "weak", "none")}})
    return pd.DataFrame(rows)

def coverage_markdown(df: pd.DataFrame) -> str:
    lines = ["| tool | calls | tool time (s) | strong | weak | none |", "|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.tool_name} | {r.calls} | {r.tool_time_s:.0f} | {r.strong_time_share:.0%} | {r.weak_time_share:.0%} | {r.none_time_share:.0%} |")
    return "\n".join(lines)
```

Modify `src/atfm/experiments/h1.py`:

```python
# in H1Config
    source: Literal["tracelab", "synthetic", "sidecar"]
    sidecar_events: str | None = None

# in _tables, before the tracelab branch
    if cfg.source == "sidecar":
        from atfm.bus import read_events
        from atfm.traces.sidecar import events_to_trace_table
        assert cfg.sidecar_events is not None
        table = events_to_trace_table(read_events(cfg.sidecar_events))
        return split_by_time_blocks(table, cfg.block_seconds, cfg.test_fraction, cfg.seed)

# in run_h1, the window line becomes
    if cfg.source == "synthetic" and cfg.synthetic:
        window = cfg.synthetic.duration_s
    elif cfg.source == "sidecar":
        window = t_max - t_min
    else:
        window = cfg.overlay_duration_s
```

```yaml
# experiments/h1b_sidecar.yaml
name: h1b_sidecar_l1
source: sidecar
sidecar_events: runs/collect/l1_events.jsonl
block_seconds: 600
test_fraction: 0.5
tick_s: 10
horizons: [10, 30, 120, 300]
n_samples: 256
models: [B0, B1, B2, M1, M2]
seed: 0
```

- [ ] **Step 4: Run tests, then the H1b experiment and coverage on the collected events**

Run: `uv run pytest -q`
Expected: PASS (whole suite)

Then: `uv run python scripts/run_h1.py experiments/h1b_sidecar.yaml` and
`uv run python -c "from atfm.bus import read_events; from atfm.traces.sidecar import events_to_trace_table; from atfm.eval.coverage import coverage_report, coverage_markdown; print(coverage_markdown(coverage_report(events_to_trace_table(read_events('runs/collect/l1_events.jsonl')))))"`
Expected: a pinball table with M2 below M1 for background at 30 to 300 s on strong-signal tools, and a coverage table. Write both into `docs/research/2026-09-23-l1-results.md` with the same "how to read" preamble as the H1 note (metric, horizons, split, workload), the per-tool coverage, the M2 minus M1 ratio per horizon, and the caveats (scripted decisions, no real model, laptop CPU contention between concurrent containers).

- [ ] **Step 5: Commit**

```bash
git add src/atfm/eval/coverage.py src/atfm/experiments/h1.py experiments/h1b_sidecar.yaml tests/eval/test_coverage.py tests/experiments/test_h1.py docs/research/2026-09-23-l1-results.md
git commit -m "feat: signal coverage report and H1b sidecar experiment with first results"
```

---

### Task 9: End-to-end run through the proxy and Mocker (gated)

**Files:**
- Create: `scripts/l1_e2e.sh`, `tests/e2e/test_proxy_mocker.py`

**Interfaces:**
- Consumes: `LocalDynamo`, `create_app`, `run_collection`, `proxy_chat`, `events_to_trace_table`.
- Produces: `tests/e2e/test_proxy_mocker.py` (skipped unless `ATFM_DYNAMO=1`): starts `LocalDynamo(port=8791)`, starts the proxy in a thread with `uvicorn.Server` on port 8792 with `window=2` and a JSONL bus, runs a `CollectionSpec` with `proxy_url="http://127.0.0.1:8792"` and two local (non-Docker) jobs of three short turns each with `concurrency=3`, then asserts: every `llm.request` event carries `hints` with `strict_priority` in {0, 1, 2}; the proxy `/state` never reported `in_flight > 2` (poll during the run); the trace table pairs every tool with a call; the frontend log contains the session ids (grep `x-dynamo-session-id` is not logged by default, so instead assert the response ids differ per request).
- `scripts/l1_e2e.sh`: `up` Dynamo, run proxy, run `experiments/l1_jobs.yaml` with `proxy_url` overridden to the proxy, run board once over the events, `down`.

- [ ] **Step 1: Write the failing test**

```python
# tests/e2e/test_proxy_mocker.py
import os, threading, time
import pytest
pytestmark = pytest.mark.skipif(os.environ.get("ATFM_DYNAMO") != "1", reason="needs the dynamo extra and ATFM_DYNAMO=1")

def test_collection_through_proxy_and_mocker(tmp_path):
    import uvicorn, urllib.request, json
    from atfm.dynamo.local import LocalDynamo
    from atfm.proxy.app import create_app
    from atfm.proxy.config import ProxyConfig
    from atfm.bus import JsonlBus, read_events
    from atfm.collect.driver import run_collection
    from atfm.collect.jobs import CollectionSpec, JobSpec, TurnSpec
    from atfm.traces.sidecar import events_to_trace_table
    from atfm.sidecar.minisweagent import SidecarLocalEnvironment
    events = tmp_path / "ev.jsonl"
    with LocalDynamo(port=8791, log_dir=str(tmp_path)) as d:
        app = create_app(ProxyConfig(upstream_url=d.base_url, window=2, events_path=str(events)))
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8792, log_level="warning"))
        th = threading.Thread(target=server.run, daemon=True); th.start()
        time.sleep(1.0)
        peak = [0]; stop = [False]
        def poll():
            while not stop[0]:
                try:
                    st = json.loads(urllib.request.urlopen("http://127.0.0.1:8792/state", timeout=1).read())
                    peak[0] = max(peak[0], st["in_flight"])
                except Exception:
                    pass
                time.sleep(0.02)
        pt = threading.Thread(target=poll, daemon=True); pt.start()
        spec = CollectionSpec(proxy_url="http://127.0.0.1:8792", model=d.model, events_path=str(events), concurrency=3, jobs=[
            JobSpec(name="a", cls="background", tenant="t", image=None, repeat=2, turns=[TurnSpec(cmd="sleep 0.2; echo a")] * 3),
            JobSpec(name="b", cls="interactive", tenant="t", image=None, repeat=1, deadline_s=5,
                    turns=[TurnSpec(cmd="printf 'collected 1 items\\nx::t PASSED [100%%]\\n'")] * 3)])
        out = run_collection(spec, env_factory=lambda job, cfg: SidecarLocalEnvironment(sidecar=cfg, timeout=60, cwd="/tmp"))
        stop[0] = True; server.should_exit = True
    assert out["errors"] == 0 and out["sessions"] == 3
    ev = read_events(events)
    reqs = [e for e in ev if e.kind == "llm.request"]
    assert reqs and all(e.hints.get("strict_priority") in (0, 1, 2) for e in reqs)
    assert any(e.hints.get("strict_priority") == 2 for e in reqs)          # interactive with a 5 s deadline
    assert peak[0] <= 2
    df = events_to_trace_table(ev).df
    assert (df[df.tool_name.notna()]["t_tool_end"] > df[df.tool_name.notna()]["t_tool_start"]).all()
```

- [ ] **Step 2: Run to verify it fails when enabled**

Run: `ATFM_DYNAMO=1 uv run pytest tests/e2e -q`
Expected: FAIL only if any earlier task left a gap (this test is integration; the code it needs exists after Tasks 3 to 7). If it passes immediately, that is acceptable for an integration test: record in the ledger that it was green on first run and why (all units were already tested).

- [ ] **Step 3: Write the script**

```bash
# scripts/l1_e2e.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/dynamo_local.py up
trap 'uv run python scripts/dynamo_local.py down' EXIT
uv run python scripts/run_proxy.py --upstream http://127.0.0.1:8000 --port 8799 --window 4 &
PROXY=$!
trap 'kill $PROXY; uv run python scripts/dynamo_local.py down' EXIT
sleep 2
sed 's#^proxy_url: null#proxy_url: http://127.0.0.1:8799#' experiments/l1_jobs.yaml > runs/l1_jobs_proxy.yaml
uv run python scripts/collect_traces.py runs/l1_jobs_proxy.yaml
uv run python scripts/run_board.py --events runs/collect/l1_events.jsonl --snapshots runs/collect/snapshots.jsonl --train data/tracelab/tracelab.parquet --once
```

- [ ] **Step 4: Run the gated test**

Run: `ATFM_DYNAMO=1 uv run pytest tests/e2e -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/l1_e2e.sh
git add scripts/l1_e2e.sh tests/e2e
git commit -m "feat: end-to-end proxy + Mocker collection test and L1 script"
```

---

## Self-review

- Spec coverage: 3.2 events (T1), 7 sidecar with parser chain, adapters, launch gate (T2, T3), 4.2 global window and 4.3 tiers, hints, objective inputs (T4), 5.1 registry and 5.4 per-request predictions (T5), 10 fail-open behaviours (T2 to T4), 13 item 2 collection, coverage and H1b (T7, T8), Dynamo integration surface limited to hints and headers (T6, T9). Not in scope by design: GDP solver (6.2, the proxy only enforces directives), tiering (6.3), planner (6.4), simulator policies (8).
- Placeholders: Task 2 shows the read loop twice (blocking readline then thread-based) and instructs to combine them; that is explicit, not a placeholder. Task 6 step 4 names a possible discovery-root variable without knowing it; the instruction is to read `--help`, which is concrete. Everything else has code.
- Type consistency: `SidecarConfig.turn_index` is a dataclass field used by `SidecarMixin`; `Entry.not_before` set in `HoldQueue.submit` and read by `tick`/`stats`; `create_app` returns a FastAPI app whose `state.queue` is a `HoldQueue`; `events_to_trace_table` consumes the `Event` union from T1; `LiveBoard.expected_service/expected_tool_next` match the `predictor` duck type used by `create_app`'s `predict`.
- Review Focus: 1 and 2 in `tests/sidecar/test_core.py` (byte-exact with invalid UTF-8; timeout); 3 in `test_window_orders_interactive_slack_first`; 4 in `test_directive_hold_capped_and_upstream_error_passthrough`; 5 in `test_registry_transitions_and_unknown_sessions`.
