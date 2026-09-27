# ATFM v1 completion Implementation Plan

> **Historical implementation plan.** Embedded code and task checklists are a design record,
> not the current source of setup instructions. See [implementation status](../../status.md),
> [operations](../../operations.md), and [core development](../../development/core.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build every component the v1.1 spec names that does not yet exist, test-first, without running experiments: the control package (GDP planner, placement touch controller, replica floor), simulator additions (touch arm, trace replay, golden test, run provenance), a Redis Streams bus, board metrics scraping and an HTTP service, proxy hardening and touch issuing, and the remaining sidecar parsers and harness adapters.

**Architecture:** New package `atfm.control` holds controller classes that are pure functions of a `ForecastSnapshot`, per-session resumption quantiles and capacity, emitting pydantic directives with expiry; the simulator and the proxy both consume them. Everything else extends an existing module behind its current interface (bus `Bus` protocol, `Policy` interface, parser chain, `HoldQueue`).

**Tech Stack:** Python 3.12, uv, pydantic v2, numpy/pandas, FastAPI + httpx TestClient, pytest (asyncio auto), redis-py (optional import, fake client in tests).

**Spec:** `docs/superpowers/specs/2026-09-22-atfm-architecture-design.md` (sections 3.5, 6, 7, 8, 10, 11); `docs/research/2026-09-27-l2-h100-study-design.md` for the touch mechanism.

## Global Constraints

- Python 3.12; no new hard dependencies (`redis` optional, imported lazily).
- Every controller is fail-open and every directive carries `expires_at` (spec 10).
- No experiment runs: unit tests may run the simulator for at most a few simulated minutes.
- Commit after each task with the Co-Authored-By line; never `git add -A` (a parallel session edits `docs/paper`).

## Review Focus

1. GDP planner given samples where the constraint can never be met must return the cap, not loop (property test).
2. Touch controller under a zero budget must emit nothing and never raise.
3. Redis bus `drain()` must be idempotent under consumer restarts (last-id bookkeeping).
4. Proxy hold queue over its max size must forward FCFS and count an alarm, not drop requests.
5. Trace replay regime with a table lacking tool phases must still produce programs (turns without tools).

---

### Task 1: Directives and the GDP planner (`atfm.control`)

**Files:** Create `src/atfm/control/__init__.py`, `src/atfm/control/directives.py`, `src/atfm/control/gdp.py`; Test `tests/control/test_gdp.py`.

**Interfaces:**
- Produces `HoldDirective(session_id, release_not_before, reason, expires_at, tenant)`, `TierDirective(session_id, action in {pin, demote, promote, prefetch}, tier, eta_q10, eta_q90, expires_at)`, `ReplicaDirective(replicas_at_least, horizon_s, expires_at)`, `TouchDirective(session_id, worker_id, expires_at)`.
- `GdpPlanner(slot_s=30, horizon_s=900, eps=0.1, max_hold_s=600).plan(now, snap, capacity: dict[str, float], deferrable: list[Deferrable], tenant_max_delay: dict|None) -> list[HoldDirective]` where `Deferrable(session_id, tenant, eta_s, kv_blocks, prefill_tokens, cost_per_s, deadline)`.

- [ ] Step 1: write tests: (a) two deferrable sessions and interactive samples that saturate slot 0 only: the first session is released at slot 1, the second at slot 0 if it fits; (b) a session whose earliest feasible slot is beyond the cap gets `release_not_before = now + max_hold_s` and reason `capped`; (c) property: over 50 random snapshots, no directive exceeds the cap and, for released slots, the chance constraint evaluated on the planner's own samples holds; (d) tenant fairness: `tenant_max_delay` bounds a tenant's imposed delay.
- [ ] Step 2: run, expect ImportError.
- [ ] Step 3: implement: slots = horizon/slot; interactive demand per slot per resource from the snapshot's horizon closest to each slot end (cumulative demand differences); greedy ration-by-schedule in `eta_s` order; each session takes the earliest slot >= its own where `P(I + assigned <= C) >= 1 - eps` on the samples for both resources; commit its samples to the slot; cap at max_hold_s; directives expire at `now + slot_s`.
- [ ] Step 4: run tests, all pass; run whole suite.
- [ ] Step 5: commit `feat(control): directives and GDP planner`.

### Task 2: Placement touch controller and tier logging

**Files:** Create `src/atfm/control/touch.py`; Test `tests/control/test_touch.py`.

**Interfaces:**
- `TouchController(horizon_s=30, age_s=20, budget_per_s=1.0).plan(now, resumptions: dict[str, tuple[q10, q50, q90]], residency: dict[str, Residency(worker_id, blocks, last_used)], evict_frontier_age: dict[worker_id, float]) -> list[TouchDirective]`: touch sessions with q50 <= horizon whose blocks are older than the frontier minus age margin, most imminent first, within budget (integer tokens accumulate per tick).
- `TierLogger.plan(now, resumptions, tier_lead_s: dict[str,float]) -> list[TierDirective]` (logged only, spec 6.3).

- [ ] Steps: tests (budget 0 emits nothing; ordering by q50; only at-risk blocks; tier logger picks the deepest tier whose lead time < q10), RED, implement, GREEN, suite, commit.

### Task 3: Replica floor proposer

**Files:** Create `src/atfm/control/replica.py`; Test `tests/control/test_replica.py`.

**Interfaces:** `ReplicaFloor(lead_time_s, blocks_per_replica, prefill_tps_per_replica, min_replicas=1).propose(now, snap) -> ReplicaDirective`; `VirtualConnector` records proposals and exposes `.history`; `propose_at_least(snapshot)` mirrors Dynamo Planner `OverrideType.AT_LEAST` semantics as a plain method.

- [ ] Steps: tests (q90 demand at the lead-time horizon divided by per-replica capacity, ceil, floor min; picks the closest horizon), RED, implement, GREEN, commit.

### Task 4: Simulator: touch arm, trace replay regime, golden test, run provenance

**Files:** Modify `src/atfm/sim/engine.py` (`touch(session_id, now)`), `src/atfm/sim/kv_placement.py` (`ForecastTouchPolicy`), `src/atfm/experiments/h2sim.py` (`regime: trace`, `trace_path`, arm `forecast_touch_M1/M2`, `oracle_touch`, provenance `manifest.json`); Test `tests/sim/test_touch_arm.py`, `tests/sim/test_golden.py`, `tests/experiments/test_provenance.py`.

- [ ] Steps: tests (touch moves a session to MRU and charges `touch_prefill_tokens` in the log; touch arm emits at most budget touches per tick; trace regime builds programs from a parquet table; golden metrics for a fixed tiny spec and seed; manifest has git sha and inputs hash), RED, implement, GREEN, suite, commit.

### Task 5: Redis Streams bus

**Files:** Create `src/atfm/bus/redis.py`; Test `tests/bus/test_redis_bus.py` (with a minimal fake client implementing `xadd`, `xread`, `xrange`).

- [ ] Steps: tests (publish/drain round trip preserving event models; drain returns only new entries; `since` cursor persisted so a restarted consumer resumes; missing `redis` package raises a clear error only on construction with a real URL), RED, implement, GREEN, commit.

### Task 6: Board metrics scraper and HTTP service

**Files:** Create `src/atfm/board/metrics.py` (Prometheus text -> `WorkerMetrics`), `src/atfm/board/service.py` (FastAPI: `GET /snapshot`, `POST /predict`, `GET /directives`); Modify `scripts/run_board.py` (`--serve`); Test `tests/board/test_metrics.py`, `tests/board/test_service.py`.

- [ ] Steps: tests (parse a sample Dynamo/vLLM metrics page into used/total blocks and queue depth per worker; service returns q50/q90 per target/class and per-request predictions with the 50 ms budget honoured), RED, implement, GREEN, commit.

### Task 7: Proxy hardening and touch issuing

**Files:** Modify `src/atfm/proxy/queue.py` (`max_size`, `alarms`, directive expiry), `src/atfm/proxy/app.py` (`POST /touch` issuing a prefix-only upstream request with `max_tokens: 1`, counting `touches`, `touch_tokens`), `src/atfm/proxy/config.py`; Test `tests/proxy/test_hardening.py`.

- [ ] Steps: tests (queue beyond `max_size` forwards FCFS and increments `alarms`; expired directive ignored; `/touch` forwards a minimal request and records it in stats), RED, implement, GREEN, commit.

### Task 8: Sidecar parsers and adapters

**Files:** Modify `src/atfm/sidecar/parsers.py` (`RowsParser`: `rows processed k/N`, `processed k of N`; `StageParser`: `stage k/N`; `TrainingParser`: `step k/N` progress and `loss=x` data); Create `src/atfm/sidecar/openhands.py`, `src/atfm/sidecar/harbor.py` (wrap an executor callable with `run_tool`); Test `tests/sidecar/test_more_parsers.py`, `tests/sidecar/test_other_adapters.py`.

- [ ] Steps: tests (each parser on sample lines; adapters call through, emit tool events, and pass output byte-exact using a fake executor), RED, implement, GREEN, suite, commit.
