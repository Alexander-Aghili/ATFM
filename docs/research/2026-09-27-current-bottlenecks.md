# Current bottlenecks: CPU kernels and control-path amplification

The largest measured control kernels are GDP slot search and per-session
forecasting. The deployed path also contains repeated log reads, queue scans,
and serial RPCs that can dominate a full control cycle independently of GDP.
This audit adds component timings and profiles; it makes no runtime changes.

Subsequent fixes are tracked in [control scaling decisions](../development/control-scaling.md).
The observations and measurements below describe the pre-change source hashes.

## Measured costs

| Component | Workload | Result | What it establishes |
| --- | --- | --- | --- |
| GDP slot search | 100,000 sessions, 10,000 slots, 1,024 draws, saturated | `_first_slot` accounts for about 82% of an 8.02 s instrumented profile. Earlier uninstrumented median: 7.32 s. | Repeated feasible-slot search dominates this stress workload. |
| GDP slot search | One million sessions, 300 slots, 128 draws, saturated | About 82% of a 24.79 s instrumented profile. Earlier uninstrumented median: 15.91 s. | Per-session search remains dominant even with fewer slots; profiles add material overhead. |
| Forecast | 8,192 / 100,000 active sessions, three horizons, 128 draws | 0.330 / 4.120 s median | Even approximately linear forecasting is substantial at high session counts. |
| Unchanged JSONL drain | 10,000 / 100,000 / one million historical events | 0.032 / 0.344 / 4.297 s median per drain | The entire old log is read and parsed again despite no new arrivals. |
| Single-slot queue drainage | 1,000 / 4,000 queued requests | 0.103 / 1.974 s median to release the backlog | The earlier batch optimization does not eliminate repeated scanning when completions arrive individually. |

The GDP cases are deliberately extreme. The supplied
[`control_local.yaml`](../../experiments/control_local.yaml) configures a
300-second horizon with 30-second slots: **10 slots**, with a 60-second hold
cap. Do not use a 10,000-slot stress result as a timing estimate for that
configuration. Similarly, the forecast fixture is homogeneous tool-running
sessions with short progress histories, no exogenous arrivals, and no fan-out.
These separate measurements must not be added into a purported end-to-end time.

## Where the work occurs

### 1. GDP repeatedly searches candidate slots

[`GdpPlanner._first_slot`](../../src/atfm/control/gdp.py) builds a new slot-index
array, feasibility mask, resource comparisons, and list of feasible positions
for each session. Chance thresholds are already cached for the planning call,
so rechecking all Monte Carlo draws is no longer the issue. The remaining
search work is O(S F), plus preparation and sorting.

The fast path currently creates indices through the entire remaining horizon
and then masks slots beyond `max_hold_s`. It still evaluates resource arrays
for masked-out slots. Bound the candidate slice before constructing arrays as
a first targeted improvement. That alone will not fix stress cases whose hold
cap intentionally covers the whole horizon.

The next algorithmic candidate is an index that skips intervals known to be
infeasible, with capacity bounds and updates after each commitment. Multiple
resources and arbitrary per-session demands mean a simple heap is not
necessarily sufficient; do not promise logarithmic worst-case search without
proving it. One directive object per session also imposes O(S) output cost.

### 2. Forecasting makes many small per-session calls

[`SessionForecaster.forecast`](../../src/atfm/board/forecaster.py) calls resumption
prediction, prompt-length sampling, aggregation, and spawn prediction for each
state. At 100,000 sessions, the profile spends about 62% of its 6.64 instrumented
seconds inside `ProgressPredictor.resumption`, including progress interpolation
and residual draws. NumPy invocation overhead and small reductions are visible
throughout. Those nested profile times overlap and should not be summed.

The board's [`/directives`](../../src/atfm/board/service.py) path separately
samples per-session resumption quantiles after the forecast tick, so the full
cycle does more than the forecast-only benchmark. Candidate improvements are
batched prediction for compatible states and reuse of per-tick summaries.
Preserve sample dependence, calibration, event freshness, and explicit RNG
semantics when changing sampling. Longer histories, different phases, and
fan-out still need representative coverage.

### 3. JSONL consumption grows with the entire history

[`JsonlBus.drain`](../../src/atfm/bus/jsonl.py) delegates to `read_events` with no
consumer cursor. [`create_board_app.ingest`](../../src/atfm/board/service.py)
applies every returned event each tick. This is both wasted work and a
correctness risk: historical starts can be counted repeatedly and progress can
be replayed into live state. The timing measures read/parse only, not registry
application, and uses a warmed local file rather than cold storage.

Incremental per-consumer consumption is the first fix to make. Keep full-file
`read_events` for offline replay. Define restart, truncation/rotation, partial
last-line, malformed-event, and retry behavior explicitly. Registry
`new_starts_since` also scans an ever-growing list independently of the bus.

### 4. Queue work is amplified by individual completions and directives

[`HoldQueue.tick`](../../src/atfm/proxy/queue.py) still scans pending entries for
promotion, eligibility, removal, and the next wake-up time. When one request
finishes at a time, these passes repeat for each release. The measured growth
is consistent with the remaining quadratic total work of draining a backlog;
it is not evidence of a particular network request rate.

More significantly, [`ControlLoop.step`](../../src/atfm/control/loop.py) sends
one synchronous HTTP request for each hold directive, and the proxy's
[`/directives`](../../src/atfm/proxy/app.py) endpoint calls `queue.tick()` for
each request. D directives against a queue of size Q can therefore induce
O(D Q) scan work as well as D serial round trips. The code emits a directive
for every deferrable session, not only changed decisions. This amplification
has not yet been measured over a real network.

Batch directive application and tick once per batch; then consider versioned
changes and indexed ready/held/promotion queues. Preserve stable ties,
cancellation, expiry, fairness, and overflow semantics. A faster GDP kernel
alone cannot remove the delivery and admission costs.

### 5. CPU control work shares the request event loop

Board `/tick` performs ingestion and forecasting synchronously, and the first
`/directives` request for each snapshot performs resumption sampling and
planning synchronously. Both are async handlers. While that work runs, the same
event loop cannot serve health checks or dispatch/complete prediction requests.
This is a scheduling hazard visible in code, not a measured p99 latency result.

Per-request predictions use a four-thread proxy pool and
[`BoardClient`](../../src/atfm/proxy/board_client.py) makes two sequential
`/predict` requests, although each response already includes both requested
values. The client socket timeout is at least 0.5 s while the proxy's default
await budget is 0.05 s. Cancelling an already-running future does not stop its
synchronous HTTP operation, so slow board responses can occupy the small pool
past the caller's deadline.

Combine prediction requests and give control-state mutation a dedicated owner
that publishes immutable snapshots. Simply moving mutable registry and RNG
operations onto arbitrary threads would introduce races; increasing the number
of independent web workers would introduce divergent state unless ownership is
redesigned. The control loop also sleeps its interval after finishing work, so
its actual cadence is work duration plus sleep, not a fixed five-second clock.

### 6. Long-running state and logging need limits

The proxy bounds remembered prompt bodies, but its `known`, `turns`, and
`last_index` collections retain historical session keys. The registry's start
history likewise has no pruning. Request completion flushes trace output
synchronously, and JSONL publication opens/appends for individual events. These
are inspected memory/I/O growth risks, not measured dominant costs in this
audit. Add expiry/retention and measure event-loop lag before choosing a logging
architecture.

## Recommended implementation order

1. Correct incremental event consumption and bound historical bookkeeping.
2. Batch directive delivery/application and combine prediction RPCs.
3. Separate the control-state owner from the request event loop; measure snapshot
   age, prediction fallback rate, and request p95/p99 under concurrent control work.
4. Bound GDP search to eligible slots, then evaluate an indexed search and
   feasible capacity-based fast rejection with parity tests.
5. Batch forecast operations and avoid redundant per-tick resumption sampling.
6. Replace repeated queue scans with indexed scheduling once ordering and timer
   semantics are covered by replay/property tests.

This ordering prioritizes correctness and whole-cycle amplification rather than
only the largest synthetic kernel. Tune the order using the eventual target
session count, planning cadence, and latency budget. Rust is a possible bounded
kernel implementation later; it is not the first remedy for rescanning history,
duplicating HTTP calls, or blocking request dispatch.

Distinguish **controller saturation** from **worker overload**. If sustained
admitted work exceeds GPU/KV capacity, even a zero-cost controller cannot make
all queues stable. Admission limits, load shedding, routing, batching/cache
efficiency, or additional capacity must address that imbalance. This audit does
not identify the dominant GPU-side constraint because no real worker/network
load experiment was run.

## Reproduce and interpret

```bash
uv run python -m atfm_experiments.profile_bottlenecks --out runs/bottleneck-audit
uv run python -m atfm_experiments.benchmark_cpu \
  --cases forecast --sizes 8192 100000 --repeats 3 --profile \
  --out runs/bottleneck-forecast
```

Run sequentially. GDP is profiled once per size with fixture construction
excluded. JSONL uses one warmup and three timed drains of the unchanged file;
events are discarded between measurements. The largest file is 119 MB of
synthetic progress events. Queue setup is excluded; it starts with all entries
eligible and one available slot, then signals individual completions without
simulating service time. Three repetitions follow one warmup. Forecast timing
and cProfile are separate runs. The environment is the same workstation as the
[stress study](2026-09-27-gdp-stress.md); affinity and background activity are
uncontrolled. Every run asserts expected counts or deterministic fingerprints.

Evidence:

- [JSONL and queue timings](results/bottlenecks-2026-09-27/timings.json),
  [environment/source hashes](results/bottlenecks-2026-09-27/environment.json).
- GDP profiles at [100,000](results/bottlenecks-2026-09-27/gdp-100000-profile.txt)
  and [one million](results/bottlenecks-2026-09-27/gdp-1000000-profile.txt) sessions.
- [Forecast timings](results/bottlenecks-2026-09-27/forecast.csv),
  [profile](results/bottlenecks-2026-09-27/forecast-profile.txt), and
  [environment/source hashes](results/bottlenecks-2026-09-27/forecast-environment.json).

The earlier H2 profiles were taken before subsequent simulator changes and are
not presented here as measurements of the current simulator. This audit covers
the current GDP, forecast, JSONL, and live admission-queue implementations.
