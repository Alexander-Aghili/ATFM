# Local HTTP load harness

This experiment runs the real ATFM proxy, board, and control loop against a fake
model worker. It tests the integrated request/control path without a GPU or paid
model API. All load-specific code and diagnostic routes live in the independent
`atfm_experiments.load` package; the core runtime is unchanged.

## Run

From the repository root on Linux/POSIX:

```bash
uv sync --extra dev --extra serve
uv run python -m atfm_experiments.load \
  --sessions 16 64 256 --patterns staggered burst --out runs/http-load-baseline
```

The runner executes cases sequentially, starting fresh processes and state for
each case. Output case directories must not already exist. It binds inherited
sockets to `127.0.0.1` on OS-assigned ports, avoiding free-port selection races.
No external service, authentication secret, Phoenix server, or production
endpoint is involved. Child processes are stopped/reaped on normal completion,
startup failure, and exceptions; each case retains shutdown exit codes and logs.
The serving extra supplies Uvicorn; optional real-socket tests skip without it.

Pass a JSON file with `--config path.json` to override defaults. Fields omitted
from JSON retain defaults. `--sessions` and `--patterns` override those two fields
for the sweep. Unknown fields, nonfinite settings, invalid ranges, and runs over
one million planned requests are rejected. For example:

```json
{
  "turns": 4,
  "arrival_window_s": 10,
  "worker_slots": 8,
  "worker_service_s": 0.05,
  "proxy_window": 16,
  "tool_mean_s": 0.15,
  "burst_period_s": 0.5,
  "control_interval_s": 0.5,
  "draws": 128,
  "slots": 10,
  "slot_s": 0.25,
  "max_hold_s": 0.5
}
```

Use `python -m atfm_experiments.load --help` and
[`LoadConfig`](../../experiments/src/atfm_experiments/load/config.py) for all knobs,
including capacities, client connection reuse, prediction/request timeouts, drain deadline, monitoring
interval, class mix, seed, and optional worker failure injection.

## Workload and architecture choices

Three independent server processes communicate over real localhost HTTP. The
parent owns load generation, monitoring, and a synchronous `ControlLoop` running
in a background thread. Combining these services in one asyncio loop would make
board CPU stalls artificially block both worker and generator. Startup and model
fitting precede the measurement period. Parent and children still share one
machine's CPU and filesystem; this is not a distributed deployment benchmark.

New sessions arrive at `i * arrival_window / sessions` independently of existing
session progress. This is **open-loop session arrival**, not a fixed population
that slows its own arrival rate when requests become slow. Within a session,
agent behavior is naturally sequential: model request, tool execution with a
midpoint progress event, tool completion, then the next model request. There are
three model turns per session by default, and one outstanding request per session.
No tool runs after the final model turn. Session completion does not invent a
session-end event absent from the core schema: board entries follow the existing
inactivity expiry and may outlive the generated turns. Failed model calls stop that session;
there are no automatic retries to hide errors or inflate offered work.

The staggered scenario uses reproducible per-session tool durations in
`[0.5 * tool_mean, 1.5 * tool_mean]`. The burst scenario rounds tool completion
up to the next shared `burst_period` boundary. It aligns completions without a
barrier that could hang when a session fails. Class and tool schedules are seeded;
OS scheduling, request order, sampled board state, and timings are not deterministic.
A separate synthetic training trace fits the real progress predictor. Forecast
accuracy is not evaluated by this fixture.

The load client's `client_keepalive_connections` defaults to zero: each model
call gets a fresh connection. A shared reuse pool produced multi-second delays
before request headers were sent during the initial large burst trial, so it
must not silently throttle the offered load. Set this field to a positive value
(up to the session count is used) to test connection reuse as its own workload
choice. Fresh connections have TCP setup overhead and are not a claim about the
best production-client configuration. Transport timings, source hashes, and the
config distinguish the two experiments; do not compare their results as a core
scheduler speedup.

The fake worker limits simultaneous service using an asyncio semaphore. Each
admitted request sleeps for `worker_service_s` and returns a nonstreaming response
with fixed token counts. It exposes worker waiting/active counts and optionally
returns HTTP 503 every N requests. It does not model token streaming, KV pressure,
cache reuse, batch efficiency, transfer cost, or GPU throughput. Its nominal
`worker_slots / worker_service_s` service rate is only a toy-worker parameter.
Reported response time is **full-response latency**, not time to first token.

Proxy and tool events use the local append-only JSONL bus. The actual board tick,
forecast, GDP planning, batched directive delivery, prediction HTTP calls, and
admission queue all run. Capacities are explicit synthetic constants, not scraped
GPU metrics. The default ten-slot horizon uses short durations to exercise several
control cycles in a brief trial; it is not a reproduction of the 30-second-slot
local deployment config. Control cadence preserves `ControlLoop.run` semantics:
work duration plus the configured sleep interval.

## Evidence files and interpretation

| File | Meaning |
| --- | --- |
| `config.json`, `environment.json` | Exact configuration, versions, CPU/platform, Git revision and source hashes, including uncommitted runner source. |
| `requests.jsonl` | Every attempted client call, class/turn, HTTP status or error, full-response duration, and HTTP transport timestamps. Timeout/cancel records remain present. |
| `arrivals.json`, `tools.json` | Intended and actual session-launch/tool-completion times and lateness. |
| `events.jsonl`, `proxy-trace.jsonl` | Core event stream and completed proxy traces; timed-out client work can still be pending at collection. |
| `control-http.jsonl`, `control-steps.jsonl`, `control.jsonl` | HTTP phase timings/statuses, whole-step duration, applied counts, and normal controller log. |
| `observations.jsonl` | Repeated queue, worker, snapshot-age, CPU/RSS, endpoint timing and event-loop-lag observations, including probe errors. |
| `summary.json` | Aggregate outcomes and distributions, class breakdowns, observed queue peaks, control errors, and final observations. |
| `stack.json`, `shutdown.json`, `*.log` | Ephemeral local endpoints/PIDs, reaped child exit codes, and server diagnostics. |

Client distributions include failed attempts; successful-only distributions are
also reported. A drain deadline cancels unfinished sessions and marks the run
incomplete instead of silently treating them as completed fast requests. Requests
that were never issued because an earlier turn failed/cancelled are visible in
the difference between `maximum_requests` and `requests_attempted`. Client
cancellation need not instantly cancel work already executing in the server.
Request completion throughput includes the arrival window and workload drain;
`elapsed_s` additionally includes final control/probe collection. It is not a
steady-state capacity estimate or the scheduled independent-request arrival rate.

Client HTTP trace milestones record elapsed time from call start to TCP setup,
request-header send, response-header receipt, and response closure. Reused
connections have no new TCP events. They use HTTPX/httpcore's trace extension,
whose event names are version-dependent; both package versions are recorded.
These timestamps help identify delays before request headers are sent. They do
not expose a dedicated connection-pool wait metric. Client-to-proxy timestamp
joins use session/turn IDs and the shared machine's wall clock; the proxy timestamp
is recorded after parsing the request body, not at TCP ingress. Those joins include
only calls with available completed proxy traces. Wall-clock adjustments can
invalidate cross-process timestamp subtraction; durations use monotonic clocks.

Arrival-to-release time from proxy traces includes prediction, explicit holds,
and queue waiting. It is not pure queue time. Probe observations may miss short
queue spikes; peak queue lengths are **observed lower bounds**. Server endpoint
and loop-lag percentiles use the most recent 4,096 samples, while endpoint total
counts are cumulative. CPU seconds are cumulative since diagnostic setup; compare
successive samples for utilization. Linux RSS is current resident memory;
`peak_rss_native` is the OS-native process high-water mark (KiB on Linux).

A delayed board probe is evidence too: do not interpret missing samples as a
healthy board. Generator launch/tool lateness reveals when parent scheduling or
synchronous JSONL writes cannot keep up. Full client logs, process diagnostics,
HTTP probes, and synchronous trace flushes themselves add overhead. Keep the same
instrumentation/configuration when comparing runs. This first harness does not
measure the proxy's exact prediction fallback count; completed board calls are
not equivalent to predictions used before the caller's deadline.

The CLI retains all artifacts and exits nonzero if requests are incomplete or
failed, the drain deadline is hit, session/tool publication fails, or control/probe
errors occur. Fault-injection runs can intentionally produce that exit status.
Repeated trials and longer arrival windows are needed before identifying a stable
saturation threshold. Compare scenarios by queue growth, latency, generator
lateness, control duration, and snapshot age; do not add component times together.

The [first six-case baseline](../research/2026-09-27-http-load-baseline.md) includes
raw evidence and an example of why client latency must be separated from
proxy scheduling time.

## Tests

```bash
uv run pytest -q tests/experiments/test_load_harness.py
```

Tests check configuration bounds, shared burst deadlines, worker concurrency,
error injection, cancellation, actual HTTP agent/control flows, event counts,
refusal to overwrite evidence, drain-deadline reporting, and child cleanup after
startup failure. Timing thresholds are not used as performance assertions.

### Repeated control attribution

Run paired trials with fresh processes, identical seeds, and alternating order:

```bash
python -m atfm_experiments.load.attribution --out runs/control-attribution \
  --sessions 256 1024 --repeats 3
```

The default is burst traffic. Use `--config` for other workloads. Each pair
shares all settings except `control_enabled`; repeat seeds increase by one.
Existing case directories are rejected before starting. Each completed case is
persisted immediately, including failed runs; failure makes the command exit
nonzero. Run sequentially on an otherwise idle host.

Control-off leaves request predictions enabled but never ticks the board or
applies directives. Its board therefore has no refreshed session registry or
forecast. This is an operational ablation of the entire periodic control path,
**not** a comparison with equally informed predictions or a measure of policy
quality. Differences can include policy effects, state population, compute,
and changed request timing.

The proxy exposes caller outcomes and bounded worker activity in `/state` and
harness observations. See [prediction work accounting](#prediction-work-accounting)
for current definitions; archived trials retain their original counter set.
Counters are process-local and reset on restart. No per-request history is
retained in the production proxy.

At large concurrency, check `ulimit -Sn`: inbound waiting requests retain
sockets, and the proxy also needs upstream/prediction connections and log files.
Provenance records the inherited soft/hard `RLIMIT_NOFILE`; the harness does
not change it automatically. For an explicit local capacity experiment:

```bash
(ulimit -Sn 8192
 python -m atfm_experiments.load.attribution --out runs/control-attribution-fd8192 \
   --sessions 1024 --repeats 3)
```

Check server logs as well as client outcomes for resource exhaustion. Client
transport failures retain exception type and message in the raw request record.

### Profiling the proxy

Install the experiment-only profiler, then set `"profile_proxy": true` in a
load configuration:

```bash
uv sync --all-packages --extra dev --extra serve --extra profiling
```

After graceful shutdown, the case directory contains `proxy.pstats` (merged
functions), a per-thread function CSV, text rankings by self/cumulative CPU time,
and JSON with profiler version and per-thread totals. Missing profile output
makes the run fail explicitly, for example after a forced kill.

[Yappi](https://github.com/sumerc/yappi) uses its CPU clock and separate thread
contexts to distinguish the serving thread from synchronous prediction workers.
Profiling starts in application lifespan and stops before shutdown cleanup.
It includes serving/health/probe work but excludes imports and app construction.
CPU time excludes waiting; cumulative function times overlap and must not be
summed as exclusive cost. The exported pstats format merges threads; use the CSV
for thread attribution.

Instrumentation changes scheduling and timeout frequency. Use profiles to locate
work, then validate changes in unprofiled runs with identical workload settings.
The initial cProfile trials are retained as exploratory evidence only: on this
Python 3.12.13 build, a minimal reproduction captured worker-thread calls in a
single profile and produced ambiguous concurrent timing attribution.

The [proxy profiling study](../research/2026-09-27-proxy-profiling.md) found repeated
missing-`sniffio` searches in HTTPcore 1.0.9. The serving extra now supplies it
explicitly. Compare environments as well as source: provenance records AnyIO,
HTTPcore, sniffio (including absence), and Yappi versions. Faster HTTP processing
can change prediction fallback frequency even when all requests succeed.

## Prediction work accounting

Set `prediction_limit` (default 4) and `prediction_budget_s` (default 0.05) in
`LoadConfig`. Service configuration calls the latter `board_timeout_s`.

| Counter | Meaning |
| --- | --- |
| `attempted` | Enabled prediction requests, including admission rejection. |
| `used`, `timeout`, `error`, `cancelled` | Mutually exclusive terminal caller outcomes for admitted/submitted work; submission failures count as errors. |
| `rejected` | Immediate fallback because prediction capacity is full or admission is closed. |
| `pending` | Callers still awaiting an outcome. |
| `disabled` | Requests with no predictor configured; excluded from attempts. |
| `accepted` | Successfully submitted jobs. |
| `outstanding`, `peak_outstanding` | Current/maximum unfinished admitted jobs, including abandoned work. |
| `running` | Jobs inside the predictor invocation. |
| `completed` | Finished futures, including failures and cancellation before dispatch. |
| `cancelled_before_start` | Submitted futures cancelled without running. |
| `expired_before_start` | Dispatched jobs whose deadline already passed; predictor was skipped. |
| `late_completed` | Non-cancelled futures whose completion callback observed an expired deadline. |

In a coherent snapshot, `attempted = used + timeout + error + cancelled + rejected
+ pending`, and `accepted = completed + outstanding`. Worker completion is not
prediction use. A cancelled caller's job can complete before its deadline and
still be unused; `late_completed` is not a count of all unused results.

Compare old/new runs using `used / attempted`, every fallback outcome, request
p95, CPU, and errors. A lower timeout percentage alone can hide increased
rejection. Bounded admission promises bounded work, not more prediction coverage
or a policy-quality improvement. The [overload study](../research/2026-09-28-prediction-overload.md)
records paired trials and limitations.
