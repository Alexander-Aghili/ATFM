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
including capacities, prediction/request timeouts, drain deadline, monitoring
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
No tool runs after the final model turn. Failed model calls stop that session;
there are no automatic retries to hide errors or inflate offered work.

The staggered scenario uses reproducible per-session tool durations in
`[0.5 * tool_mean, 1.5 * tool_mean]`. The burst scenario rounds tool completion
up to the next shared `burst_period` boundary. It aligns completions without a
barrier that could hang when a session fails. Class and tool schedules are seeded;
OS scheduling, request order, sampled board state, and timings are not deterministic.
A separate synthetic training trace fits the real progress predictor. Forecast
accuracy is not evaluated by this fixture.

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
| `requests.jsonl` | Every attempted client call, class/turn, HTTP status or error, and full-response duration. Timeout/cancel records remain present. |
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

## Tests

```bash
uv run pytest -q tests/experiments/test_load_harness.py
```

Tests check configuration bounds, shared burst deadlines, worker concurrency,
error injection, cancellation, actual HTTP agent/control flows, event counts,
refusal to overwrite evidence, drain-deadline reporting, and child cleanup after
startup failure. Timing thresholds are not used as performance assertions.
