# First integrated HTTP load baseline

This is the first smoke/load baseline for the separately packaged
[`atfm_experiments.load`](../../experiments/src/atfm_experiments/load/) harness.
It runs the real proxy, board, progress forecast, GDP planner, and control loop
against a fake model worker over localhost HTTP. No production runtime code was
changed. The [operator/developer guide](../development/load-testing.md) specifies
configuration, measurement definitions, cleanup, and limitations.

## Configuration and purpose

Each fresh case schedules 16, 64, or 256 new sessions over two seconds, with three
sequential model calls per session and tool work between calls. Session arrivals
continue independently of earlier sessions. The class mixture is seeded with a
25% interactive probability. Tool durations are uniform around a 150 ms mean;
burst completions align to shared 500 ms boundaries. The worker has eight service
slots and a fixed 50 ms service delay. The proxy admits up to 16 concurrent calls.
Control work runs every 500 ms plus step duration, with 128 forecast draws, ten
250 ms slots, a 500 ms hold cap, and synthetic capacity constants. These short
settings exercise multiple control cycles; they are not GPU calibration.

The proxy, board, and worker run in three independent Uvicorn processes. The load
generator and controller thread share a fourth process. Sessions use real HTTP
requests, JSONL events, and production queue/prediction/directive paths. The first
sweep ran sequentially on an Intel Core Ultra 9 185H with Python 3.12.13, HTTPX
0.28.1, and Uvicorn 0.53.0. Source hashes and exact configurations accompany every
case. CPU affinity and background activity were uncontrolled. Each row is one
finite run, not a confidence interval or a steady-state capacity estimate.

The initial sweep used a shared client pool retaining up to 256 connections
(capped by session count). The final harness makes reuse configurable and defaults
to fresh connections after the transport follow-up below.

## Baseline observations

| Tool completions | Sessions | Successful calls | Client response p95 | Observed proxy queue peak | Control-step p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Staggered | 16 | 48 / 48 | 58.6 ms | 0 | 15.1 ms |
| Staggered | 64 | 192 / 192 | 60.7 ms | 0 | 14.6 ms |
| Staggered | 256 | 768 / 768 | 1,285.8 ms | 158 | 79.4 ms |
| Synchronized bursts | 16 | 48 / 48 | 75.9 ms | 0 | 11.6 ms |
| Synchronized bursts | 64 | 192 / 192 | 262.5 ms | 6 | 22.4 ms |
| Synchronized bursts | 256 | 768 / 768 | 3,934.3 ms | 134 | 160.0 ms |

All 2,016 calls succeeded. There were no client, control, probe, or tool-publication
errors; no run reached its drain deadline. Every case issued hold updates and
completed multiple control cycles. All server processes were reaped. Higher load
and synchronized tool returns exercise contention, so the harness is suitable
for follow-up diagnosis. They do not establish the source of every delay.

The largest staggered case has client p95 near proxy request duration p95
(1.286 s versus 1.280 s). Its slowest calls spend about 1.21 s between proxy
arrival and release and about 0.10 s upstream. Admission time includes prediction,
policy holds, and queue waiting; upstream time includes worker queueing plus the
fixed 50 ms fake service. These intervals do not isolate policy effectiveness.

The largest burst case has a different signature: client p95 is 3.934 s, but the
proxy endpoint's p95 is 1.212 s. Matching by session and turn reveals calls spending
several seconds before the proxy's recorded request timestamp. For example,
`load-214`, turn 1, takes 5.750 s at the client, of which 5.617 s precedes the
proxy timestamp, 0.051 s is arrival-to-release, and 0.082 s is upstream. The proxy
records arrival after parsing JSON; this gap includes client/transport/body-read
work, not just queue scheduling. It must not be attributed to GDP or admission.

The largest burst case also observes a 56.6 ms board event-loop-lag maximum and
160 ms control-step p95. Those observations justify concurrent control/request
investigation, but do not explain the multi-second pre-proxy gap by themselves.
Maximum scheduled session-launch lateness was about 27 ms across the baseline;
that checks one aspect of generator health, not HTTP connection scheduling.
The harness does not yet count exact prediction fallbacks.

## Transport follow-up

After the baseline exposed the pre-proxy gap, the harness added per-request
HTTPX/httpcore trace milestones for connection setup, sending headers, receiving
headers, and response closure. Joined client/proxy intervals are now summarized
explicitly. Reused connections do not emit new connect events. Trace names depend
on the recorded library version; this is test-only instrumentation. It is not
an admission optimization, and differences between baseline and follow-up timings
must not be reported as runtime speedups.

The instrumented reuse follow-up completed 767 of 768 calls and retained one
client `ReadError`. Client p95 was 4.409 s; elapsed time before request headers
were sent had p95 4.282 s. The slowest example used a reused connection (no new TCP
connect event), waited 5.639 s before the send-headers phase started, then completed
about 0.11 s later. This locates substantial delay on the client side before
request transmission; it does not identify the precise HTTPX/httpcore mechanism.

A subsequent run with `client_keepalive_connections=0` completed 768 of 768 calls
with no request/control/probe errors. Its client p95 was 1.381 s, time to headers
sent p95 was 72.5 ms, and client-to-proxy-timestamp p95 was 80.5 ms. Observed proxy
queue peak was 144 and control-step p95 was 63.5 ms. The remaining delay now has a
much smaller pre-proxy contribution. These are single trials with different
connection policies, not a production optimization or proof that reuse is
universally harmful. Fresh connections add TCP overhead. The harness defaults to
fresh connections to avoid this particular shared-pool delay masking offered
bursts; reuse remains an explicit configuration choice for separate investigation.

[Instrumented reuse evidence](results/http-load-2026-09-27/transport-followup/summary.json)
and [fresh-connection evidence](results/http-load-2026-09-27/fresh-connections/summary.json)
include the failed attempt, raw transport timestamps, exact configurations, and
source hashes. The fresh-connection trial used the uncommitted configuration
change recorded by its source hashes; its Git revision alone is insufficient to
reconstruct that trial.

## Evidence and reproduction

[Aggregate summaries](results/http-load-2026-09-27/summary.json) and
[comparison CSV](results/http-load-2026-09-27/comparison.csv) preserve the first
sweep at commit `c2ed0b6`. Each case directory contains configuration, source
hashes, summary, shutdown records, and gzip-compressed raw requests, proxy traces,
events, control records, observations, arrivals, and tool completions. The
[manifest](results/http-load-2026-09-27/manifest.json) records uncompressed SHA-256
hashes. Logs have synthetic identifiers and localhost endpoints only.

```bash
uv sync --extra dev --extra serve
uv run python -m atfm_experiments.load \
  --sessions 16 64 256 --patterns staggered burst --out runs/http-load-new
uv run pytest -q tests/experiments/test_load_harness.py
```

Current runs use fresh connections and include transport milestones. To exercise
the initial reuse policy with current instrumentation, pass `--config` pointing
to JSON containing `{"client_keepalive_connections": 256}`. To reproduce the
original source exactly, use its recorded commit and package versions.
The initial full suite passed 399 tests with six optional skips. The 15 harness
tests include real socket integration, concurrency/failure/cancellation behavior,
startup failure cleanup, and drain deadline reporting. No timing threshold is
used as a performance assertion. The next step is repeated runs and attribution
using the captured intervals, before changing production scheduling or worker
architecture.
