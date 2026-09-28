# Repeated HTTP control attribution

## Question and design

Does periodic board/control work explain the integrated latency observed in the
[first HTTP baseline](2026-09-27-http-load-baseline.md), and how often do request
predictions actually reach their caller before the deadline?

The experiment uses the real local proxy, board, and control loop with the
bounded fake worker. Source revision: `88f3079`; each case also records source
hashes, package versions, CPU/platform details, and its complete configuration.
Three seeds (7–9) are paired across periodic control on/off, alternating pair
order. Cases run sequentially, with fresh processes for every case. CPU affinity,
frequency scaling, and background host activity are uncontrolled. Repetitions
describe local variation; they are not a confidence interval or independent
hardware replications.

Common settings: burst-aligned tool returns; three model calls per session;
new sessions spread across two seconds; eight worker slots; fixed 50 ms service;
proxy window 16; caller prediction budget 50 ms; 128 forecast draws; ten slots;
periodic control sleeps 500 ms after each step. Client connections are fresh,
following the earlier transport investigation.

Control-off disables periodic ticks and actuation together, but leaves the
per-request board client running. The board is consequently unpopulated and has
no forecast snapshot. This intentionally measures the combined operational
difference; it cannot isolate CPU interference from policy or prediction-state
effects. Shared seeds do not produce identical wall-clock event trajectories:
each session waits for its own preceding response.

## Reproduction and evidence

```bash
UV_PROJECT_ENVIRONMENT=/tmp/atfm-load-check uv sync --frozen --extra dev --extra serve
/tmp/atfm-load-check/bin/python -m atfm_experiments.load.attribution \
  --out runs/control-attribution-2026-09-27 --sessions 256 1024 --repeats 3
```

The [load-testing guide](../development/load-testing.md#repeated-control-attribution)
defines each prediction counter. A successful HTTP response does not establish
that its scheduling prediction arrived on time. Likewise, a completed board
HTTP call does not establish that the proxy used it. Counters measure the caller's
outcome; expired synchronous calls may still occupy one of four prediction
threads. Exact thread occupancy is not measured here.

Raw per-request transport traces, proxy traces, bus events, control timings,
observations, generator/tool timing, and shutdown records are retained with
configuration and provenance. Queue maxima are sampled observations, not exact
peaks. Process CPU is cumulative from diagnostics initialization, including
health probes and final collection. Board event-loop lag is sampled every 50 ms;
it is not a profiler or a direct measure of forecast duration.

## File-descriptor limit discovered during the first sweep

The shell's inherited soft `RLIMIT_NOFILE` was 1,024 (hard limit 1,048,576).
At 1,024 sessions, proxy logs reported `OSError: [Errno 24] Too many open files`
during socket accept. Some clients consequently received `ConnectError`; their
sessions stopped and never issued subsequent turns. Failed attempts and missing
turns are kept in the results, not retried or counted as successes.

These cases are evidence of an operating-system resource bottleneck, not a clean
controller comparison. A proxy needs descriptors for inbound connections,
upstream connections, prediction/control connections, and logs, not just one per
active worker. Waiting requests retain inbound sockets. Increasing the queue
or worker budget alone does not increase this process limit.

The corrective trials explicitly raise the soft limit to 8,192 **only in the
benchmark shell and its children**. No machine-wide or production setting is
changed. Higher limits remove this confound; they do not establish a safe
production session ceiling.

## Results

[Archived evidence](results/control-attribution-2026-09-27/) contains all 24 cases,
including the descriptor-limited failures. The
[comparison CSV](results/control-attribution-2026-09-27/comparison.csv) preserves
individual trials. The manifest records uncompressed SHA-256 and byte length
for every retained file; raw logs/traces are gzip-compressed. The corrected and
headroom runs use revision `c744fa5`, which adds descriptor provenance and error
messages without changing scheduling.

Ranges below are the minimum–maximum of the three **per-run p95s**, not pooled
latency percentiles. Timeout percentages pool caller timeout counts over
attempted predictions. They exclude clients that never reached the proxy.
All prediction errors other than timeouts were zero in these trials.

| Scenario | Control | Successful / planned calls | Client p95 range (s) | Prediction timeouts |
| --- | --- | ---: | ---: | ---: |
| 256 sessions, original limit | On | 2,304 / 2,304 | 1.380–1.417 | 7.73% |
| 256 sessions, original limit | Off | 2,304 / 2,304 | 1.374–1.394 | 5.38% |
| 1,024 sessions, original limit (confounded) | On | 9,210 / 9,216 | 7.998–8.104 | 35.10% |
| 1,024 sessions, original limit (confounded) | Off | 9,195 / 9,216 | 8.139–8.411 | 11.59% |
| 1,024 sessions, limit 8,192 | On | 9,216 / 9,216 | 8.170–8.276 | 37.47% |
| 1,024 sessions, limit 8,192 | Off | 9,216 / 9,216 | 8.267–8.690 | 10.88% |
| 1,024 sessions, faster worker, limit 8,192 | On | 9,216 / 9,216 | 3.599–4.228 | 56.98% |
| 1,024 sessions, faster worker, limit 8,192 | Off | 9,216 / 9,216 | 2.827–4.141 | 46.41% |

No case hit the drain deadline or reported control, control-HTTP, probe, or
tool-publication errors. The corrected trials had no client errors or descriptor
exhaustion in server logs. Timeout fallback kept HTTP calls successful; success
alone would have hidden the loss of prediction coverage.

### Worker saturation masks some control costs

Eight slots at 50 ms imply an ideal upper bound of 160 calls/second for this
fake worker, before HTTP and scheduling overhead. The corrected 1,024-session
runs delivered 147–151 calls/second. P95 arrival-to-release delay was 7.97–8.49 s,
while p95 time after release was 0.124–0.139 s. These are separate distributions;
their percentiles must not be added.

Periodic control did not produce a consistent latency penalty in this saturated
regime. It did coincide with much more prediction fallback: 37.47% versus
10.88%. Board maximum sampled event-loop lag was 197–214 ms with control,
versus about 2 ms without. Control-step p95 was 441–520 ms; the configured
500 ms interval is a sleep **after** work, not a guaranteed tick period.
Board CPU was 8.99–9.19 s per control-on run versus 1.64–1.83 s off.
The synchronous tick/directive handlers are a plausible interference source.
This ablation still combines computation, policy, and populated-state effects.

### Faster service exposes the proxy path

A second paired series changes only worker slots to 64, worker delay to 5 ms,
and proxy window to 64, keeping the 8,192 descriptor limit and the original
arrival schedule. This is a different diagnostic workload, not a production
configuration recommendation. Reproduce with:

```bash
printf '%s\n' '{"pattern":"burst","worker_slots":64,"worker_service_s":0.005,"proxy_window":64}' > headroom.json
(ulimit -Sn 8192
 /tmp/atfm-load-check/bin/python -m atfm_experiments.load.attribution \
   --out runs/control-attribution-headroom-2026-09-27 --sessions 1024 --repeats 3 \
   --config headroom.json)
```

The worker's ideal service capacity becomes 12,800 calls/second, but the
integrated finite workload delivered only 307–345 calls/second. This is **not**
a sustained capacity estimate: arrival structure, sequential turns, HTTP,
logging, and the load generator all contribute.

No worker waiters were observed. Proxy CPU was 9.08–10.17 s over roughly
8.9–10.0 s of workload, consistent with approximately one busy CPU core
(accounting windows differ slightly). Client time to sent headers had p95
255–497 ms, and maximum session-launch lateness was 73–201 ms, showing that the
generator/transport is also material. Proxy upstream p95 remained 537–654 ms
despite only 5 ms of fake service; this interval includes HTTP client/pool and
proxy scheduling, not just worker execution.

Prediction timeouts remained high even without periodic control (46.41%).
With control they reached 56.98%. End-to-end p95 differences varied across seeds,
including one seed where control-on was faster. This does not establish a
stable policy-latency effect. It does establish that removing simulated model
work does not make the rest of the system free.

## What to optimize next

1. **Profile the proxy under the faster-worker workload.** Separate HTTP client
   and serialization/logging costs, prediction submission/waiting, and queue
   ranking/maintenance. CPU counters identify a busy process, not the hottest
   function. Avoid assuming the remaining queue scan explains all of it.
2. **Bound prediction work and isolate control computation.** The four-thread
   synchronous prediction path and its longer transport timeout can retain work
   after the caller falls back. A bounded, deadline-aware request path and
   independently served immutable board snapshots are candidate changes.
   Preserve fallback semantics and validate state/RNG ownership before moving
   board work across threads or processes.
3. **Retest the same pairs after each change.** Compare successful throughput,
   prediction coverage, generator lateness, queue delay, CPU, and errors. Use a
   separate generator or reduce its overhead before claiming a service ceiling.

These findings do not justify a wholesale Rust rewrite. They identify resource
limits, concurrency/deadline design, synchronous CPU work, and HTTP overhead to
separate first. A native implementation could reduce a measured CPU hotspot,
but cannot remove the fake worker's service bound or fix descriptor sizing.
No conclusion about real GPU kernels, KV transfer, cache behavior, or model
serving capacity follows from these localhost tests.

## Verification

Full suite after instrumentation and attribution support: **407 passed,
6 skipped**, with seven existing dependency/empty-slice warnings. Tests cover
prediction success, error, timeout, cancellation, disabled prediction, seed
pairing/order, descriptor provenance, and a real HTTP control-off trial.
All 468 archived files pass their recorded uncompressed length/hash checks.
The updated paper section compiles without unresolved references, overflowing
boxes, missing characters, or duplicate labels and has been visually checked.
