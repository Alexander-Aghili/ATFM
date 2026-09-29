# Proxy request phases, exact peer ranks, and rejected pool tuning

Date: 28 September 2026. This follows the [board isolation study](2026-09-28-board-isolation.md).
The retained changes are request-phase observability and exact queued-peer counts.
A larger upstream HTTP connection pool was tested and reverted. Prediction
transport, board ownership, scheduling policy, and HTTPX pool defaults are unchanged.

## What the profile found

A thread-aware Yappi CPU profile of revision `36f0860` ran the real local proxy,
board and control loop with 1,024 burst sessions, three turns each, a 64-slot proxy
and fake worker, and 5 ms fake service. All 3,072 calls completed. The proxy main
thread used 23.016 CPU seconds; its four prediction threads used about 2.388 in
total. Cumulative costs included 2.674 seconds in HTTPcore connection assignment,
1.278 seconds collecting peer indices, and 1.167 seconds opening upstream TCP
connections. There were 3,072 upstream TCP connects for 3,072 calls.

These costs overlap and must not be added together. Profiling changes scheduling;
these totals locate work, not uninstrumented request overhead or production
capacity. The profile uses 120-second request/drain and 20-second control timeouts
to permit instrumentation. Unprofiled trials retain the normal 20/30/2-second
limits. Prediction admission remains four unfinished jobs with a 50 ms caller
budget in both modes.

## Separate the waits before optimizing

Revision `f180401` adds three trace/report phases:

| Phase | Source | Interpretation |
| --- | --- | --- |
| Prediction wait | Monotonic `prediction_s` | Elapsed time awaiting the local prediction attempt or fallback. |
| Admission wait | `t_admitted - t_enqueued` | Time after prediction until a slot is granted, including imposed holds. |
| Release dispatch | `t_release - t_admitted` | Delay before the released handler resumes forwarding. |

`t_admitted` comes from the existing queue release timestamp. Trace timestamps
are wall-clock values and are recorded only for completed traces; clock changes
can invalidate their differences. Older traces without new fields contribute no
phase samples. Quantiles from separate distributions cannot be added to recover
a request quantile. These fields locate waiting, not its unique root cause.

## Rejected experiment: retain more upstream connections

Revision `14cc14a` tested an idle keep-alive limit of `max(20, admission_window)`
with a total connection limit of at least 100 and the window size. The 64-slot
workload therefore retained 64 rather than 20 idle connections. Both streaming
and ordinary chat used the same configured client. Seven real-HTTP tests checked
reuse across a simultaneous 32-request batch, explicit lower/zero limits, and
owned versus injected client cleanup. They passed, but performance did not.

The installed HTTPcore 1.0.9 pool implementation repeatedly traverses its request
and connection lists. Its idle cleanup at the measured revision also evaluates
the length of a list of idle predicates, which counts all connections. Increasing
the retained pool removed many TCP setups but exposed more expensive pool scans.
The rejected candidate profile opened only 70 upstream connections, while
connection-assignment cumulative CPU increased from 2.674 to 11.836 seconds and
calls from 6,144 to 21,029. There were about 3.76 million connection `is_idle`
calls. Main-thread CPU increased to 31.150 seconds.

That candidate profile was **incomplete**: 3,072 calls were planned, 3,070 were
attempted, 3,069 succeeded, and one returned 502. Two subsequent turns were never
issued. It is retained as diagnostic evidence, not a successful load benchmark.
Its request/error logs remain in the archive. The unprofiled pool comparisons all
completed successfully. The candidate was reverted in `880b232`, including its
configuration, CLI and tests. No new pool-tuning API is retained.

## Retained implementation: exact counts of peer priorities

Revision `3084f34` adds `proxy/peers.py`. For each tier, it stores a count for each
index value and a total number of queued peers. If n peers remain and b have
indices strictly below the outgoing request's index, the hint is

`bucket = min(3, floor(4 * b / n))`, or `3` when `n = 0`.

Identical values share a count. `HoldQueue` updates it when entries are submitted,
cancelled, released, promoted or rebuilt. Held peers count; released peers do not.
The existing list-based helper remains available. Strict ties, empty tiers,
signed zero and nonfinite comparison behavior match the old NumPy definition.
Only queue-owned mutations may change an entry's tier/index after submission.

| Operation | Before | Retained implementation |
| --- | --- | --- |
| Rank query | O(Q) traversal plus list/array allocation | O(U_tier) distinct-index traversal, no peer list/array |
| Additional mutation bookkeeping | None | Expected O(1) insert/remove; two updates on promotion |
| Additional persistent storage | None | O(U) counts across all tiers |
| Whole index rebuild | O(Q) heap construction | O(Q), including count construction |
| All-unique worst case | O(Q) per rank | Still O(Q) per rank |

Q is the entire queued backlog; U_tier is the number of distinct indices in the
queried tier and U their total across tiers. Draining Q unique priorities can
still require O(Q²) total ranking work despite logarithmic heap admission.
An order-statistics tree would address that bound but adds balancing, deletion
and nonfinite-value handling. The histogram is a smaller exact change for
repeated values, not a claim that priority diversity is always low. It does not
change the scheduler's ordering, holds, admission window or overflow policy.

## Isolated rank-query timing

One local process constructs each backlog outside timing, verifies equal answers,
and reports the median of seven repetitions. Both functions query the same
candidate queue: the baseline explicitly collects the original peer list and
uses the old NumPy formula. All peers occupy one tier. Calls per repetition are
1,000, 100 and 10 for the three respective sizes. Mutation and setup costs are
excluded. This is a query microbenchmark, not a full queue throughput test.

| Queued peers | Distinct indices | Old query (microseconds) | Histogram (microseconds) |
| ---: | ---: | ---: | ---: |
| 1,000 | 1 | 51.72 | 0.45 |
| 1,000 | 16 | 51.79 | 0.80 |
| 1,000 | 1,000 | 51.06 | 20.91 |
| 10,000 | 1 | 518.57 | 0.42 |
| 10,000 | 16 | 514.50 | 0.74 |
| 10,000 | 10,000 | 515.25 | 202.94 |
| 100,000 | 1 | 6,153.08 | 0.45 |
| 100,000 | 16 | 6,238.12 | 0.72 |
| 100,000 | 100,000 | 6,212.76 | 2,103.21 |

## Paired HTTP trials

Each experiment independently compares `f180401` against its candidate in fresh
processes, sequentially, with seeds 7, 8 and 9 at 256 and 1,024 sessions. Pair order
alternates by seed. The shared interpreter is Python 3.12.13; the host is an Intel
Core Ultra 9 185H. Soft file-descriptor limit is 8,192. Both sides include identical
phase instrumentation. Candidates use unchanged committed runtime hashes; the
baseline uses a detached checkout with explicit source paths.

The workload uses 64 admission and fake-worker slots, 5 ms service, three
sequential calls per session, two-second open-loop session arrivals, burst-aligned
tools, fresh client connections, four prediction jobs, 50 ms prediction budget,
128 draws, ten 250 ms forecast slots, 500 ms hold cap, and 500 ms sleep after each
control step. The generator, board, proxy and worker share the same host. No GPU,
KV transfer, token streaming or model inference is simulated by the fake worker.
Ranges below are minima and maxima of three per-run measurements, not confidence
intervals. Coverage pools timely prediction counts within each group.

| Experiment | Sessions | Revision | Client p95 range (s) | Proxy CPU range (s) | Timely prediction use |
| --- | ---: | --- | ---: | ---: | ---: |
| pool | 256 | before | 0.378-0.833 | 1.824-2.009 | 66.97% |
| pool | 256 | after | 0.427-0.467 | 1.968-2.178 | 66.71% |
| pool | 1024 | before | 2.681-2.743 | 6.932-7.019 | 33.33% |
| pool | 1024 | after | 4.087-4.466 | 8.103-8.744 | 35.00% |
| ranks | 256 | before | 0.356-0.432 | 1.788-1.883 | 69.88% |
| ranks | 256 | after | 0.352-0.376 | 1.788-1.842 | 69.70% |
| ranks | 1024 | before | 2.564-4.067 | 6.812-8.940 | 30.83% |
| ranks | 1024 | after | 1.823-2.630 | 6.664-6.818 | 33.51% |

All **46,080 unprofiled calls** succeeded, with no client, session, control,
probe or tool-publication errors. Final prediction jobs were drained and peak
unfinished jobs stayed at or below four. The pool candidate increased client p95
in all three large pairs; it was rejected. In those pairs, admission-wait p95 rose
from 2.207-2.423 seconds to 3.868-4.216 seconds, while prediction-wait p95 fell.
Reduced prediction waiting did not compensate for the longer admission wait.

The retained rank candidate reduced proxy CPU in all six pairs, but improved
client p95 in only four. At 1,024 sessions, seed 7 changed from 2.564 to 2.630
seconds (a regression), seed 8 from 3.983 to 2.582, and seed 9 from 4.067 to 1.823.
The large and variable baseline range prevents a robust single latency-speedup
claim. At 256 sessions, seed 9 regressed from 0.356 to 0.371 seconds. All individual
values, including these regressions, remain in `comparison.csv`.

For the rank candidate's large cases, admission-wait p95 was 1.230-2.196 seconds,
prediction-wait p95 0.025-0.086 seconds, and release-dispatch p95 0.003-0.016 seconds.
The largest measured pre-forward wait remains admission. A slot stays occupied
through upstream response handling; this observation does not distinguish holds,
transport delay, service, or event-loop scheduling as a unique cause.

A final instrumented run completed all 3,072 calls. The rank query used 0.0293
cumulative CPU seconds across 3,072 calls; histogram add/remove work used about
0.0127 seconds combined (3,553 adds including rebuilds, 3,072 removals). The old
profile spent 1.278 seconds collecting peer lists alone, before NumPy ranking.
The retained profile's main thread used 22.461 CPU seconds. HTTPcore connection
assignment still used 2.550 seconds, and 3,072 upstream connections were opened.
Query work is reduced; the transport cost is not solved by this change.


## Validation and evidence

The retained code passes **496 tests, three skips and six existing warnings**.
All tracked Python functions stay within 20 physical lines. Rank tests cover
strict ties, boundaries, empty tiers, signed zero, infinities, NaNs, cancellation,
release and backlog replacement. Eight randomized differential traces compare
8,000 queue operations with the original scheduler, checking rank probes at every
step as well as release order and state. The 10,000-entry fixture verifies that
three tiers with five repeated values occupy exactly 15 histogram entries.
Phase tests verify legacy trace omission and timestamp calculations; real-HTTP
load tests verify phase sample counts. Optional Yappi tests run in this environment.

[Archived evidence](results/proxy-ranking-2026-09-28/) includes every trial,
profiles, configuration, requests, traces, observations, logs, shutdown records,
source hashes, executed drivers, the query microbenchmark and final pytest log.
The manifest covers 559 files. Large text and pstats files use gzip. `manifest.json` records uncompressed SHA-256
and byte lengths; `comparison.csv` retains per-case outcomes, including the failed
profile. Runtime source hashes are checked against their recorded Git revisions.
Profiled and unprofiled results are kept in separate groups.

To reproduce, use clean checkouts of the revisions named above with the same
interpreter/dependencies, adapt the archived drivers' absolute paths, set
`PYTHONPATH` to each checkout's `src` and `experiments/src`, and use new output
directories. Preserve configurations, descriptor limit and alternating order.
The host was not reserved; sequential pairing reduces but does not eliminate
machine-load or scheduling noise. Different runs deliver different control-event
schedules, so total CPU includes differing numbers of control cycles. Do not
substitute the older study's baselines for these fresh paired measurements.

The next target is upstream transport behavior under a busy admission window,
including pool maintenance, concurrent connection creation and time until a slot
is freed. A focused real-HTTP transport fixture should precede another pool or
transport change. Prediction transport is still worth measuring, but its small
CPU profile share does not justify assuming it dominates the observed delay.
A Rust rewrite is not established as necessary by these measurements.
