# Python, performance, and when to use Rust

The [current bottleneck audit](../research/2026-09-27-current-bottlenecks.md)
records the pre-change profiles. [Implementation decisions](control-scaling.md)
track the subsequent fixes and their validation.

## Current decision

Keep the core in Python until measurements identify a runtime limit that
algorithmic or architectural changes cannot meet economically. This is an
engineering recommendation, not a measured production capacity claim.
[CPU scaling trials](../research/2026-09-27-cpu-scaling.md) now quantify selected
CPU workloads and document verified optimizations. The architecture's
original D7 decision selected Python for research iteration and access to the
numerical and serving ecosystem.

ATFM orchestrates an external LLM server; it does not implement token generation
or GPU kernels. Forecasting uses NumPy/SciPy, and trace evaluation uses pandas.
Many NumPy operations run outside the Python interpreter lock, although Python
loops and object operations do not automatically gain that benefit. See the
[NumPy thread-safety documentation](https://numpy.org/doc/stable/reference/thread_safety.html).

The research runners live in the separate `atfm-experiments` workspace package.
This packaging boundary makes runtime ownership clearer but is not itself a
performance optimization.

## Concrete places to measure

| Area | Current implementation | Scaling risk / first investigation |
| --- | --- | --- |
| Board ticks | `board/service.py` submits forecasting and planning to one bounded worker. | Prediction reads use a published view; measure GIL contention, view age, stage timings and event-loop lag. |
| Per-request predictions | `proxy/board_client.py` retrieves service time and next-tool duration together. | One HTTP round trip; unfinished work is bounded, including jobs whose callers timed out. |
| Event ingestion | `JsonlBus.drain()` consumes complete appended records using a per-instance byte cursor. | O(new bytes) parsing; idle drains perform a metadata check. Restart replays once; downstream transactional recovery remains separate. |
| Forecast aggregation | Per-session loops with streamlined empirical draws and `(horizons, draws)` NumPy operations. | Work grows with sessions, horizons, draws, and fan-out. Profile sampling versus aggregation and allocation. |
| Admission queue | `HoldQueue` maintains ready, delayed, promotion, and FCFS heaps. | Admission release is amortized O(log Q); exact peer ranks scan U distinct indices in the tier, while diagnostic counts remain O(Q). |
| Logging | Request completion writes and flushes a trace; JSONL publishing opens/appends a file. | Synchronous disk work can delay the request handler. Measure event-loop lag and buffered/asynchronous alternatives. |
| Simulation sweeps | Python event loop and worker/policy logic across many arms and seeds. | CPU cost affects experiment throughput, not automatically live request latency. Parallelize independent runs or optimize measured kernels. |

These are code-review findings, not proof of the dominant bottleneck. The
[Python asyncio guidance](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
explains why synchronous CPU work delays other tasks on an event loop. Moving
work to threads requires explicit state ownership; additional service workers
also need coordinated registry/queue state rather than independent copies.

## Measurement before a rewrite

Use a representative range of active sessions, queued requests, prompt sizes,
event rates, forecast horizons, and Monte Carlo sample counts. Separate upstream
model latency from ATFM's overhead.

Measure:

- Proxy-added p50/p95/p99 latency and first-byte latency under streaming load.
- Event-loop lag, queue scheduling time, CPU utilization, and memory growth.
- Board tick p95/p99 duration, snapshot age, and overlap with prediction calls.
- Prediction timeout/fallback frequency and worker-pool saturation.
- Telemetry throughput as the event log grows, including restart semantics.

Compare a direct-to-upstream baseline, proxy without the board, and proxy with
the board. The current 50 ms prediction budget is a fallback deadline, not a
latency target. The default five-second control interval is a scheduling
interval, not evidence that a tick finishes within it.

## Optimizations measured in the CPU study

Cache eviction now computes free space once per batch and walks candidate
victims once. Full workers defer queue sorting until admission is possible.
Training/replay consume batched session records instead of constructing and
converting a DataFrame per session. Conditional LLM-duration draws cache their
sorted empirical distribution, invalidated on refit. Paired bootstrap uses
positional selection, and explicit `MeanMetric` statistics prepare their values
once instead of rebuilding filtered DataFrames in every resample.

See the [results and complexity table](../research/2026-09-27-cpu-scaling.md)
for speedups and bounds. The live HTTP concerns above require separate measurements. Incremental JSONL
ingestion, bounded prediction admission, and board worker isolation are implemented;
GIL contention, freshness, and logging remain investigation targets. CPU benchmark
success is not a live-service load test.

## Large-session control paths

GDP now prepares exact empirical chance thresholds once per slot and reuses them
across sessions, removing the draw-count multiplier from assignment searches.
Admission selects each release batch once instead of rescanning the queue for
every released request. See the [large-session study](../research/2026-09-27-large-control-paths.md)
for measured gains, a small-workload regression, remaining linear scans, and
why neither result establishes that CPU cost is negligible compared with I/O.

The [larger stress study](../research/2026-09-27-gdp-stress.md) found 7.32-second
plans at 100,000 sessions and 10,000 slots, and 15.91-second plans at one million
sessions and 300 slots under saturation. Those synthetic cases exceed a
five-second serial control interval. They motivate further search/scope work;
they do not establish deployed fleet capacity.

## A targeted Rust path

If profiling shows a hot CPU kernel dominates forecasting or simulation, move
that bounded computation behind a stable Python interface and retain model
fitting, evaluation, and experiment tooling in Python. Preserve seeded semantics
or explicitly review any numerical differences.

If the proxy cannot meet an agreed throughput/tail-latency target after removing
avoidable I/O and queue overhead, a Rust admission/streaming service is a
reasonable next boundary. It can consume forecasts from the Python board over
the existing service boundary. Test cancellation, streaming cleanup, directive
expiry, ordering, and fallback behavior before replacing the current path.

A full rewrite now would duplicate scientific and integration validation without
an established performance target. Neither Python nor Rust fixes duplicate
network calls, repeated file scans, or an inefficient queue algorithm by itself.

## Function-boundary refactor (28 September 2026)

The repository now enforces a 20-physical-line function limit. This preserves
algorithms and public behavior but adds Python call boundaries; exact timing
equality is not assumed. See the [measured equivalence and timing record](../research/results/modularity-2026-09-28/).

One measured regression was repeated GDP hold-window boundary lookup. The
planner now memoizes the existing binary search by starting slot for each plan.
The cache is cleared before every plan, including when configuration changes.
It retains the original floating-point comparisons and never caches feasibility,
which changes as demand is committed. For U distinct start slots, this boundary
work changes from O(N log S) to O(U log S + N), with O(U) storage and U <= S.
The chance-constraint thresholds, range index, assignment order, tenant caps,
and scan fallback are unchanged.

## Bounded prediction work (28 September 2026)

The proxy now admits at most `prediction_limit` unfinished jobs, defaults to four,
and immediately falls back when full. This removes an unbounded executor waiting
queue. Deadline-aware HTTP requests use the remaining caller budget, replacing
the 500 ms minimum transport timeout. Accounting separates callers from worker
lifetimes. That admission change alone did not isolate board ticks or make arbitrary
Python work cancellable. The subsequent isolation work is described below. See the [implementation and paired load study](../research/2026-09-28-prediction-overload.md).

## Isolated board computation (28 September 2026)

The [board isolation study](../research/2026-09-28-board-isolation.md) measures the
follow-up change: one owner of mutable control state, atomic prediction views,
monotonic freshness, and worker-side JSON preparation. It retains the proxy's
four-job cap and 50 ms caller budget. Request prediction is O(1) in session count
and duration training samples after a per-tick O(S + D) projection. This trades
an extra O(S) view for detached request reads; it does not reduce Monte Carlo or
GDP planning complexity. CPU-bound Python in the worker still contends for the
GIL. Judge improvement using timely prediction use and view age, not timeout
counts alone; a stale result is deliberately unavailable.

## Exact peer ranking and HTTP transport (28 September 2026)

The [proxy ranking and transport study](../research/2026-09-28-proxy-ranking.md)
profiles the next request-path costs. `proxy/peers.py` maintains per-tier counts
of queued indices. Each release queries O(U) distinct indices instead of building
an O(Q) peer list and NumPy array. Insert/remove/promotion bookkeeping is expected
O(1), with O(U) additional storage. U can equal Q: this is not a logarithmic rank
index or a removal of the worst-case quadratic cost across Q releases. An
order-statistics tree remains a possible follow-up for diverse priorities.

A larger upstream keep-alive pool was tested and reverted after it increased
pool-management CPU and large-case latency. Fewer TCP connections alone did not
make the request path faster. The retained HTTPX defaults and the prediction
executor are unchanged. Use unprofiled paired runs for latency comparisons;
Yappi instrumentation changes scheduling and timeout behavior.

## Bounded upstream pool sharding (28 September 2026)

The follow-up [transport study](../research/2026-09-28-sharded-transport.md) isolates
HTTP work without the board or admission queue. Direct upstream traffic uses
16 smaller HTTPX pools when the initial admission window exceeds 20, with a
combined 100-connection cap. Smaller windows keep stock HTTPX by default because
the focused 8-concurrency fixture regressed latency despite lower CPU. This preserves reuse
while reducing the size of each internal pool scan. Selection is O(P) over P
shards; internal HTTPcore scans and reassignment remain. With fixed C=100 this is
a measured constant-factor optimization, not a new fleet-size complexity bound.

Response lifetimes drive load balancing, including streaming and error cleanup.
The shared TLS context avoids loading one certificate store per shard. A setting
of one shard or detected environment/system HTTP proxies preserves stock HTTPX.
The retained idle connection cap increases from 20 to 100 across shards, so
connection reuse trades additional idle sockets for less setup work. Idle expiry
remains five seconds. Compare CPU, latency, connection counts and workload shape;
a connection-count decrease alone did not justify the earlier single-pool change.
