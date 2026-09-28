# Python, performance, and when to use Rust

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
| Board ticks | `board/service.py` calls `board.step` synchronously inside an async endpoint. | Forecast computation blocks that event loop; measure tick duration and overlapping `/predict` latency. |
| Per-request predictions | `proxy/board_client.py` makes a request for service time and another for next-tool duration. | Two HTTP round trips and a four-worker prediction pool; consider a combined response path before changing language. |
| Event ingestion | `JsonlBus.drain()` rereads the event file; HTTP ticks apply the returned events. | Cost grows with file history, and old events are reapplied. An incremental cursor needs correctness tests as well as timing measurements. |
| Forecast aggregation | Per-session Python loops plus `(horizons, draws)` NumPy operations. | Work grows with sessions, horizons, draws, and fan-out. Profile sampling versus aggregation and allocation. |
| Admission queue | `HoldQueue.tick()` scans pending entries and selects eligible requests. | Deep queues can cause repeated linear scans; measure queue depth and scheduling CPU time. |
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
for speedups and bounds. The live HTTP/JSONL concerns above remain separate,
unfixed findings; CPU benchmark success is not a live-service load test.

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
