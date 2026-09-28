# CPU scaling and the Python/Rust decision

Keep Python for the current core. These local trials found avoidable repeated
work, and removing it accelerated complete evaluations without a language
rewrite. They do not establish production capacity or proxy tail latency.
Experiments now live in the separate `experiments/src/atfm_experiments` package;
`src/atfm` contains reusable runtime, model, simulation, and evaluation code.

## Measured results

Median wall time from three repetitions after one untimed warmup:

| Workload | Size | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Evict half the resident contexts | 8,192 contexts | 649.84 ms | 4.18 ms | 155.5× |
| Conditional duration sampling | 8,192 observations | 12.76 ms | 2.24 ms | 5.7× |
| Generic paired bootstrap | 8,192 aligned sessions | 933.94 ms | 144.88 ms | 6.4× |
| Arrivals while worker is full | 2,048 arrivals | 235.93 ms | 1.26 ms | 187.0× |
| Complete H1 evaluation | 128 expected root sessions | 813.80 ms | 308.59 ms | 2.6× |
| Complete H2 evaluation | 32 expected root sessions | 2,907.04 ms | 455.16 ms | 6.4× |
| Complete H2 evaluation | 128 expected root sessions | 4,926.39 ms | 2,070.92 ms | 2.4× |

The optimized forecast-only trial took 5.42, 20.95, 83.31, and 334.81 ms at
128, 512, 2,048, and 8,192 active sessions respectively. It uses 128 Monte Carlo
draws, three horizons, homogeneous tool-running sessions with one progress event,
and no exogenous arrivals or fan-out. This is approximately linear over the
measured range; heterogeneous sessions, longer histories, and live HTTP traffic
need separate trials.

## What changed and why

- **Eviction:** compute free space once and walk eligible victims once. Repeated
  cache summation and candidate construction previously made a large eviction
  batch quadratic. Running, pinned, and explicitly retained contexts remain
  protected, including partial-eviction failure behavior.
- **Worker scheduling:** return before sorting when all batch slots are occupied.
  The next scheduling opportunity still applies the same priority ordering.
  The benchmark measures arrivals into a full worker, not complete queue drainage.
- **Trace processing:** convert records in batches and group contiguous sessions.
  Training and replay no longer construct and convert a DataFrame for every
  session. Canonical stable ordering is retained, including equal timestamps;
  the previous redundant per-session sort could reorder timestamp ties.
- **Conditional duration:** cache the sorted empirical distribution and invalidate
  it on refit. Unconditional sampling retains the original observation order,
  preserving its seeded behavior. The timing uses a warm cache.
- **Bootstrap:** align sessions once and resample integer positions. H2's mean
  statistics additionally prepare numeric values once through `MeanMetric`.
  Missing values remain excluded and arms share the same sampled sessions.
  Arbitrary metrics and duplicate session IDs retain the general DataFrame path.
  The isolated bootstrap benchmark deliberately uses a generic mean callable;
  it does not include the additional `MeanMetric` fast path used in H2.

Before optimization, per-session DataFrame conversion dominated H1 setup and
bootstrap dominated H2's profile. Afterward, H2 simulation and conditional
sampling account for most of its CPU work. Profiles are diagnostic: their
instrumented durations are not the wall times in the table.

## Complexity: what can exceed quadratic?

There is no useful single universal N. Let R be trace rows, S active sessions,
P progress events per session, H forecast horizons, D Monte Carlo draws, T ticks,
A policy arms, B bootstrap repetitions, K resident contexts, Q pending requests,
E future events, and F candidate GDP slots. Resource count is fixed today.
These are implementation bounds, not fitted scaling exponents.

| Operation | Time and qualifications |
| --- | --- |
| Trace construction / record traversal | Initial sort O(R log R); subsequent batched conversion O(R). Temporary record memory is bounded by a batch plus the largest session. |
| Replay state reconstruction | Up to O(T R): ticks inspect sessions and their rows. Progress history processing adds work with history length. |
| Forecast aggregation | Approximately O(S H D) per tick, plus progress processing, exogenous arrivals, and fan-out. The progress rate fallback can sort history, adding O(S P log P). |
| Empirical conditional duration | One O(R log R) sort per distribution, then O(log R + D) per query on the empirical path. Tail fallback still scans observations and performs up to 20 rejection rounds. |
| CRPS scoring | O(D log D) per scored distribution; uses sorted samples rather than a quadratic all-pairs distance matrix. |
| Paired mean bootstrap | O(A B N) for N aligned sessions, plus alignment/sorting. Working memory O(A N + A B), without allocating a B-by-N sample matrix. Arbitrary metrics add their own cost. |
| LRU batch eviction | Now O(K), previously O(K²) when evicting a substantial fraction. Custom victim ordering and oracle inspection add their own costs. |
| Priority worker queue | Full-worker scheduling returns immediately. C admission opportunities can still cost up to O(C Q log Q); draining a static backlog can retain quadratic scans. Python's adaptive sort can reduce sorting work on partially ordered queues. |
| Live hold queue | Linear selection/removal per release; releasing Q queued requests can total O(Q²). This is separate from the optimized simulator full-worker path. |
| Simulator event heap | O(log E) push/pop, but surrounding policy work can dominate. |
| Oracle future inspection | Sorts future events at O(E log E) per query. T ticks or C eviction queries multiply that cost. |
| GDP assignment | Worst-case O(S F D) feasibility work plus sorting sessions and preparing slot samples. Hold caps may reduce the slots actually searched. |

**Yes, some compositions exceed O(N²).** Oracle inspection becomes
O(N² log N) when both query count and future-event count grow with N. The queue
upper bound has the same form if both admission opportunities and queue depth
grow with N. GDP feasibility can be O(N³) if sessions, searched slots, and draw
count all grow with N; with fixed slots and draws it is linear in sessions,
apart from sorting. Bootstrap similarly multiplies independent arm, repetition,
and session dimensions. No exponential or factorial search was found in the
reviewed core paths.

Fixed configuration dimensions matter: doubling sessions at constant horizons
and draws is a different experiment from doubling all three. H2 also increases
congestion when its arrival count rises in a fixed window, so its elapsed time
is not a pure input-size complexity measurement.

## Remaining bottlenecks and next trials

The live JSONL bus rereads historical events, and board ticks reapply returned
events. Repeated history scans can accumulate O(T R), or quadratic work as the
log grows with tick count. Incremental consumption needs restart and replay
correctness tests. Registry start-history scans are another growth risk.
Neither behavior was changed in this study.

Board computation runs synchronously inside an async HTTP handler. Prediction
uses two board requests, and trace logging performs synchronous I/O. These can
hurt tail latency even when average model CPU time is acceptable. The next
serving trial should compare direct upstream, proxy-only, and proxy-plus-board
at growing active-session counts, queue depths, and event-log sizes. Record
proxy-added p95/p99, tick p95/p99, event-loop lag, snapshot age, CPU/RSS, and
prediction timeout/fallback frequency. Keep upstream model latency separate.

For offline work, profile long progress histories, tail-conditioned duration
sampling, oracle arms with large future heaps, and queues that actually drain.
Use independent evaluation processes for sweep parallelism only after budgeting
memory and numerical-library threads. This report does not claim those remaining
paths are optimized.

## Why Python now, and where Rust could help

ATFM coordinates an external inference engine; Python is not implementing its
GPU token-generation kernels. Python supports the project's NumPy/SciPy models,
pandas traces, and rapid experiment iteration. Many NumPy operations release
the interpreter lock, although Python loops and object operations do not inherit
that property ([NumPy documentation](https://numpy.org/doc/stable/reference/thread_safety.html)).
The measured improvements support retaining this ecosystem for the present core.

Rust becomes attractive at a measured boundary: a dominant CPU kernel exposed
to Python, or an admission/streaming proxy with a demonstrated throughput or
tail-latency shortfall. Retain Python fitting and evaluation around that boundary.
First fix redundant scans, network calls, and blocking work; a language rewrite
alone preserves their complexity. Threads are not a general remedy for Python
CPU loops, and moving board work out of the event loop requires explicit state
ownership ([asyncio guidance](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)).
See the [performance decision guide](../development/performance.md) for the
service boundaries and measurements required before a targeted rewrite.

## Method and reproducibility

Machine: Intel Core Ultra 9 185H, 22 logical CPUs, Linux x86-64; Python 3.12.13,
NumPy 2.5.3, pandas 3.0.6. Numerical-library thread counts were not pinned.
Baseline core revision: `b2f9247`; optimized sources are in the commit containing
this report. The benchmark driver was added during this work and is shared
between both versions. Final before/after timing runs were sequential, with
three repetitions after a warmup; cProfile runs were separate. Results are local
microbenchmarks and small synthetic evaluations, not statistical performance
confidence intervals. CPU scheduling, thermals, and dependency versions affect
reproduction.

Kernel fixture setup is excluded except construction intentionally inside the
eviction and queue workloads. Duration runs 256 queries of 256 draws. Bootstrap
uses three arms and 100 resamples. H1 includes fitting, replay, scoring, and
output for B0/M1/M2 over a 600-second arrival window, two horizons and 128 draws.
H2 includes simulation and output for three arms over a 300-second arrival
window, one seed, eight worker batch slots, and 300 bootstrap resamples.
Expected root-session counts are stochastic workload parameters, not exact
realized counts. Each timed H1/H2 call includes its result-file writes.

Reproduce using the [benchmark commands](../../experiments/README.md#cpu-scaling-trials).
For a baseline comparison, use a separate checkout at `b2f9247` and copy this
revision's benchmark module into its experiment package; keep the environment,
arguments, and hardware identical. Do not run the two versions concurrently.

The complete suite passed **271 tests, with 2 skips**. Added regression tests
compare eviction to the previous algorithm, preserve bootstrap pairing and RNG
state, cover duplicate IDs and missing values, verify cache invalidation and
full-worker priority behavior, and exercise batched trace boundaries and stable
ties. Benchmark return fingerprints matched before/after; H2 metrics and paired
intervals at both evaluated sizes agreed within 1e-13. These checks cover the
measured workloads, not every possible input or live-service interleaving.

Artifacts:

- [Comparison table](results/cpu-2026-09-27/comparison.csv),
  [environment and source hashes](results/cpu-2026-09-27/environment.json).
- Kernel timings [before](results/cpu-2026-09-27/kernels-before.csv) /
  [after](results/cpu-2026-09-27/kernels-after.csv).
- Complete evaluations [before](results/cpu-2026-09-27/e2e-before.csv) /
  [after](results/cpu-2026-09-27/e2e-after.csv).
- Queue timings [before](results/cpu-2026-09-27/queue-before.csv) /
  [after](results/cpu-2026-09-27/queue-after.csv),
  [forecast timings](results/cpu-2026-09-27/forecast.csv).
- H1 profiles [before](results/cpu-2026-09-27/h1-before-profile.txt) /
  [after](results/cpu-2026-09-27/h1-after-profile.txt); H2 profiles
  [before](results/cpu-2026-09-27/h2-before-profile.txt) /
  [after](results/cpu-2026-09-27/h2-after-profile.txt).
