# Large-session control paths: exact GDP thresholds and batch admission

CPU overhead is not automatically negligible compared with data movement.
A native FIFO or rules-only policy does less work because it omits forecasting
and risk-aware slot assignment. The relevant deployment question is whether
that extra control work fits its tick and request-latency budgets and earns
better serving outcomes. These trials compare two implementations of the same
control semantics, not the serving benefit of ATFM versus FIFO, and do not
measure network or GPU transfers.

## Results

Local median wall times, one warmup and three measured runs, fixed synthetic
inputs. Fixture construction is outside GDP timing; planning, interpolation,
threshold preparation, and directive construction are inside. Serialization is
excluded. Admission timing includes constructing entries and releasing one
batch, without HTTP or an asyncio event loop.

| Workload | Previous | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| GDP saturated: 1,024 sessions, 300 slots, 128 draws | 1,273.68 ms | 14.87 ms | 85.7× |
| GDP saturated: 1,024 sessions, 300 slots, 1,024 draws | 1,631.11 ms | 25.10 ms | 65.0× |
| GDP open: 1,024 sessions, 300 slots, 1,024 draws | 23.12 ms | 16.79 ms | 1.4× |
| GDP open: 128 sessions, 300 slots, 1,024 draws | 11.57 ms | 14.33 ms | 0.8× |
| Release 2,048 eligible admissions in one batch | 394.61 ms | 2.11 ms | 186.8× |

The small open GDP case regresses by about 2.8 ms: preparing thresholds for all
slots costs more than the original early-exit scan there. The optimization is
most valuable under congestion. Both extremes are retained in the benchmark;
the speedups should not be generalized to every traffic mix.

Larger optimized-only GDP trials:

| Sessions | Slots | Draws | Open | Saturated |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 300 | 128 | 33.72 ms | 113.72 ms |
| 8,192 | 300 | 1,024 | 42.71 ms | 132.60 ms |
| 8,192 | 3,000 | 128 | 52.96 ms | 239.53 ms |
| 8,192 | 3,000 | 1,024 | 244.04 ms | 429.05 ms |

At a hypothetical five-second planning interval, the final 429 ms measurement
consumes about 8.6% of one core's wall-time budget, before forecasting and other
work. If synchronous inside an async handler, it can also block that event loop
for the duration. At a one-second planning interval, it is about 43%. Those are
budget calculations, not measured production utilization or latency percentiles.
The model here has one horizon per slot, deliberately stressing preparation too.

## What is essential, and what was avoidable?

GDP must account for uncertain interactive demand and find feasible release slots
in session order. It does not need to recompute every sample comparison for every
session. For D samples, let k be the smallest count with k / D >= 1 - epsilon.
The original predicate is exactly equivalent, for finite slot samples and
0 <= epsilon < 1, to:

```text
k-th smallest interactive sample + committed load + new demand <= capacity
```

The implementation finds that order statistic once per resource and slot using
`numpy.partition`. It preserves the original addition order and determines k
using the same floating-point probability comparison as the Boolean mean.
This is an exact empirical check, not an interpolated percentile approximation
or a replacement of samples with their mean. Nonfinite samples and unusual
epsilon values use the previous general scan.

The first candidate slot has a scalar fast path; remaining candidates are
checked together with NumPy. Each session still commits its load before the
next session is considered, preserving greedy ration-by-schedule, both resource
constraints, hold caps, tenant overrides, and directive order. Thresholds are
local to a planning call, so changed snapshots cannot reuse stale values.

For S sessions, F slots, D draws, and a fixed number of resources, the repeated
feasibility work changes from **O(S F D)** to **O(F D + S F)**, plus session
sorting and the existing interpolation preparation. With all three dimensions
growing together, this component becomes quadratic rather than cubic. Slot
search remains linear in F in the worst case. Prepared thresholds use O(F)
space; slot samples and partition scratch still require O(F D).

Admission previously scanned and removed an entry repeatedly for every available
window slot. It now selects the best W eligible entries once with a bounded heap
and removes them in a single pass. It preserves stable ties, promotion, overflow
FIFO behavior, holds, and release timestamps. For Q queued entries, selection
cost is O(Q log(W + 1)), with O(Q) filtering/storage; releasing all Q entries is
O(Q log Q) rather than quadratic. When only one request completes at a time,
each tick still scans Q entries, so draining a backlog that way can remain
quadratic overall. This change does not claim to solve that separate workload.

## What the numbers do not establish

These are standalone component measurements. The full `GdpPlanner` is available
to the deployed board; H2 forecast arms use `GdpLite`, so these GDP improvements
must not be presented as equivalent H2 speedups. Batch admission helps when many
slots become available together; it is not a proxy requests-per-second result.

We have not measured an apples-to-apples native-versus-ATFM proxy load test or
CPU time versus actual network/KV-transfer time. Transfer cost depends on bytes,
bandwidth, placement, and overlap; CPU cost depends on sessions, samples, slots,
and tick frequency. A comparison to total generation time would also hide
request-tail delays from synchronous control work.

The next priorities remain incremental event ingestion, avoiding synchronous
board work in the HTTP event loop, combining redundant prediction requests,
and batching directive delivery. Measure proxy-added p95/p99, snapshot age,
event-loop lag, CPU/RSS, and fallback rate while varying backlog and event-log
age. For still-larger planners, bounded slot resolution, incremental forecasts,
and indexed feasible-slot search are candidates, but require workload evidence
and semantic review. Lowering sample counts or coarsening slots changes the
accuracy/control tradeoff; it is not a free implementation optimization.

These results continue to favor Python for models and evaluation. Rust may help
an isolated CPU kernel or streaming proxy when measured budgets demand it, but
the large gains here came from removing repeated work rather than changing
language.

## Reproduction and evidence

```bash
uv run python -m atfm_experiments.benchmark_gdp --out runs/gdp
uv run python -m atfm_experiments.benchmark_gdp \
  --sessions 8192 --slots 300 3000 --draws 128 1024 --out runs/gdp-large
uv run python -m atfm_experiments.benchmark_cpu \
  --cases admission --sizes 128 512 2048 --out runs/admission
```

GDP baseline is the frozen implementation in
`tests/control/reference/gdp_original.py`; commit `3c8103d` contains that
implementation in the core and the GDP benchmark. Admission baseline is
`61582eb`; use the later benchmark driver with its admission case in that
checkout. Optimized GDP and admission are committed separately in `61582eb`
and `ebeaadf`. Use isolated checkouts and the same environment; run versions
sequentially, with no concurrent benchmarks. Hardware matches the
[initial CPU study](2026-09-27-cpu-scaling.md): Intel Core Ultra 9 185H, Python
3.12.13, NumPy 2.5.3. Thread counts were not pinned. Three repetitions describe
local timing variability, not statistical confidence intervals.

The complete test suite passed 294 tests, with two skips and five existing
empty-metric warnings.

All 16 GDP before/after workload fingerprints and all three admission
fingerprints match. Randomized regression tests compare complete GDP directives,
assignment maps, and tenant delays against the frozen implementation across
sample counts and epsilon values. Additional checks cover floating-point
probability boundaries, capacity boundaries, priority ties, promotion, held
entries, overflow, and different available-window sizes.

A separate [profile of the large saturated case](results/gdp-2026-09-27/gdp-large-profile.txt)
places most remaining CPU time in slot search and slot-demand interpolation.
Profiling was separate from the timing trials.

Raw data and provenance:

- [GDP comparison](results/gdp-2026-09-27/gdp-comparison.csv),
  [before](results/gdp-2026-09-27/gdp-before.csv),
  [after](results/gdp-2026-09-27/gdp-after.csv),
  [large trials](results/gdp-2026-09-27/gdp-large.csv).
- [Admission comparison](results/gdp-2026-09-27/admission-comparison.csv),
  [before](results/gdp-2026-09-27/admission-before.csv),
  [after](results/gdp-2026-09-27/admission-after.csv).
- [GDP environment](results/gdp-2026-09-27/gdp-after-environment.json),
  [admission environment and source hashes](results/gdp-2026-09-27/admission-after-environment.json).
