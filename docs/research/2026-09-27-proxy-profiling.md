# Proxy CPU profiling and redundant import searches

## Why this experiment

The [control-attribution study](2026-09-27-control-attribution.md) found roughly
one busy proxy CPU core with a fast fake worker. That identifies a process, not a
hot function. We profiled the same 1,024-session burst workload to choose one
small, measurable change before modifying admission or prediction architecture.

The first two cProfile runs are retained as exploratory evidence. A minimal
reproduction on this Python 3.12.13 build showed that cProfile also captured a
worker thread. Its interleaved concurrent timing attribution was unsuitable for
our original assumption of an event-loop-only report. Those numbers are not used
to claim CPU percentages or speedups.

The harness now uses [Yappi's per-thread CPU clock and context filtering](https://github.com/sumerc/yappi/blob/master/doc/api.md).
This separates serving-thread CPU from synchronous prediction workers and excludes
I/O wait. Profiling is an optional experiment dependency, not a core dependency.
Profiles begin in application lifespan and stop before shutdown cleanup.
Merged pstats, per-thread CSV/text, and profiler/thread metadata are retained.
A completion metadata file is written after export; the runner rejects missing
artifacts. Tests verify thread separation and export even when the profiled scope
raises.

## What the profiles found

Two Yappi 1.7.6 runs used the faster-worker configuration: 1,024 sessions,
three turns, two-second arrival window, burst-aligned tool returns, 64 worker
slots, 5 ms service, and proxy window 64. Prediction budget remains 50 ms.
Profiling increases overhead substantially, so only these instrumented runs use
a 60-second request timeout and 120-second drain timeout. All calls completed.
Use unprofiled trials below for latency comparisons.

| Serving-thread operation | Calls per run | Cumulative CPU, control on (s) | Control off (s) |
| --- | ---: | ---: | ---: |
| HTTPcore async-runtime detection | 18,433 | 4.095 | 4.435 |
| HTTPcore connection assignment | 6,144 | 3.159 | 3.264 |
| Queue peer-index scan | 3,072 | 1.318 | 1.333 |
| JSONL event publication | 10,240 | 0.608 | 0.578 |
| Priority quartile calculation | 3,072 | 0.166 | 0.167 |

Total profiled serving-thread CPU was 26.72–26.77 seconds. These cumulative
function timings include children; arbitrary rows in a full profile overlap.
Instrumentation changes scheduling, backlog size, and prediction coverage, so
these values are not estimates of uninstrumented milliseconds per request.
The profile shows substantial framework and transport work in addition to ATFM
queue work. It does not support blaming the entire cost on the remaining queue
scan or treating JSONL publication as the largest cost.

## First change: supply the optional runtime detector

In the tested HTTPcore 1.0.9/AnyIO 4.15.1 environment, `sniffio` was absent.
[HTTPcore's runtime detector](https://github.com/encode/httpcore/blob/1.0.9/httpcore/_synchronization.py)
tries importing it on each call and falls back to asyncio on ImportError.
The missing-module search repeats; a successfully imported module is cached.

A focused test inside a running asyncio loop makes 100 warmup checks followed by
seven repetitions of 10,000 checks. Median elapsed time was **255.11 ms without
sniffio versus 2.45 ms with sniffio 1.3.1** (about 104 times faster for this
specific helper, not the whole service). Both variants detect asyncio.

Supplying the dependency removes repeated import searches without changing
ATFM's queue, ranking, forecast, deadlines, or fallback policy. Runtime detection
remains linear in the number of checks; this is a constant-factor reduction for
a fixed import path, not a better complexity class in session count. The
per-release peer-index scan is still O(Q), and draining a growing backlog can
still accumulate quadratic ranking work.

## Unprofiled paired experiment

Two isolated environments have identical installed package versions except for
sniffio 1.3.1 in the candidate environment. The archive contains full package
inventories. Both environments run the same source revision and workload;
the candidate dependency is available to the generator and all service processes,
not only the proxy.

Three seeds (7–9) each run control on/off with and without the candidate.
Dependency order and control order alternate across seeds. Processes are fresh;
cases run sequentially; the descriptor limit is 8,192. The original 20-second
request timeout and 30-second drain timeout apply. CPU affinity, frequency, and
background host activity remain uncontrolled. Each seed is only one finite
trial per condition, not a hardware-capacity estimate.

All **36,864 / 36,864** requests succeeded, with no session, control, probe,
publication, or drain-deadline failures.

| Control | Dependency | Median of run p95s (s) | Median proxy CPU (s) | Median requests/s | Pooled prediction timeouts |
| --- | --- | ---: | ---: | ---: | ---: |
| On | Absent | 3.495 | 9.046 | 346.5 | 55.62% |
| On | Present | 2.437 | 6.892 | 434.6 | 63.31% |
| Off | Absent | 4.035 | 9.772 | 306.8 | 38.43% |
| Off | Present | 3.007 | 7.825 | 402.7 | 47.81% |

The median-of-run-p95 reduction is 30.3% with control and 25.5% without;
median proxy CPU falls 23.8% and 19.9%, respectively. Every individual paired
seed has lower p95 and lower proxy CPU with the dependency. These are descriptive
results from three pairs per mode, not confidence intervals or a universal
speedup. End-to-end changes include effects on the generator, request timing,
prediction completions/cancellations, and control work; they cannot all be
assigned directly to the isolated helper's saved CPU.

**Prediction coverage worsens.** More callers fall back despite the lower CPU
and request latency. Faster progress through other parts of the system can
concentrate work at the four-thread prediction path, but that mechanism is a
hypothesis, not an isolated result of this comparison. Do not describe the change
as improved scheduling quality. The next change should bound prediction work
and measure accepted, queued, expired, and completed work, preserving fail-open
behavior while making the remaining bottleneck observable.

The serving extra now explicitly includes `sniffio>=1.3.1`, locked to 1.3.1
for these results. No transport internals are patched and no scheduler algorithm
is changed. Base/core-only installs do not gain this dependency; Yappi remains
in the experiments-only `profiling` extra. Revisit the explicit detector
dependency if a future HTTPcore version changes its runtime detection path.

## Confirmation, reproduction, and evidence

Follow-up Yappi profiles with sniffio reduced HTTPcore runtime detection from
4.095–4.435 s to **0.067–0.068 s** for the same 18,433 checks. Connection
assignment remained 2.36–2.81 s and peer-index scanning 1.28–1.29 s under
instrumentation. Those remain candidates after prediction deadline handling;
event publication measured about 0.568 s. The explicit detector dependency is
the only serving change in this step.

The [archive](results/proxy-profiling-2026-09-27/) contains 18 cases:
two exploratory cProfile runs, two CPU profiles before and two after, and twelve
unprofiled paired trials. Its [comparison CSV](results/proxy-profiling-2026-09-27/comparison.csv)
labels instrumented cases explicitly. Do not pool their latency with unprofiled
trials. All 384 retained files pass manifest length/SHA-256 checks after
decompression. Raw traces, server logs, shutdown records, profiles, package
inventories, focused-test results, and exact scripts are retained.

Exploratory profiles used revision `166afc4`; initial Yappi profiles used
`726844f`; paired trials and follow-up profiles used `c11cfab`.
The serving dependency change is committed as `c720cfa`. Source hashes in all
twelve paired cases match. Environment differences are recorded separately:
a revision alone does not identify installed optional packages.

For a new profiled case, create a configuration like the archived
[CPU-profile configuration](results/proxy-profiling-2026-09-27/reproduction/yappi.json):

```bash
uv sync --frozen --all-packages --extra dev --extra serve --extra profiling
(ulimit -Sn 8192
 uv run --no-sync python -m atfm_experiments.load.attribution \
   --config profile.json --sessions 1024 --repeats 1 --out runs/new-proxy-profile)
```

For dependency comparisons, use two disposable environments. The current serving
extra installs sniffio; remove only sniffio from the baseline environment with
`uv pip uninstall --python <baseline-python> sniffio`, and keep version 1.3.1
in the candidate. Verify full package inventories differ only in that package.
Run through the environments' Python executables directly so automatic syncing
does not restore the baseline dependency. The archived comparison and detection
scripts record the exact original workload and paths; adjust their environment
and output paths to fresh locations before reuse. The archival script is a
historical record with a fixed destination, not a safe in-place rerun command.

Final validation: **409 passed, 6 skipped**, including real HTTP profile export
and separate-thread attribution on exceptional exit. Seven warnings remain from
existing dependency deprecations and empty-slice evaluation fixtures.
