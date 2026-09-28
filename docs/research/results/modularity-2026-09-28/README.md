# Function-boundary refactor

This pass applies a maximum of 20 physical lines per function, measured from
`def` through the last statement (including multiline signatures, docstrings,
comments, and blank lines). Helpers represent named operations; splitting must
not hide work in long expressions, introduce recursive orchestration, or change
algorithmic complexity. Existing mutable state, event ordering, RNG ordering,
public entry points, and failure behavior remain part of the contract.

## Runtime boundaries

- Proxy creation registers methods on a per-app runtime. Prediction fallback,
  session bookkeeping, admission, response forwarding, and stream cleanup are
  separate operations. State remains available through `app.state`.
- Board creation uses a per-app runtime; scraping and controller planning have
  independent boundaries. Directive results remain cached by snapshot identity.
- ControlLoop separates hold validation, bounded batch delivery, touch delivery,
  tier actions, and accounting. Batch acknowledgements are still validated.
- Subprocess capture separates launch, byte collection, process-group timeout
  cleanup, and completion. A shared parser event dispatcher handles both live
  subprocess output and completed executor output without changing event order.

Initial runtime stage: 217 tests passed across proxy, board, control, and sidecar;
one dependency deprecation warning. Broader validation follows subsequent stages.

## Forecast and simulation boundaries

Training-data collection, empirical fitting, calibration-grid search, forecast
aggregation, and evaluation summaries now have separate functions. Numeric
operations and RNG ordering are unchanged. Simulator orchestration dispatches
arrival, completion, tool, and tick events through explicit handlers; recording
results is separate from advancing sessions. JSONL reading keeps its original
cursor commit boundary and bounded-read behavior.

Synthetic program/session generation now uses an explicit depth-first stack.
Parents resume only after their children finish, preserving seeded draw order.
Program cloning also uses a stack. Both avoid Python recursion limits while
retaining O(depth) traversal storage. New tests exercise 2,000-level families.

Core-stage validation: the existing full suite passed (420 passed, five skipped,
six existing warnings). All maintained core functions meet the 20-line limit.

## Experiment and command-line boundaries

The load harness now has an explicit trial object for owned clients, logs, tasks,
and counters, with reporting in `load/report.py`. HTTP exchanges, tool delays,
monitoring, workload draining, and task cleanup are separate operations. Server
construction, inherited-socket startup, health polling, and profile persistence
are independently readable. H1 separates calibration, tick event cursors,
scoring, and reporting; H2 separates arm construction, paired runs, and summaries.
Benchmark cases and command-line parsing are small functions with unchanged
argument names and timing boundaries. Phoenix remains test-only infrastructure.

Experiment stage: 43 passed, three optional tests skipped, four existing numeric
warnings. Real HTTP load-harness tests cover shutdown, startup failure, worker
cancellation, and transport metrics. All experiment and script functions are at
most 20 lines. Reference algorithms in tests remain independent implementations;
only function boundaries change, with no reuse of optimized production logic.

## Final equivalence and performance evidence

Baseline: `f484a72`, immediately before this refactor. Timings are local CPU
trials, not GPU or production serving measurements. Simulator trials alternate
before/after order, pin the process to CPU 0, and pool nine measured repetitions
per policy after warmups. Every policy preserves its lifecycle/RNG hash.

| Policy | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
| native | 9.20 | 9.54 | +3.7% |
| rules | 10.27 | 10.63 | +3.5% |
| oracle | 10.31 | 10.49 | +1.7% |
| working_set | 21.97 | 22.23 | +1.2% |
| forecast | 68.47 | 68.29 | -0.3% |
| oracle_rule | 39.09 | 39.42 | +0.8% |
| forecast_kv | 36.17 | 36.32 | +0.4% |
| oracle_kv | 17.58 | 17.69 | +0.6% |
| forecast_touch | 36.00 | 36.06 | +0.2% |
| oracle_touch | 11.48 | 11.64 | +1.4% |

The small positive timing changes are retained rather than described as zero
overhead. Forecast-heavy paths differ by less than 1%; the largest simulator
change is 3.7% (about 0.34 ms for the complete 360-call native trial).

### GDP at 100,000 sessions

300 slots, 128 draws, three timed repetitions after warmup, CPU 0. Serialization
and output hashing are outside the timed section. All directive hashes match.
The first refactor exposed extra boundary-lookup overhead; the final version
caches only hold-window bounds within a plan, retaining exact comparisons.

| Regime | Before (s) | Final (s) | Change |
| --- | ---: | ---: | ---: |
| open | 0.507 | 0.527 | +3.9% |
| saturated | 0.612 | 0.583 | -4.7% |
| mixed | 0.908 | 0.877 | -3.5% |

The raw `gdp-after` directory records the intermediate regression; `gdp-cached`
is the final algorithm. Results vary across runs: the final open-case minimum
and maximum are 0.471–0.532 s, so these medians are not proof of a speedup.
No asymptotic degradation was introduced. Cached boundary work is
O(U log S + N), U <= S, with O(U) additional storage per plan.

Additional checks:

- Five seeded generation trials match exactly, including trace rows, full program
  families, and final RNG state (`generation-before/after.json`).
- Sidecar trials preserve byte/event hashes for 10,002 events. Wrapped median:
  30.03 -> 30.81 ms; subprocess median: 59.81 -> 50.50 ms. Subprocess timings
  are noisy and should not be treated as a claimed speedup.
- Seventeen rendered plots are pixel-identical; SVG charts and result tables
  are text-identical to their pre-refactor versions. The background PDF and
  standalone bundle rebuild successfully.
- The function check uses the AST span, including async and nested definitions.
  It covers every tracked Python file without reference-fixture exemptions.
- Deep-family tests exercise 2,000 levels without recursion. A focused GDP test
  verifies the slot-bound cache is discarded when the next plan changes its cap.

Reproduce simulator and sidecar comparisons with the benchmark scripts under
`../core-refactor-2026-09-28/`, selecting each checkout using `PYTHONPATH`.
`generation_equivalence.py` provides the seeded-generation comparison.
GDP commands and source hashes are in each run's `environment.json`.

Smaller functions add named helper definitions, so this pass increases total
source lines. It favors readable boundaries over packing statements to reduce
line count. Existing independent paper edits are preserved separately.

Final validation: **425 passed, five skipped, six existing warnings**. The skips
remain optional integration/profiling checks; they are not claimed as executed.
The AST size audit and undefined-name checks pass. Both LaTeX manuscripts compile
without overfull boxes, unresolved references, missing glyphs, or duplicate labels.
