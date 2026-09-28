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
