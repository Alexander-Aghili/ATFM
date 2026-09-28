# Working on the ATFM core

This guide describes the executable forecasting and control paths. Start with
[the README](../../README.md) for installation and runnable commands; use the
[architecture spec](../superpowers/specs/2026-09-22-atfm-architecture-design.md)
for the research design and [results notes](../research/) for measured outcomes.
The implementation contracts below matter when changing a predictor or policy.

## Package boundary

The `atfm` runtime lives in `src/atfm/`. H1/H2 configuration, sweeps, and run
orchestration live in `experiments/src/atfm_experiments/`, a separate uv workspace
package depending on the core. Keep dependencies one-way. The root development
group installs the runners; `uv sync --no-dev --extra serve` omits them. Existing
`scripts/run_h1.py` and `scripts/run_h2sim.py` commands remain supported.

See the [experiment workspace guide](../../experiments/README.md) for packaging
and [performance assessment](performance.md) for Python/Rust tradeoffs.

## Where behavior belongs

| Responsibility | Implementation | Contract |
| --- | --- | --- |
| Event and forecast data | `src/atfm/schema/` | Shared types; no HTTP or simulator dependencies. |
| Session lifecycle | `board/live.py:SessionRegistry` | Events update owned mutable states; inactivity expires them. |
| Per-session models | `board/predictors/` | Draw relative return times, prompt sizes, and child counts. |
| Fleet aggregation | `board/forecaster.py` | Combine session draws, child demand, and exogenous arrivals. |
| Tick orchestration | `board/live.py:LiveBoard` | Feed arrivals once, expire inactive sessions, forecast. |
| Placement summaries | `board/resumption.py` | Convert finite return-time draws into q10/q50/q90. |
| Controllers | `control/` | Convert predictions and capacity into expiring directives. |
| Serving boundary | `board/service.py` | Ingest events, enforce prediction budgets, cache directives. |
| Simulation | `sim/` | Execute policies against workers and a simulated clock. |

`ForecastPolicy` composes a `LiveBoard`; it does not implement a second arrival
cursor or tool-duration lookup. The simulator still owns event draining, prompt
size observations, and removal of completed sessions. Those responsibilities
require simulator state and do not belong in the reusable board.

Keep new shared behavior at the lowest layer that has enough information.
Controllers consume schema types; they should not import the HTTP service or
simulator. Avoid a general utility module for unrelated calculations.

## Time, ownership, and randomness

Event times, tick times, and absolute return times must use one time base. Live
serving normally uses wall-clock seconds; simulations use seconds from their
origin. Predictors return **durations from `now`**, not absolute timestamps.

`LiveBoard.step(now, rng)` feeds starts in `[previous_tick, now)` into its
exogenous model. Its initial cursor is zero. A start exactly at the current tick
is counted on a later tick; repeating a tick does not count that start twice.
Call with nondecreasing times and ingest events before stepping. The cursor does
not provide late-event recovery: starts older than the previous tick are not
replayed. Consumed and late start records are discarded; future timestamps remain
pending, preserving arrival order without assuming timestamp order. Offline H1 evaluation has its own trace-based observation schedule.

`SessionRegistry.get(id)` returns an owned, mutable state or `None`. It neither
creates a session nor updates its activity timestamp. `session_ids()` returns a
stable tuple that remains safe to iterate while dropping sessions.
`states(now)` expires sessions only when inactivity is **greater than** the
configured expiry interval, then returns the remaining owned states. These are
single-owner APIs, not a thread-synchronization mechanism.

Pass an explicit NumPy generator through model and policy calls. Preserve state
iteration order and the number of random draws during a refactor: an extra draw
can change every subsequent event in a seeded experiment. Never construct a new
generator inside a sampling helper.

For row-oriented fitting and replay, use `TraceTable.session_records()`.
It converts fixed-size batches and retains canonical stable session/time order;
its temporary memory is bounded by a batch plus the largest session. Nested
event containers are shared and should be treated as read-only. Avoid reordering
`TraceTable.df` in place; construct a new table after changing its ordering.

## Forecast and placement contracts

A `ForecastSnapshot.samples[target][class]` array has shape
`(number_of_horizons, number_of_draws)`. Targets are `kv_blocks` and
`prefill_tokens`; classes are `interactive` and `background`. Horizons are in
seconds. Session aggregation and replay truth count the first returning call
per session per horizon. Child and exogenous demand are additional components.
For session forecasting, supply nonempty ascending horizons because child
sampling uses the final horizon as the largest one.

`resumption_quantiles` draws once per supplied state, in input order. It returns
`{session_id: (q10, q50, q90)}`, all in relative seconds. NaN and either infinity
are excluded before calculating quantiles; a session with no finite draws is
omitted. Consequently these are **conditional finite-return summaries**, not an
estimate of the probability that the session will return. Do not use them as a
replacement for the fleet's full Monte Carlo samples.

```python
import numpy as np
from atfm.board.resumption import resumption_quantiles

# predictor is a fitted SessionPredictor; registry contains ingested events.
resumptions = resumption_quantiles(
    predictor, registry.states(now), now, 64, np.random.default_rng(7)
)
```

The default propagates predictor errors so experiments fail visibly. The HTTP
service explicitly uses `skip_errors=True`, omitting only the failed session
and continuing with others. Touch and pin simulation policies use the strict
default. Oracle policies convert known absolute return times into identical
q10/q50/q90 values through `oracle_resumptions`.

KV eviction uses a median return time and assigns infinity when no finite draw
exists, so an unknown session is considered for eviction first. Touch and pin
planning omit such sessions. Keep this distinction when extending placement.

## Admission and side effects

`forecast_hold_until` is the common simulator admission calculation for forecast
and oracle-rule policies. Both use observed mean prompt size and configured
throughput to estimate service occupancy. Forecast policies estimate a running
request's completion as its start plus expected service time. Only the oracle
path can use the engine's true completion time. The policy remains responsible
for background-only eligibility, enabling holds, and recording the reason.

Do not merge `GdpLite` and `GdpPlanner` merely because both can hold work.
`GdpLite` tests one slot against upper demand quantiles and available capacity.
`GdpPlanner` assigns deferrable requests across slots and tracks commitments,
chance constraints, and tenant delay caps.

`POST /directives` runs controllers once per snapshot object and caches the
answer for later pollers. This matters because touch planning spends credit and
GDP planning resets assignment state. Repeated polling must not rerun planning.
Directives have expiry times; callers and actuators must respect them.

## Adding behavior

1. Implement a predictor using `SessionPredictor` in `board/predictors/base.py`.
   Fit only on the training split. Keep its sampling methods independent of the
   HTTP framework and simulator internals.
2. Reuse `LiveBoard` for event-driven ticks and `resumption_quantiles` for
   placement summaries. Keep a policy's action separate from its prediction
   source so forecast, oracle, and ablation variants can share the action.
3. Add a focused behavioral test for the new contract, including empty input,
   boundary times, and failure handling where relevant.
4. Run the relevant package tests, then the full suite. For simulation changes,
   verify the fixed-seed golden metrics without rewriting the expected file.
5. Update this guide if units, ownership, timing, error handling, or extension
   points change. Put measured claims in the research notes with their run data.

## Verification

```bash
uv sync --extra dev
uv run pytest -q tests/board tests/control tests/sim
uv run pytest -q
uv run pytest tests/board tests/control tests/sim \
  --cov=atfm.board --cov=atfm.control --cov=atfm.sim --cov-report=term-missing
```

The shared-contract tests are `tests/board/test_resumption.py`,
`tests/board/test_shared_live.py`, and `tests/sim/test_shared_admission.py`.
They cover finite-draw semantics, error isolation, random-stream preservation,
registry expiry, tick boundaries, live/simulated forecast equivalence, and the
oracle's distinct completion knowledge. Existing placement, service, and golden
simulator tests exercise the composed behavior.

Two Dynamo integration tests are opt-in with `ATFM_DYNAMO=1` and require the
Dynamo runtime. Ordinary suite success does not establish real-worker behavior.

## Documentation and comments

Document public contracts in docstrings: units, shapes, ownership, side effects,
failure behavior, and assumptions that callers cannot infer from a name.
Examples should show composition with real APIs, and commands must match
`pyproject.toml` and the scripts in the repository.

Use inline comments for a non-obvious invariant or a reason an apparently
simpler implementation is wrong, such as avoiding future-data leakage or
preserving sample correlation. Remove comments that restate the next statement.
Keep experiment history and design discussions in linked documentation rather
than embedding them in implementation comments. Do not remove a useful warning
about a scientific assumption just to reduce the comment count.

## Related references

- [Implementation status](../status.md) separates code paths from hardware evidence.
- [Operations](../operations.md) documents the HTTP and CLI boundaries.
- [Contributing](../../CONTRIBUTING.md) covers test and review expectations.
- [Research index](../research/README.md) links dated findings without treating
  historical run instructions as the current API.

## Live JSONL consumption

`JsonlBus.drain()` maintains a byte cursor per instance and consumes only complete
newline-terminated records appended since its last successful drain. Offline
`read_events()` still reads the whole file. A fresh instance starts at zero so a
fresh board can reconstruct its state; do not reuse a live reader with an empty
board. Repeated drains with no new records return an empty list.

Reads stop at the file size observed on opening. An incomplete last line is
retried, blank lines are skipped, and malformed complete records increment
`malformed` and are consumed. An I/O failure leaves the cursor unchanged. Missing
files return no events. Replacement (device/inode change) or observed truncation
resets the cursor. Writers must use append-only files: same-inode rewriting, or
copy-truncate followed by regrowth past the cursor between polls, is unsupported.
Rotation must coordinate writers/readers: an unread tail of a renamed file is not
recovered automatically. The lock coordinates one instance, not multiple writers.

This is a transport cursor, not transactional acknowledgement. Successful drain
advances before registry application; downstream failure is not automatically
retried, and restarting replays history. Durable recovery requires checkpointing
board state and log position together, plus explicit duplicate/sequence handling.
See [implementation decisions](control-scaling.md) for complexity and evidence.

## Tool sidecar boundaries

`sidecar/config.py` owns configuration and bounded gate waiting; it has no
mini-SWE-agent dependency. `SidecarConfig` remains importable from
`sidecar.minisweagent` for compatibility. `sidecar/adapters.py:SidecarExecutor`
owns command classification, turn advancement, launch, and result conversion.
The mini-SWE-agent mixin delegates there, then calls its harness-specific
`_check_finished` hook exactly once, including launch failures.

The per-line parser loops remain inline in `core.py` and `adapters.py`.
Extraction added about 5% in a tight parsing benchmark and was rejected to
preserve throughput. A progress match stops the chain even when its completed
count duplicates the previous match; data matches allow later parsers to run.
Each execution owns parser state. Live output retains per-line timestamps;
wrapped output retains completion timestamps. Preserve these contracts in both
loops when changing parsers.

Keep subprocess timeout/drain handling separate from wrapped-executor error
handling: only the subprocess owner can terminate a process group. Consolidating
parsing must not change which errors reach the caller or invent live progress
for an executor whose output is available only at completion.


Simulation policies share inert event callbacks through `_PolicyHooks`.
`ForecastPolicy` extends `ProxyRulesPolicy` for window and tier/index behavior,
overriding expected next-tool duration and forecast/hold handling. Placement
mixins still override the duration term to zero. Sharing these methods must not
change method-resolution order for placement, event draining, or RNG consumption.

Trace consumers share `schema.trace.is_missing_scalar`, which recognizes only
`None` and floating-point NaN. This deliberately preserves the existing scalar
contract; replacing it with a broader pandas missing-value check can change
behavior for arrays and nullable scalars. Forecast class/target order lives in
`schema.forecast`; the forecaster and calibration modules retain their previous
imports for compatibility. The order determines array axes and must remain
stable across prediction and calibration.

See the [refactor validation](../research/2026-09-28-core-refactor.md) for exact
output checks, benchmark limitations, and the parser extraction we rejected.
