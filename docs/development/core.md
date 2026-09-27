# Working on the ATFM core

This guide describes the executable forecasting and control paths. Start with
[the README](../../README.md) for installation and runnable commands; use the
[architecture spec](../superpowers/specs/2026-09-22-atfm-architecture-design.md)
for the research design and [results notes](../research/) for measured outcomes.
The implementation contracts below matter when changing a predictor or policy.

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
replayed. Offline H1 evaluation has its own trace-based observation schedule.

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
