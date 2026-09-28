# Control-path scaling decisions

Implement changes independently: incremental ingestion, batched delivery,
indexed admission, then planner/forecast kernels. The
[pre-change audit](../research/2026-09-27-current-bottlenecks.md) is historical
evidence, not a description of the updated runtime. Keep Python while removing
avoidable repeated work; these component trials do not establish GPU capacity.

## 1. Incremental ingestion

The old live JSONL drain reused the full-file offline reader. The paper provides
no rationale for replaying that file on every live tick; it does distinguish
transport cursors from state checkpoints. Preserve offline replay while giving
each live reader its own byte offset and file identity. This prevents repeated
registry application in a running consumer. The implementation deliberately does
not claim crash-safe exactly-once delivery.

For E historical events and delta new events of bounded size, normal drain parsing
changes from O(E) to O(delta); an unchanged file needs O(1) metadata work. Initial
reconstruction remains O(E). A snapshot size bounds each read against concurrent
appends; temporary memory is proportional to newly parsed records, including the
full initial replay. Extremely large recovery logs still need bounded batches or
checkpoints. Registry arrival bookkeeping drops consumed/late records each tick,
retains future records, and preserves existing [previous tick, now) semantics.
It scans pending starts, not all historical starts; future-dated input can still
grow that pending set. No clock-order assumption is imposed on event arrival.

Tests cover independent readers, reconstruction, repeated drains, partial UTF-8
and partial lines, malformed records, replacement, observed truncation, I/O retry,
and repeated HTTP ticks without duplicating starts/progress. The exact transport
and recovery contract is in [core.md](core.md#live-jsonl-consumption).

Reproduce the unchanged-log trial:

```bash
uv run python -m atfm_experiments.profile_bottlenecks \
  --cases jsonl --out runs/incremental-ingestion
```

Historical benchmark artifacts retain their source hashes and original semantics.
The updated runner asserts that timed unchanged drains return zero events.

The unchanged million-event fixture decreased from 4.297 s to 0.00000633 s median
on this workstation (one warmup, three measurements). This compares full reparsing
with an empty incremental read, not processing a million new events. Warm storage,
OS metadata caching, and uncontrolled background activity limit interpretation.
[Timings and source hashes](../research/results/control-scaling-2026-09-27/ingestion/)
are retained; 54 bus/board tests passed before the first implementation commit.

## 2. Batched holds and combined predictions

The controller sends up to 1,024 holds per `POST /directives/batch`; deployments
can lower `ControlLoop.hold_batch_size`. The proxy validates every item before
mutation, skips expired holds at receipt time, installs the updates synchronously,
and ticks once per nonempty applied batch. It acknowledges applied/expired counts;
non-200 or invalid acknowledgements increment controller errors rather than
claiming delivery. Expiry is checked again at receipt because a slow controller
may have outlived the decision. Empty batches do not tick. Duplicate session IDs
within a batch use last nonexpired update wins; counts count updates, not unique
sessions. Nonfinite timestamps are rejected. A batch is atomic with respect to
other handlers on the same event loop, not a durable transaction.

For D holds and Q queued requests, the list scheduler's repeated scan term drops
from O(D Q) to O(D + ceil(D/B) Q), plus release selection, for chunk size B. HTTP
round trips drop from D to ceil(D/B); payload work remains O(D). Chunks are applied
independently. No automatic retry, cross-controller sequencing, delta protocol, or
whole-plan atomicity is claimed. Those require epochs, ordered versions, expiry
renewal, and resynchronization; introducing them is not necessary to remove the
existing per-directive overhead. Deploy the batch-capable proxy before the new
controller. Legacy `/directives` callers remain supported.

Hold policy is unchanged: directives affect the session's next submission, not
an already waiting request. Deadline expiry prevents new submissions from using a
stale directive; it does not revoke the hold already copied onto a queued entry.

`BoardClient.expected_times()` retrieves both predictions in one HTTP request.
The proxy uses it when supported and retains compatibility with injected
predictors exposing the two legacy methods. Failed, malformed, nonfinite,
negative, or over-budget combined results trigger local service-time fallback.
The four-thread pool and cancellation limitation still exist; this change removes
one round trip rather than claiming to isolate control CPU from the event loop.

Tests cover payload validation before mutation, bounded chunks, one tick per
applied batch, expiry, legacy endpoint parity, unchanged waiting requests,
acknowledgement failures, and one board request per proxy request with fallback.

## 3. Persistent admission indexes

`HoldQueue` owns insertion-sequenced request records (not session IDs, since a
session can have multiple waiting requests). A ready heap orders by descending
tier/index then ascending arrival/sequence; a second ready heap provides FCFS in
overflow mode. Delayed and promotion min-heaps drive eligibility and tier changes.
A request-identity map makes cancellation a dictionary operation. The original
scan implementation is retained only as a test oracle.

Completion pops a ready entry rather than scanning the backlog. Promotions bump
an entry version; stale priority records and cancelled/released entries are
ignored when popped. Heap storage is rebuilt with linear-time heapify when the
combined record count exceeds eight times the active backlog plus 64. This bounds
stale-record memory to O(Q + 1). Rebuilds occasionally cost O(Q), so operation
bounds are amortized: insertion, scheduling updates, and release O(log Q), and
individual drainage O(Q log Q), excluding service time and diagnostics. A burst
of K due timers still costs O(K log Q). Overflow entry drops existing holds and
rebuilds in O(Q); subsequent FCFS selection uses the arrival heap. Hysteresis and
priority restoration match the existing policy. Timestamps must be nondecreasing
and entry scheduling fields are queue-owned after submission.

The wake-up timer covers both hold expiry and tier promotion, including when
there is no incoming traffic, and is cancelled when no deadline remains. Priority
ties retain insertion order. Hold directives still govern future submissions;
there is no unnecessary session-to-waiting-request index because queued requests
are not retimed by directives. `pending` is an O(Q) diagnostic snapshot, with a
bulk setter for benchmark fixtures; hot loops use the O(1) `queued` count.

Remaining scans are explicit: `stats()` counts held entries and `tier_indices()`
collects peer indices for priority-bucket ranking, both O(Q). The HTTP request
path therefore still has a linear rank query even though admission completion no
longer scans the queue. An order-statistics tree could address this separately;
it is not necessary for preserving admission priority semantics.

Randomized differential tests compare 8,000 operations against the previous
scheduler, including duplicate session IDs, hold caps/expiry, promotions,
window changes, cancellation, completion, overflow and ties. Separate tests
exercise heap compaction and actual asyncio timers without external traffic.

The repeated single-slot trial measured 1.63 ms for 1,000, 7.03 ms for 4,000,
and 356 ms for 100,000 queued requests. The first two pre-change medians were
103 ms and 1,974 ms. Setup, HTTP, service time, and priority-bucket computation are
excluded; one warmup and three repetitions use the same local synthetic fixture.
[Timings/source hashes](../research/results/control-scaling-2026-09-27/admission/)
are retained. The targeted proxy/control/deployment suite passed 84 tests.

## 4. Planner pruning and empirical sampling

GDP first tests the requested slot. On failure it finds the exclusive end of the
permitted hold interval by binary search using the original floating-point delay
comparison. The vector fallback allocates only that eligible interval: O(S W)
search rather than O(S F), plus O(S log F) bound finding, where S is sessions,
F all slots, and W eligible slots. Threshold construction and session sorting
remain separate costs. This bound preserves behavior at fractional hold limits.

For at least 64 slots and a hold allowance spanning at least 64 slots, a segment
tree is enabled lazily on the first nontrivial search. Leaves contain the exact
`threshold + committed` quantity per resource; internal nodes contain per-resource
minima. If a minimum plus the session's need exceeds capacity, the entire interval
is impossible. Otherwise search visits the left interval first and checks leaves.
Minima for different resources can occur at different slots: passing an internal
node is necessary, not sufficient, for a feasible assignment. Worst-case search
is still O(F), not a universal O(log F). The index uses O(resources * F) memory and
O(resources * F) construction, with O(resources * log F) commitment updates once
built. Open cases that always fit the first slot never build it. Short horizons
and short hold windows retain the bounded vector path. Nonfinite session needs
disable the index; unsupported chance-threshold inputs retain the sample scan.

The numerical comparison remains `(threshold + committed) + need <= capacity`.
We do not subtract demand from capacity, which would change floating-point edge
cases. Tenant-cap overrides update the actual chosen leaf. Parity tests compare
full directives and assignments against the frozen sample-scan planner, including
fractional caps, nonfinite fallback, non-power-of-two horizons, and resources
whose minima disagree.

Forecasting now shares `empirical_draw`, a narrow primitive for uniform draws
with replacement from a one-dimensional array. It indexes the empirical array
with `Generator.integers` instead of invoking `Generator.choice`'s generic
shape/axis machinery thousands of times. This reduces constants without changing
O(S M H) aggregation (sessions, draws, horizons), model semantics, dependence,
or interleaving of random draws. Exact sample and subsequent RNG-stream parity
are tested for four NumPy generators and for a full forecast fixture.

Cross-session batching and reuse of forecast draws for directive quantiles remain
future work: they would change random-stream ordering or the existing 128/64-draw
protocol. The current optimization deliberately preserves seeded experiments.
Any such later change needs explicit statistical and reproducibility validation.
Separating mutable control-state ownership from HTTP handling also remains open;
these kernels do not remove the event-loop blocking hazard.

Repeated local measurements (one warmup, three repetitions, setup excluded):

| Component and fixture | Previous median | Updated median |
| --- | ---: | ---: |
| GDP saturated: 100,000 sessions, 10,000 slots, 1,024 draws | 7.322 s | 1.691 s |
| GDP mixed: same dimensions, heterogeneous sessions | 4.706 s | 1.874 s |
| GDP saturated: one million sessions, 300 slots, 128 draws | 15.910 s | 7.477 s |
| Forecast: 100,000 sessions, 3 horizons, 128 draws | 4.120 s | 3.785 s |

All four output hashes match the earlier measurements exactly. The separate
8,192-session GDP checks measured 40.6 ms open and 76.3 ms mixed; they test additional
regimes, not a matched historical speedup. Forecast at 8,192 sessions measured
304 ms. Full suite: 384 passed, 6 optional-dependency tests skipped. Warnings are
existing dependency deprecations and empty-slice evaluation fixtures. No live
worker or concurrent network capacity conclusion follows from these trials.

[Raw results, profiles, environments and source hashes](../research/results/control-scaling-2026-09-27/)
include each fixture. Reproduce sequentially to avoid CPU contention:

```bash
uv run python -m atfm_experiments.benchmark_gdp --sessions 100000 \\
  --slots 10000 --draws 1024 --regimes saturated mixed --repeats 3 --out runs/gdp-index
uv run python -m atfm_experiments.benchmark_gdp --sessions 1000000 \\
  --slots 300 --draws 128 --regimes saturated --repeats 3 --out runs/gdp-index-million
uv run python -m atfm_experiments.benchmark_cpu --cases forecast \\
  --sizes 8192 100000 --repeats 3 --profile --out runs/forecast-sampling
uv run python -m atfm_experiments.profile_bottlenecks --cases queue --out runs/queue-index
```

The queue fixture measures individual completion scheduling with setup excluded.
Persistent indexes add construction/memory overhead; this result is not a claim
that bulk loading and releasing an entire backlog in one call is faster than the
previous batch-selection implementation. Historical measurements were taken at
separate times on the same workstation, with uncontrolled affinity and background
activity. Component times must not be added into an end-to-end cycle estimate.
