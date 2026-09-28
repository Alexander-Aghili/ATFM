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
