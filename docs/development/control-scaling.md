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
