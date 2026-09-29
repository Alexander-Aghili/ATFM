# Bounded upstream pool sharding

Date: 28 September 2026. Baseline runtime: `bb2a561`; measured candidate: `0425a58`; final automatic selection: `321bf59`.
This follows the [peer-rank and rejected pool study](2026-09-28-proxy-ranking.md).
The new transport preserves connection reuse without putting every request into
one large pool. It improves all six full-proxy latency/CPU pairs in this study,
with lower timely prediction use and some light-load latency regressions.

## Why another transport experiment

The previous profile found substantial HTTPcore connection-assignment CPU. A
single larger keep-alive pool opened fewer TCP connections but made performance
worse. To distinguish transport behavior from board and admission effects, this
study first runs only a load client and a fake worker over real localhost HTTP.
No ATFM proxy, forecasting, tools or control loop participates in that fixture.

An exploratory fixed-lane probe compared the existing default, a single pool
retaining 64 connections, and 8/16/64 independent clients with lane affinity.
At 64 concurrent calls and 3,072 total requests, the default used about 2.24 CPU
seconds, the larger shared pool 11.20, and the 16-client variant 1.39. This was
hypothesis generation, not the retained implementation: fixed lane affinity does
not balance arbitrary request or streaming lifetimes. Its script and results are
archived separately and lack the formal fixture's source-hash provenance.

## Exact implementation and ownership

`proxy/transport.py` implements the public HTTPX asynchronous transport/stream
interfaces. For P shards and connection budget C=100, it gives shard i

`capacity[i] = floor(C/P) + (i < C mod P)`.

The sum is exactly C. The automatic policy selects P=16 for initial admission windows above 20,
producing four pools of seven connections and twelve pools of six. Windows of
20 or fewer retain stock HTTPX because of the measured small-case latency cost. Every pool may retain its capacity as idle connections;
the combined idle cap therefore rises from stock HTTPX's 20 to 100. Idle expiry
remains five seconds. No eager connections are opened. One shared TLS context
retains HTTPX's certificate environment settings, certificate verification and
hostname checking, avoiding a separate trust-store load for every pool.

For each request, selection minimizes `active[i] / capacity[i]`. The synchronous
selection and counter increment occur before awaiting dispatch. `active` counts
unfinished response lifetimes, including requests awaiting headers or a connection
and streams whose bodies remain open. It is not a count of TCP sockets. The
response wrapper forwards bytes unchanged and returns occupancy exactly once on
close, including close errors. Pre-header failures and cancellation return it
directly. As with ordinary HTTPX streaming, callers must close responses.

The outer HTTPX client still owns request construction, cookies, redirects and
timeouts. Redirects remain disabled by default. Touches share the upstream client.
The proxy owns and closes its client/pools; injected clients remain caller-owned.
Close attempts all pools before propagating an error. No request retries, queue
priority changes, new admission permits or background cleanup workers are added.
Use a transport only within one event loop; counters are not thread-safe.

`ProxyConfig.upstream_pool_shards` and `--upstream-pool-shards` accept explicit strict
integers 1 through 100; leaving the setting unset selects the window-based
policy. Setting 1 restores stock HTTPX rather than a custom
one-pool transport. If standard environment/system proxy discovery reports any
proxy, the factory also retains stock HTTPX, even when `NO_PROXY` could exempt the
particular upstream. That conservative fallback preserves routing semantics. A
`NO_PROXY` entry by itself permits sharding. Pools are fixed at construction and
are not resized when the admission window changes.

## Costs and limits

Selection scans O(P) counters. Pools and counters add O(P) objects, while maximum
connections stay C=100. HTTPcore's request/connection scans and internal assignment
retries remain, but their scope is a shard. Cleanup can have nested scans and
assignment visits queued requests; this change does not make those algorithms
logarithmic. With fixed P and C, this is principally a constant-factor transport
optimization, not a different asymptotic bound in the number of sessions.

Once assigned, a request is not migrated when another shard becomes free. Uneven
stream lifetimes or overload can therefore leave local queueing despite free
capacity elsewhere. Partitioned pools also maintain independent DNS connections
and keep-alive state. The retained total cap remains independent of the admission
window. More idle sockets can be retained than before. These are reasons to
measure the target workload and preserve an explicit rollback setting.

## Focused HTTP fixture

`atfm_experiments.load.transport_benchmark` compares four variants:

- `default`: unmodified HTTPX transport, total cap 100 and idle cap 20.
- `retained`: one pool with total/idle caps both 100.
- `shards8`: eight pools with combined total/idle cap 100.
- `shards16`: sixteen pools with combined total/idle cap 100.

Each lane issues sequential nonstreaming requests; all lanes start together.
The worker has as many service slots as offered concurrency and sleeps for 5 ms
per request. This is closed-loop offered concurrency, not an open-loop arrival
rate or a production capacity estimate. Three repetitions reverse variant order
on alternating repetitions. A fresh HTTP client is constructed for each variant;
the worker persists across variants. Construction, server startup and final close
are excluded from CPU/wall timing. First-request TCP setup remains included.
CPU is client-process time only, including trace recording, excluding worker CPU.

Trials use 8 lanes × 96 turns, 64 × 48, and 256 × 12. At 256 lanes, demand exceeds
the 100-connection transport cap. Every request's status/error, duration and TCP
trace events are retained. All lanes continue after errors, which remain visible
and make the command fail. The service is nonstreaming HTTP/1.1 over localhost;
this does not measure TLS handshakes, remote network latency or inference.

The 8/64-concurrency trials use the experimental implementation later committed
as `bb2a561`. The 64-lane run records HEAD `b7e7250`, with its new experiment code
still uncommitted; all captured source hashes were verified against `bb2a561`.
The 8-lane run records `bb2a561`. The 256-lane trial uses `0425a58`, which moves
the wrapper into core and shares its TLS context. Construction is excluded from
these timings. This difference remains explicit rather than rewriting provenance.

| Concurrency | Variant | Client CPU range (s) | Per-run p95 range (ms) | TCP connects per run |
| ---: | --- | ---: | ---: | ---: |
| 8 | default | 0.349-0.377 | 7.66-8.32 | 8-8 |
| 8 | retained | 0.352-0.372 | 7.97-8.01 | 8-8 |
| 8 | shards8 | 0.312-0.319 | 8.28-8.90 | 8-8 |
| 8 | shards16 | 0.310-0.326 | 8.54-9.08 | 8-8 |
| 64 | default | 1.979-2.052 | 56.52-59.29 | 3072-3072 |
| 64 | retained | 7.156-7.706 | 536.58-562.38 | 68-73 |
| 64 | shards8 | 1.473-1.549 | 53.36-61.69 | 64-64 |
| 64 | shards16 | 1.278-1.305 | 40.19-45.70 | 64-64 |
| 256 | default | 16.557-17.440 | 1710.63-1819.36 | 3072-3072 |
| 256 | retained | 8.454-9.049 | 3065.14-3148.74 | 100-108 |
| 256 | shards8 | 2.503-2.666 | 571.99-599.83 | 100-100 |
| 256 | shards16 | 1.754-1.849 | 460.48-500.21 | 100-100 |

All **82,944 formal component requests** succeeded. Ranges describe the minimum
and maximum of three run statistics, not confidence intervals. At 64 concurrency,
16 shards beat stock CPU and p95 in all three repetitions. At 256 concurrency,
stock HTTPX spends much more CPU under its connection cap; sharding reduces that
cost but does not eliminate queueing. The large retained single pool can reduce
CPU at 256 while worsening tail latency, another reason to measure both.

At eight concurrent requests, 16 shards reduced CPU but increased p95 in every
repetition, from 7.66-8.32 ms to 8.54-9.08 ms. Wall duration rose from 0.669-0.711 s
to 0.742-0.827 s. The automatic small-window fallback was added in `321bf59`
after these results; it retains stock HTTPX at initial admission windows of 20
or fewer. This threshold follows the measured stock idle cap, not an optimized
universal crossover. A large configured window with low actual traffic still
uses shards and may incur the small-load cost. Explicit configuration can override
it. Tests verify both sides of the 20/21 boundary and the 64-slot paired path.


## Full-proxy paired validation

Six fresh pairs compare `bb2a561` against `0425a58` with seeds 7, 8, 9 at 256 and
1,024 sessions. Pair order alternates by seed; runs are sequential with fresh
server processes. The baseline uses a detached worktree and both sides use the
same interpreter with explicit source paths. All captured runtime hashes match
their commits. No HTTP proxy environment settings were active.

Both sides use the existing real proxy, board, control loop and fake worker;
64 proxy/worker slots, 5 ms worker service, three calls per session, two-second
open-loop session arrivals, burst-aligned tools, fresh client connections, four
unfinished prediction jobs, 50 ms prediction budget, 128 forecast draws, ten
250 ms slots, 500 ms maximum hold and 500 ms sleep after each control step.
Soft descriptor limit is 8,192. The host is an Intel Core Ultra 9 185H, Python
3.12.13, HTTPX 0.28.1 and HTTPcore 1.0.9. It is not reserved; pairing does not remove
all host-load noise. The generator and service processes share the same host.

| Sessions | Revision | Client p95 range (s) | Proxy CPU range (s) | Timely prediction use |
| ---: | --- | ---: | ---: | ---: |
| 256 | before | 0.344-0.377 | 1.807-1.901 | 69.44% |
| 256 | after | 0.272-0.329 | 1.304-1.328 | 67.62% |
| 1024 | before | 2.462-2.654 | 6.840-7.099 | 35.50% |
| 1024 | after | 1.422-1.465 | 4.459-4.750 | 31.93% |

All **23,040 paired requests** succeeded, with no client, session, control, probe
or tool-publication errors. CPU and client p95 improved in all six pairs. At
1,024 sessions the per-pair p95 reductions were about 40.5%, 45.1% and 46.4%, and
CPU reductions about 34.1%, 36.2% and 33.1%. Those are short-run local measurements,
not an estimated production speedup. Admission-wait p95 fell from 2.095-2.317 s
to 0.562-0.651 s, while upstream phase p95 fell from 0.426-0.446 s to 0.156-0.200 s.
These separate percentiles must not be added.

Timely prediction use fell in every pair. Across the large runs, used predictions
fell from 3,272/9,216 (35.50%) to 2,943/9,216 (31.93%). Immediate rejections fell
from 5,866 to 4,788, but caller timeouts rose from 78 to 1,485; prediction errors
remained zero. Prediction-wait p95 rose from 0.022-0.031 s to 0.108-0.136 s. A 50 ms
async deadline does not bound handler resumption under event-loop scheduling
pressure. Faster forwarding changes request/tool timing and the overlap of
prediction attempts. These observations do not establish one causal explanation.
The workload's beta=0 and identical fallback/service rates limit its ability to
measure the policy cost of lower prediction coverage. Keep that tradeoff visible
and validate prediction-sensitive workloads before a production claim.

The benchmark candidate is `0425a58`, before automatic small-window selection.
The final `321bf59` retains the same transport and 16-shard selection for this
64-slot workload, adding stock selection only for small windows when no explicit
setting is supplied. Boundary/factory tests establish that selection; the paired
numbers are not relabeled as measurements of a later revision.

A separate Yappi profile at `0425a58` completed all 3,072 requests. The main thread
used 15.861 CPU seconds; HTTPcore connection assignment used 0.798 cumulative
seconds across 6,462 calls and opened 64 upstream TCP connections. The prior
rank-candidate diagnostic profile used 22.461 main-thread CPU seconds, 2.550 in
connection assignment and 3,072 connections. These profiles were collected at
different times and are diagnostic, not paired latency evidence. Profile request/
drain limits are 120 seconds and the control limit is 20 seconds to tolerate
instrumentation; ordinary trials keep 20/30/2-second limits. Cumulative function
costs overlap and must not be summed as exclusive CPU.


## Validation and provenance

The final suite passes **528 tests, three skips and six existing warnings**.
All tracked Python functions remain at most 20 physical lines. Tests cover
capacity allocation/balancing, request bytes/headers/extensions, open response
ownership, pre-header cancellation/failure, body-read errors/cancellation,
idempotent and failed closes, all-pool cleanup after a close failure, shared
verifying TLS context, proxy-discovery fallback, invalid configuration and
automatic window boundaries. Real-socket tests partially consume a blocked
stream while other requests progress, verify disconnect cleanup, and check
connection reuse within a four-connection cap. Fixture tests preserve failures,
refuse output overwrite and verify child cleanup. These are not a streaming
performance benchmark or a remote TLS interoperability study.

[Archived evidence](results/sharded-transport-2026-09-28/) contains the isolated
trials, all full-stack pairs, exploratory probe, candidate CPU profile, exact
executed drivers, configuration, environment, client records, source hashes,
control observations, logs, shutdown records and pytest output. The manifest covers 363 artifacts. Large artifacts
are gzip-compressed. `manifest.json` records hashes/lengths of decompressed
contents and separately identifies recorded HEAD and matching source revision.
`comparison.csv` retains individual full-stack outcomes; profiles are marked
separately and must not be used as uninstrumented latency comparisons. Child
exit code -15 records normal harness SIGTERM.

To reproduce, use the named checkouts, interpreter/dependencies and archived
configurations, adapt the drivers' absolute paths, and select fresh output
directories. Keep paired order, workload and descriptor limit. The component
fixture can be run from the current checkout using:

```bash
uv run python -m atfm_experiments.load.transport_benchmark \
  --concurrency 64 --turns 48 --repeats 3 --out runs/transport-comparison
```

The next validation boundary is mixed-duration streaming over realistic network
latency and TLS, including slow consumers, cancellation and queue growth. The
present tests establish local lifecycle correctness, not streaming performance or
production readiness. Timely prediction coverage also needs evaluation against
forecast quality and serving outcomes; it must not be hidden behind the latency
improvement. These results do not establish a need for a Rust rewrite or a GPU
inference/cache benefit.
