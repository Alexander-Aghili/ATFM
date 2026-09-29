# Implementation and evidence status

Prediction-path status updated on 28 September 2026. This page describes the
implemented paths and the limits of the recorded evidence. Dated plans and
research notes retain the assumptions and findings of their original runs.

| Area | Implemented path | Evidence and remaining work |
| --- | --- | --- |
| Offline forecasting | Trace adapters; B0/B1 aggregate baselines; B2/M1/M2/M3 session models; Monte Carlo aggregation and calibration. | H1 and H1b results are recorded in dated research notes. Results depend on corpus, split, horizon, and signal quality. |
| CPU simulation | Closed-loop workers, queues, KV eviction, trace replay, admission, touch/pin arms, paired comparisons. | Seeded and golden tests plus H2 run summaries. The engine remains a model of serving behavior. |
| Live board and proxy | HTTP predictions, JSONL event ingestion, directive polling, holds, launch gates, and touches. | Mocker integration verifies wiring; it does not prove a serving-performance benefit. |
| Prediction overload | Bounded executor admission, one caller deadline, remaining-budget board HTTP, worker/caller counters and owned-resource shutdown. | Overload regression tests and [paired HTTP trials](research/2026-09-28-prediction-overload.md); CPU isolation is recorded separately below. |
| Board isolation | One bounded control worker, atomic prediction views, cached JSON, monotonic freshness and stage metrics. | [Paired trials](research/2026-09-28-board-isolation.md) improved coverage and board lag but regressed large-case client p95. Threads still share the GIL. |
| Upstream transport | Response-lifetime balancing across bounded HTTPX pools; stock fallback for proxy discovery. | [Focused and paired trials](research/2026-09-28-sharded-transport.md), lifecycle tests; remote TLS and production streaming performance remain unvalidated. |
| Peer ranking | Exact per-tier index counts, maintained with queue mutations. | [Profiles and paired trials](research/2026-09-28-proxy-ranking.md); O(U) queries, still O(Q) when all priorities differ. |
| Controller capacity input | Prometheus worker-metrics parsing and configurable scraping. | The recorded Mocker frontend did not expose the required KV metrics, so the end-to-end run issued no holds or touches. |
| LMCache placement | Tokenization/prompt lookup and configurable pin/move/lookup/unpin HTTP adapter. | Adapter tests exist. Compatibility and performance must be validated on the actual controller and real workers. |
| Tier planning | Quantile-based tier recommendations; optional LMCache actuation. | Without LMCache, the runtime logs tier recommendations. No general hardware tier-placement result is claimed here. |
| Replica floor | Forecast-based minimum-replica proposals and a virtual connector. | The default runtime logs proposals; no external autoscaler is wired by the launcher. |
| Redis Streams | Library bus with resumable consumer cursor and malformed-event handling. | Local launch scripts use JSONL. Cursor persistence is not a checkpoint of board/model state. |
| Sidecar adapters | Tool wrappers, progress parsers, mini-SWE-agent, OpenHands, and Harbor adapters. | Tests cover adapters and byte-preserving execution; coverage depends on available tool signals. |
| GPU study | Study design and experiment configurations. | The repository's local integration evidence is not a completed H100 performance study. |

## Which document is authoritative?

- [README](../README.md): current setup, executable examples, package map.
- [Operations](operations.md): launcher behavior and integration requirements.
- [Core guide](development/core.md): units, ownership, randomness, extension contracts.
- [Architecture views](architecture/02-container.md): system boundaries and design intent.
- [Research notes](research/README.md): findings from identified experiments.
- [Design and implementation plans](superpowers/README.md): historical decisions and task recipes.
- [Paper guide](paper/README.md): the publication source, frozen evidence, and build process.

H2's manifest records the current Git SHA and configuration/input hashes; it
does not capture uncommitted source changes. Preserve the actual source revision
and inputs when releasing a result. H1 writes resolved configuration and scores,
but its runner does not currently emit the same provenance manifest as H2.
