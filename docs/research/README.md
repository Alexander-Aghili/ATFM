# Research notes

These are dated records of studies, findings, or plans. Their numerical claims
apply to the workloads and revisions described, and their setup snippets may
predate today's launchers. Use the [project README](../../README.md) and
[operations guide](../operations.md) for current commands, and the
[status page](../status.md) for implemented versus validated capabilities.

| Note | Role |
| --- | --- |
| [HTTP load baseline](2026-09-27-http-load-baseline.md) | Real local service processes, fake worker, scheduled agent turns, burst scenarios, and raw evidence. |
| [Control scaling implementation](../development/control-scaling.md) | Implemented fixes, design tradeoffs, parity tests, and before/after measurements. |
| [Current bottleneck audit](2026-09-27-current-bottlenecks.md) | Fresh CPU profiles, history/queue trials, and whole-control-cycle priorities. |
| [GDP stress tests](2026-09-27-gdp-stress.md) | 100,000- and one-million-session synthetic CPU/memory trials. |
| [Agent observability infrastructure](2026-09-27-agent-observability.md) | Existing tool tracing/evaluation systems and proposed ATFM signal boundaries. |
| [Large-session control paths](2026-09-27-large-control-paths.md) | Exact GDP threshold reuse, admission batching, and 8,192-session trials. |
| [CPU scaling and Python/Rust](2026-09-27-cpu-scaling.md) | Measured CPU optimizations, asymptotic bounds, and language decision. |
| [Demonstration ladder](2026-09-22-demonstration-ladder.md) | Evaluation stages and the claims each can support. |
| [Evaluation benchmarks](2026-09-22-evaluation-benchmarks.md) | Benchmark and dataset survey. |
| [Platform acquisition](2026-09-22-platform-acquisition.md) | Infrastructure planning at the time of the survey. |
| [First H1 results](2026-09-22-h1-first-results.md) | Initial TraceLab forecast evaluation. |
| [Second corpus and H1b](2026-09-23-h1-second-corpus-and-h1b.md) | AgentX generalization and progress-conditioned resumption. |
| [L1 results](2026-09-23-l1-results.md) | Local collection and serving integration. |
| [First H2 results](2026-09-24-h2sim-first-results.md) | Closed-loop simulation results and later diagnostic additions. |
| [Deployment end to end](2026-09-27-deploy-e2e.md) | Board/proxy/control wiring and missing Mocker capacity input. |
| [H100 study design](2026-09-27-l2-h100-study-design.md) | Proposed real-worker validation and experimental controls. |
| [DSec paper notes](2026-09-27-dsec-paper-notes.md) | Related-work notes, not an ATFM performance experiment. |

When extending a study, retain the original workload and uncertainty definitions.
Identify added runs explicitly rather than silently replacing earlier results.
Link the configuration, preserve input provenance, distinguish illustrative
calculations from observations, and report action costs alongside benefits.
