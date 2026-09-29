# Research notes

These are dated records of studies, findings, or plans. Their numerical claims
apply to the workloads and revisions described, and their setup snippets may
predate today's launchers. Use the [project README](../../README.md) and
[operations guide](../operations.md) for current commands, and the
[status page](../status.md) for implemented versus validated capabilities.

| Note | Role |
| --- | --- |
| [GPU results and caching tradeoffs](gpu-results-report/README.md) | Detailed [PDF](gpu-results-report/atfm-gpu-results-2026-09-29.pdf): completed and incomplete runs, nine figures, capacity and transfer analysis. |
| [Public H100 workloads](2026-09-29-public-gpu-workloads.md) | BFCL tool-call API cases, complete Weka sessions, cache pressure and retained replay failures. |
| [H100 cache compatibility](2026-09-29-h100-cache.md) | Native deployment fixes, completed CPU warming and real external-cache reuse. |
| [Real GPU cache validation](2026-09-28-gpu-cache.md) | Pinned vLLM/LMCache, completed CPU warming, real external reuse and rental handoff. |
| [Offline policy-selection workflow](2026-09-28-policy-tuning.md) | Constrained selection, held-out uncertainty, reproducible smoke validation and limits. |
| [Bounded upstream pool sharding](2026-09-28-sharded-transport.md) | Isolated transport trials, response-lifetime balancing, cleanup contracts and full-proxy validation. |
| [Proxy ranks and transport](2026-09-28-proxy-ranking.md) | Exact peer counts, request phases, paired trials and a rejected connection-pool change. |
| [Board computation isolation](2026-09-28-board-isolation.md) | Versioned prediction views, bounded control worker, freshness and paired load evidence. |
| [Prediction overload protection](2026-09-28-prediction-overload.md) | Bounded prediction work, deadline semantics, and paired load evidence. |
| [LLM inference serving background](../paper/background/README.md) | Foundations, current systems, agent scheduling, and ATFM rationale; primary-source survey dated 28 September 2026. |
| [Core refactor validation](2026-09-28-core-refactor.md) | Shared boundaries, exact output parity, retained timings, and rejected parser extraction. |
| [Proxy CPU profiling](2026-09-27-proxy-profiling.md) | Thread-aware profiles, redundant import searches, and paired dependency optimization trials. |
| [Repeated control attribution](2026-09-27-control-attribution.md) | Paired trials, prediction fallback counts, descriptor limits, and worker headroom. |
| [HTTP load baseline](2026-09-27-http-load-baseline.md) | Real local service processes, fake worker, scheduled agent turns, burst scenarios, and raw evidence. |
| [Control scaling implementation](../development/control-scaling.md) | Implemented fixes, design tradeoffs, parity tests, and before/after measurements. |
| [Current bottleneck audit](2026-09-27-current-bottlenecks.md) | Fresh CPU profiles, history/queue trials, and whole-control-cycle priorities. |
| [GDP stress tests](2026-09-27-gdp-stress.md) | 100,000- and one-million-session synthetic CPU/memory trials. |
| [Agent observability infrastructure](2026-09-27-agent-observability.md) | Existing tool tracing/evaluation systems and proposed ATFM signal boundaries. |
| [Large-session control paths](2026-09-27-large-control-paths.md) | Exact GDP threshold reuse, admission batching, and 8,192-session trials. |
| [CPU scaling and Python/Rust](2026-09-27-cpu-scaling.md) | Measured CPU optimizations, asymptotic bounds, and language decision. |
| [Demonstration ladder](2026-09-22-demonstration-ladder.md) | Evaluation stages and the claims each can support. |
| [Evaluation benchmarks](2026-09-22-evaluation-benchmarks.md) | Benchmark and dataset survey. |
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

- [Local cluster preparation, 28 September](2026-09-28-local-cluster.md): public AgentX replay through AIPerf and two/four Dynamo Mocker workers; integration evidence only.
