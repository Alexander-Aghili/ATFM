# Documentation

## Start here

| Goal | Read |
| --- | --- |
| Learn LLM serving, KV caches, and ATFM's motivation | [Technical background paper](paper/background/README.md) |
| Understand ATFM and run a CPU example | [Project README](../README.md) |
| Distinguish implemented features from validated outcomes | [Implementation status](status.md) |
| Start the proxy, board, and controller | [Local operations](operations.md) |
| Inspect tool-call test traces and evaluations | [Phoenix test workflow](development/observability-tests.md) |
| Change core code safely | [Core developer guide](development/core.md) |
| Understand experiment packaging | [Experiment workspace](../experiments/README.md) |
| Assess Python/Rust and scaling risks | [Performance assessment](development/performance.md) |
| Prepare tests and a contribution | [Contributing](../CONTRIBUTING.md) |

## Architecture and design

- [Detailed code and documentation architecture](architecture/03-code-and-documentation.md): [PDF edition](architecture/atfm-architecture.pdf), six diagrams, module ownership, endpoints, data flow, evaluation and publication pipelines.

- [System context](architecture/01-context.md) and [container view](architecture/02-container.md).
- [Architecture design specification](superpowers/specs/2026-09-22-atfm-architecture-design.md).
- [Master research plan](../detail.md).

The [design-record index](superpowers/README.md) distinguishes historical task plans from current instructions.

Design documents describe intent. For implemented behavior and extension
contracts, use the developer guide and the source code together.

## Evaluation and evidence

- [Prediction overload protection](research/2026-09-28-prediction-overload.md): bounded jobs, deadlines, and paired HTTP trials.

- [Large synthetic GDP stress tests](research/2026-09-27-gdp-stress.md): CPU and memory at 100,000 and one million sessions.
- [Agent observability infrastructure](research/2026-09-27-agent-observability.md): existing analytics systems and proposed ATFM integrations.

- [CPU scaling trials](research/2026-09-27-cpu-scaling.md): measured optimizations, complexity bounds, and Python/Rust assessment.

- [Demonstration ladder](research/2026-09-22-demonstration-ladder.md): the roles of offline, simulated, and serving evaluations.
- [Evaluation benchmarks](research/2026-09-22-evaluation-benchmarks.md): datasets, splits, and metrics.
- [H1 first results](research/2026-09-22-h1-first-results.md): initial forecast evaluation.
- [H2 simulation first results](research/2026-09-24-h2sim-first-results.md): initial closed-loop evaluation.
- [Local deployment check](research/2026-09-27-deploy-e2e.md): integration results and missing capacity telemetry.
- [H100 study design](research/2026-09-27-l2-h100-study-design.md): planned real-worker evaluation.
- [Research index](research/README.md): additional dated study notes and diagnostics.

Treat dated notes as records of particular experiments, not blanket claims
about all workloads or the current working tree.

## Paper

The [paper guide](paper/README.md) covers the manuscript and build artifacts.
The [visuals guide](paper/VISUALS.md) describes the paper's figures. Paper rendering
has its own tooling and is separate from running the Python test suite.

[Board isolation study](research/2026-09-28-board-isolation.md) records worker ownership, prediction freshness,
concurrency tests, and paired HTTP measurements.

[Proxy ranking and transport study](research/2026-09-28-proxy-ranking.md) records exact peer counts, phase timing, profiles, and retained/rejected optimization evidence.

[Upstream transport study](research/2026-09-28-sharded-transport.md) records focused HTTP trials, bounded pool sharding, response ownership and integrated measurements.

[Policy tuning](development/policy-tuning.md) describes constrained search, independent validation, Pareto tradeoffs and the path toward context-dependent policies.

- [Local cluster preparation](development/local-cluster.md): pinned AIPerf/Dynamo smoke, evidence scope and LMCache integration gaps.
