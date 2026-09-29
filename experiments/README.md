# Experiment workspace

This directory contains research configurations and the separately packaged
`atfm_experiments` runners. The reusable library remains in `../src/atfm/`.
Experiment orchestration depends on the core; the core must not depend on these
runners.

| Path | Purpose |
| --- | --- |
| `src/atfm_experiments/h1.py` | H1 configuration, fitting, calibration, scoring, and result output. |
| `src/atfm_experiments/h2sim.py` | H2 workloads, policy selection, simulation runs, comparisons, and provenance. |
| `*.yaml` | Experiment and local integration configurations. |
| `pyproject.toml` | Independent `atfm-experiments` distribution, part of the root uv workspace. |

From the repository root:

```bash
uv sync --extra dev
uv run python scripts/run_h2sim.py experiments/quickstart.yaml
uv run python scripts/run_h1.py experiments/h1_synthetic.yaml
```

The root's default `dev` dependency group installs the experiment package in
editable mode. The existing `dev` extra supplies test tools. Commands and YAML
paths are unchanged by the package split; Python imports now use
`atfm_experiments.h1` and `atfm_experiments.h2sim` instead of `atfm.experiments`.
There is intentionally no compatibility module inside the core package.

For a core-only service environment, exclude development groups:

```bash
uv sync --no-dev --extra serve
```

Keep `--no-dev` on subsequent `uv run` commands in this environment.
This omits the experiment package and is not an environment for running the full
repository test suite. Restore `uv sync --extra dev` for development. Building
the core wheel includes only `atfm`; the experiment package has its own wheel:

```bash
uv build --package atfm
uv build --package atfm-experiments
```

See the [experiment guide](guide.md),
[core development guide](../docs/development/core.md), and
[research index](../docs/research/README.md) for model contracts and evidence.

## CPU scaling trials

The benchmark driver uses fixed seeds, an untimed warmup, repeated wall-clock
measurements, and optional separate cProfile runs. It writes CSV timings, result
fingerprints, source hashes, and environment metadata; profiles are not included
in the timed measurements.

```bash
uv run python -m atfm_experiments.benchmark_cpu \
  --cases eviction duration bootstrap --sizes 128 512 2048 8192 \
  --repeats 3 --profile --out runs/cpu-kernels
uv run python -m atfm_experiments.benchmark_cpu \
  --cases h1 h2 --sizes 32 128 --repeats 3 --profile --out runs/cpu-e2e
uv run python -m atfm_experiments.benchmark_cpu \
  --cases queue forecast --sizes 128 512 2048 8192 \
  --repeats 3 --profile --out runs/cpu-scaling
```

Run before/after trials sequentially on the same otherwise-idle machine. `size`
means resident contexts for eviction, queued arrivals for queue, active sessions
for forecast, empirical observations for duration, aligned sessions for
bootstrap, and expected root sessions for H1/H2. H2's arrival window is fixed;
raising its session count also increases congestion and the run's completion
time. These trials are not constant-load fleet capacity estimates.

See the [CPU scaling report](../docs/research/2026-09-27-cpu-scaling.md) for
results, complexity bounds, and remaining limits. Timing thresholds are not
asserted in the unit suite; deterministic numerical contracts are.

For independent GDP session/slot/sample sweeps, use
`uv run python -m atfm_experiments.benchmark_gdp --out runs/gdp`.
The CPU driver also supports `--cases admission`, measuring one full-window
release batch. See the [large-session study](../docs/research/2026-09-27-large-control-paths.md)
for baseline revisions, open/saturated workloads, and the remaining limits.

The [GDP stress study](../docs/research/2026-09-27-gdp-stress.md) supplies larger
100,000- and one-million-session commands. Use `--regimes mixed` for varied
ETAs, resource demands, and tenants. GDP results now include process CPU time
and peak RSS; use one case per process to attribute the memory high-water mark.

## Optional tool-call observability

[Phoenix test integration](../docs/development/observability-tests.md) compares
the alternatives and gives reproducible local tracing/evaluation commands.
The client extra is `observability`; the Phoenix server has a separate locked
project under `observability/server/` to avoid changing the core's SDK versions.
Default tests require neither the server nor its optional packages.

## Local HTTP load trials

The [load harness](../docs/development/load-testing.md) runs the proxy, board, and
fake model worker in separate local processes, with scheduled session arrivals,
agent/tool turns, synchronized completion bursts, and periodic control updates.
It requires serving dependencies but no GPU or external API:

```bash
uv sync --extra dev --extra serve
uv run python -m atfm_experiments.load \
  --sessions 16 64 256 --patterns staggered burst --out runs/http-load-baseline
```

Keep output directories fresh; raw request, control, queue, process, and generator
lag records accompany configuration/source hashes and summaries. These are
integrated CPU/HTTP tests, not measurements of model-worker hardware capacity.

For repeated control-on/off pairs and inherited descriptor limits, see the
[attribution workflow](../docs/development/load-testing.md#repeated-control-attribution).

Opt-in [proxy CPU profiling](../docs/development/load-testing.md#profiling-the-proxy)
uses the experiments-only `profiling` extra and retains per-thread results.

Prediction load tests support `prediction_limit` and distinguish immediate
admission fallback from timed-out work. See [overload accounting and trials](../docs/development/load-testing.md#prediction-work-accounting).

[Board isolation study](../docs/research/2026-09-28-board-isolation.md) records worker ownership, prediction freshness,
concurrency tests, and paired HTTP measurements.

The [proxy ranking and transport study](../docs/research/2026-09-28-proxy-ranking.md) retains paired burst trials, thread-aware profiles, exact-rank microbenchmarks and a rejected HTTP pool change.

Run `python -m atfm_experiments.load.transport_benchmark --out runs/transport-comparison` for focused real-HTTP pool comparisons. See the [transport study](../docs/research/2026-09-28-sharded-transport.md) for configurations, limits and integrated results.

## Offline policy tuning

Run `python -m atfm_experiments.load.tune --plan experiments/tuning/smoke.json --out runs/tuning-smoke`
to exercise constrained settings selection and independent validation. The smoke
suite is deliberately small; see the [protocol](../docs/development/policy-tuning.md)
for objectives, workload contexts, bounded trials, Pareto results, uncertainty,
manual application and future Bayesian/contextual search. No core defaults change.

## Local serving preparation

The [local cluster recipe](../docs/development/local-cluster.md) runs AIPerf against Dynamo Mocker with a public AgentX root. The isolated client lock lives in `local-cluster/`; the runner lives in `src/atfm_experiments/local_cluster/`. This is CPU integration evidence only.

## Real GPU integration

[GPU cache validation](../docs/development/gpu-cache.md) runs pinned vLLM and LMCache
with real inference. `gpu-cache/requirements.lock` isolates the serving environment;
`atfm_experiments.gpu_cache` validates ATFM CPU warming, records evidence and cleans up.
