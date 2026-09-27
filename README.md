# ATFM — Agent Traffic Flow Management

ATFM forecasts when active AI-agent sessions will return to an LLM, estimates
near-term demand, and uses those forecasts to control admission and KV-cache
placement on a shared serving pool.

An agent waiting for a test suite or build is temporarily absent from the LLM
queue, but may soon need its cached context again. ATFM combines session history,
elapsed tool time, progress signals, and backend observations to estimate that
return. It evaluates both prediction quality and the serving consequences of
acting on the prediction.

**Status:** research implementation with CPU simulation, trace evaluation, and
local service integration. Simulator results and Dynamo Mocker checks do not
establish real-GPU performance or production readiness. See
[deployment status and limitations](docs/operations.md#status-and-limitations).

[Quickstart](#quickstart) · [Experiments](#experiments) ·
[Local services](docs/operations.md) · [Core development](docs/development/core.md) ·
[Contributing](CONTRIBUTING.md) · [Documentation index](docs/README.md)

## What is included

- A predictor ladder, from history-only baselines to progress-aware and
  backend-aware session models.
- Monte Carlo forecasts of KV blocks and prefill tokens by traffic class and
  horizon, plus calibration and forecast scoring.
- An OpenAI-compatible chat-completions proxy with an admission window,
  priority ordering, bounded holds, launch gates, and keep-alive touches.
- Controllers for scheduled admission, cache touches, tier recommendations,
  and a proposed minimum replica count; an LMCache placement actuator.
- A closed-loop simulator with worker queues, KV eviction, recomputation,
  touches, pins, oracle comparisons, and paired experiment metrics.
- Sidecar tool instrumentation, trace adapters, and in-memory, JSONL, and
  Redis Streams event buses.

## Quickstart

The simulator needs **Python 3.12** and **uv**. It does not need a GPU, a running
LLM server, Docker, or a downloaded trace corpus. Run commands from the repository
root. The project currently requires Python `>=3.12,<3.13`.

```bash
git clone git@github.com:Alexander-Aghili/ATFM.git
cd ATFM
uv sync --extra dev
uv run pytest -q
uv run python scripts/run_h2sim.py experiments/quickstart.yaml
```

The quickstart compares `native`, `proxy_rules`, and `forecast_M2_kv` across
three seeds with a 300-second simulated arrival window. It prints serving
metrics and paired comparisons and writes artifacts under `runs/quickstart/`.
This is a functional example, not a statistically sufficient benchmark.
Simulated time is not wall-clock runtime; sessions may finish after the arrival
window closes.

Inspect the per-seed metrics:

```bash
uv run python -c "import pandas as pd; print(pd.read_csv('runs/quickstart/metrics.csv').to_string(index=False))"
```

`runs/` and `data/` are ignored by Git. Reusing an experiment's `name` and
`out_dir` writes to the same output directory; use a new name to preserve a run.

## Installation options

The core numerical and trace dependencies are installed by `uv sync`. Optional
extras are defined in [pyproject.toml](pyproject.toml).

| Extra | Adds | When to use it |
| --- | --- | --- |
| `dev` | pytest, coverage, async tests, HTTP test dependencies | Development and ordinary tests. |
| `serve` | FastAPI, Uvicorn, HTTPX | Board, proxy, and control-loop processes. |
| `harness` | mini-SWE-agent | The mini-SWE-agent adapter. |
| `dynamo` | NVIDIA Dynamo | Local Mocker/frontend integration. |

Combine the extras you need in one sync command:

```bash
uv sync --extra dev --extra serve
# For the local Dynamo workflow:
uv sync --extra dev --extra serve --extra dynamo
```

Docker is required for the scripted container-based trace collector. Redis and
LMCache workflows have additional runtime requirements; see the
[operations guide](docs/operations.md). Installing an extra does not start its
external services or provide a trace corpus.

## How the pieces fit

1. The proxy and sidecar emit session, LLM, and tool events.
2. The board reconstructs session state and samples future return times and
   demand. The same tick orchestration is reused in simulation.
3. Controllers combine predictions with capacity and residency observations to
   produce expiring directives.
4. The proxy or placement actuator executes supported actions; the simulator
   measures comparable actions against modeled workers.
5. Evaluation separates forecast accuracy from serving outcomes and action costs.

| Package under `src/atfm/` | Responsibility | Starting point |
| --- | --- | --- |
| `schema` | Events, trace tables, forecast snapshots | `TraceTable`, `parse_event`, `ForecastSnapshot` |
| `traces` | Corpus adapters, synthetic workloads, overlays, family-aware splits | `generate`, `overlay_sessions`, `events_to_trace_table` |
| `board` | Session registry, predictors, aggregation, calibration, HTTP service | `LiveBoard`, `SessionForecaster`, `create_board_app` |
| `control` | Admission, placement, replica proposals, actuation | `GdpPlanner`, `TouchController`, `ControlLoop`, `LMCacheActuator` |
| `proxy` | Admission queue and chat-completions forwarding | `ProxyConfig`, `create_app` |
| `sidecar` | Tool execution and progress instrumentation | `run_tool`, `SidecarExecutor`, `wrap_executor` |
| `bus` | Event transport and replay | `InMemoryBus`, `JsonlBus`, `RedisStreamsBus` |
| `sim` | Programs, workers, event loop, policy arms | `Simulator`, `EngineConfig` |
| `eval` | Forecast and serving metrics, paired comparisons | `score_tick`, `serving_metrics`, `paired_contrasts` |
| `experiments` | Configuration, runners, output artifacts | `H1Config`, `H2SimConfig` |
| `dynamo`, `collect` | Local serving processes and Docker trace collection | `LocalDynamo`, `run_collection` |

For timing, array shapes, state ownership, and extension contracts, read
[Working on the ATFM core](docs/development/core.md). The
[architecture documents](docs/architecture/01-context.md) describe the broader
system boundaries.

## Experiments

### H1: forecast accuracy

Run a synthetic fleet without external data:

```bash
uv run python scripts/run_h1.py experiments/h1_synthetic.yaml
```

The supplied configuration writes to `runs/h1_synth_mixed/`. H1 produces
`metrics.csv`, `surge.json`, and `config.json`, plus `calibration.json` when
calibration is enabled. Metrics include CRPS, pinball loss, interval coverage,
and surge detection. H1 evaluates demand forecasts; it does not measure a
controller's effect on serving latency.

| Model | Information used |
| --- | --- |
| `B0` | Last-observation persistence, repeated across draws (zero without history). |
| `B1` | Kalman aggregate time-series baseline. |
| `B2` | Per-tool historical durations without survival conditioning. |
| `M1` | Historical durations conditioned on elapsed time. |
| `M2` | Progress-aware return-time prediction. |
| `M3` | Progress prediction with a shared backend speed factor. |

Corpus runs require the input files named in their YAML configurations:

```bash
uv run python scripts/run_h1.py experiments/h1_tracelab.yaml
uv run python scripts/h1b_resumption.py runs/collect/l1_varied_events.jsonl --min-duration 30
```

The repository does not supply the ignored `data/` directory. Configure trace
paths and splits explicitly; keep related session families together when
splitting. See [evaluation benchmarks](docs/research/2026-09-22-evaluation-benchmarks.md).

### H2: closed-loop serving outcomes

```bash
uv run python scripts/run_h2sim.py experiments/h2sim_loaded_kv.yaml
uv run python scripts/eviction_diagnostic.py experiments/h2sim_loaded_kv.yaml \
  --arms oracle_kv forecast_M2_kv --duration 900
```

H2 supports `short_tool`, `long_tool`, `interactive_long_tool`, and `trace`
regimes. The trace regime requires a canonical parquet table configured through
`trace_path`. Complete options live in
[`H2SimConfig`](src/atfm/experiments/h2sim.py).

| Policy family | Examples | What changes |
| --- | --- | --- |
| Baselines | `native`, `proxy_rules`, `working_set` | Native scheduling, proxy rules, or a working-set budget. |
| Admission forecasts | `forecast_M1`, `forecast_M2`, their `_nohold` variants | Forecast-based index and optional holds. |
| Admission oracles | `oracle`, `oracle_rule`, `oracle_rule_noidx` | True future information for comparison. |
| Eviction placement | `forecast_M1_kv`, `forecast_M2_kv`, `oracle_kv` | Order idle contexts by expected next use. |
| Placement variants | `_kv_size`, `_kv_cw` | Size-aware or class-weighted eviction scores. |
| Oracle diagnostic | `oracle_kv_fresh` | Refresh true return times at each eviction. |
| Keep-alive touches | `forecast_M1_touch`, `forecast_M2_touch`, `oracle_touch`, `touch_random` | Refresh selected contexts within a touch budget. |
| Hard pins | `forecast_M1_pin`, `forecast_M2_pin`, `oracle_pin`, `pin_random` | Protect selected contexts from eviction within a pin budget. |

Touch options include `touch_prefetch`, `touch_retry`, and `touch_yield`. Pin
options include `pin_horizon_s` and `pin_budget_blocks`. Oracle policies use
information unavailable to deployable predictors; their outcomes also depend on
the chosen control rule and objective.

H2 writes the following beneath `runs/<name>/` by default:

| Artifact | Contents |
| --- | --- |
| `metrics.csv` | Serving metrics for each arm and seed. |
| `paired.csv` | Paired comparisons against the native arm. |
| `contrasts.csv` | Direct arm-to-arm comparisons, when applicable. |
| `queue_location.csv` | Queueing and hold-cost breakdown. |
| `log_<arm>_<seed>.parquet` | Per-call simulation logs. |
| `config.json` | Resolved experiment configuration. |
| `manifest.json` | Version, Git/config provenance, and input hashes. |

Interpret latency together with background completion time, recomputed prefill,
and hold/touch/pin costs. Include the workload, seed set, metric, and horizon or
SLO with every reported result. Research notes under [docs/research](docs/research/)
record specific runs and their limitations.

## Running local services

The [operations guide](docs/operations.md) contains the full startup sequence,
required files, API examples, controller wiring, and shutdown commands.
The service processes are started separately:

```bash
uv run python scripts/run_proxy.py --help
uv run python scripts/run_board.py --help
uv run python scripts/run_control.py --help
```

The proxy exposes `/v1/chat/completions`. Clients can supply `x-atfm-session`,
`x-atfm-class`, and `x-atfm-tenant` headers to identify traffic. A stable session
identifier is needed to associate successive calls with the same context.

## Testing and development

```bash
uv sync --extra dev
uv run pytest -q
uv run pytest -q tests/board tests/control tests/sim
uv run pytest tests/board tests/control tests/sim \
  --cov=atfm.board --cov=atfm.control --cov=atfm.sim --cov-report=term-missing
```

The ordinary suite covers unit contracts, HTTP integration, seeded simulation,
and golden serving metrics. Two Dynamo tests are opt-in with `ATFM_DYNAMO=1` and
require the Dynamo runtime. Run them after installing the relevant extras:

```bash
ATFM_DYNAMO=1 uv run pytest -q tests/dynamo tests/e2e
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for change scope, test expectations,
reproducibility, and documentation conventions. Keep scientific invariants in
comments and public contracts in docstrings; put design discussion in the docs.

## Further reading

- [Documentation index](docs/README.md): a map of setup, design, research, and paper material.
- [Core developer guide](docs/development/core.md): extension points and behavioral contracts.
- [Architecture design](docs/superpowers/specs/2026-09-22-atfm-architecture-design.md): detailed system design.
- [Demonstration ladder](docs/research/2026-09-22-demonstration-ladder.md): what each evaluation stage establishes.
- [Paper build guide](docs/paper/README.md): manuscript and figure artifacts.
- [Master plan](detail.md): the longer research and implementation plan.
