# Running and interpreting experiments


Runners live in the separate [experiment workspace](../experiments/README.md),
under `experiments/src/atfm_experiments/`. The default development dependency
group installs them; the `atfm` wheel contains only the reusable core. Use
`uv sync --no-dev --extra serve` for a core-only service environment.

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
splitting. See [evaluation benchmarks](../docs/research/2026-09-22-evaluation-benchmarks.md).

### H2: closed-loop serving outcomes

```bash
uv run python scripts/run_h2sim.py experiments/h2sim_loaded_kv.yaml
uv run python scripts/eviction_diagnostic.py experiments/h2sim_loaded_kv.yaml \
  --arms oracle_kv forecast_M2_kv --duration 900
```

H2 supports `short_tool`, `long_tool`, `interactive_long_tool`, and `trace`
regimes. The trace regime requires a canonical parquet table configured through
`trace_path`. Complete options live in
[`H2SimConfig`](../experiments/src/atfm_experiments/h2sim.py).

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
SLO with every reported result. Research notes under [docs/research](../docs/research/)
record specific runs and their limitations.
