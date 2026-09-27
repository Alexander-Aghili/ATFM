# ATFM: Agent Traffic Flow Management

Forecast LLM demand from in-flight agent sessions on a shared NVIDIA Dynamo pool, and act on it: order and admit calls at a proxy, keep the KV of sessions about to return resident, and measure what every action costs.

- Master plan: `detail.md`. Design spec v1.1: `docs/superpowers/specs/2026-09-22-atfm-architecture-design.md`.
- Results notes: `docs/research/`. Paper: `docs/paper/` (`python3 docs/paper/build_paper.py --pdf`).
- Architecture diagrams (reladraw): `docs/architecture/`.

## Install

```bash
uv sync                                  # Python 3.12; adds the atfm package in editable mode
uv sync --extra dynamo                   # ai-dynamo v1.5 (Mocker workers, frontend) for the L1 path
uv sync --extra minisweagent             # mini-SWE-agent harness adapter
uv run pytest -q                         # ~200 tests, about 35 s; 2 tests gated on ATFM_DYNAMO=1
```

## Components (`src/atfm/`)

| package | what it is | entry points |
|---|---|---|
| `schema` | canonical trace row/table, forecast snapshot, event models | `TraceTable.from_parquet`, `parse_event` |
| `traces` | TraceLab and AgentX adapters, Poisson overlay and family-aware splits, sidecar events to trace table, synthetic workload generator | `overlay_sessions`, `events_to_trace_table` |
| `board` | session registry, predictor ladder B0..M3, Monte Carlo fleet forecaster, dispersion calibration, Prometheus worker-metrics scraper, HTTP service | `LiveBoard`, `board.service.create_board_app` |
| `control` | GDP planner (ration-by-schedule on samples), placement touch controller, tier logger, replica floor with a virtual Planner connector; every directive expires | `GdpPlanner.plan`, `TouchController.plan`, `ReplicaFloor.propose` |
| `proxy` | OpenAI-compatible admission proxy: global window, priority tiers and index, hold directives with cap and expiry, overflow to FCFS with alarms, keep-alive `/touch`, launch gate `/gate` | `proxy.app.create_app` |
| `sidecar` | byte-exact tool wrapper with a parser chain (pytest, percent, rows, stage, training, counter, line rate), launch gate client, adapters for mini-SWE-agent, OpenHands and Harbor | `run_tool`, `SidecarExecutor`, `wrap_executor` |
| `bus` | event buses: in-memory, JSONL (dev and replay), Redis Streams with a resumable cursor (deploy) | `InMemoryBus`, `JsonlBus`, `RedisStreamsBus` |
| `sim` | closed-loop fleet simulator: programs, KV-aware worker engine (LRU with pluggable eviction order and keep-alive touch), heap event loop, policy arms | `Simulator`, `sim.policies`, `sim.forecast_arm`, `sim.kv_placement` |
| `eval` | CRPS, pinball, coverage, surge lead time; per-session resumption scoring; serving metrics with paired bootstrap and direct contrasts | `score_tick`, `leave_one_family_out`, `serving_metrics`, `paired_contrasts` |
| `experiments` | H1 forecast runner and sweeps; H2 closed-loop runner with regimes (synthetic or trace replay), arms, provenance manifest | `run_h1`, `run_h2sim` |
| `dynamo`, `collect` | Dynamo Mocker + frontend launcher with file discovery; scripted Docker trace collection through the sidecar | `scripts/dynamo_local.py`, `scripts/collect_traces.py` |

## Running things

```bash
# forecast evaluation on a corpus (needs data/ inputs, gitignored)
uv run python scripts/run_h1.py experiments/h1_tracelab.yaml
uv run python scripts/h1b_resumption.py runs/collect/l1_varied_events.jsonl --min-duration 30

# closed-loop simulator (writes runs/<name>/{metrics,paired,contrasts}.csv, manifest.json, per-arm logs)
uv run python scripts/run_h2sim.py experiments/h2sim_loaded_kv.yaml

# proxy in front of a Dynamo frontend, and the board service
uv run python scripts/run_proxy.py --upstream http://127.0.0.1:8000 --window 8 --events runs/events.jsonl
uv run python scripts/run_board.py --events runs/events.jsonl --train data/tracelab/tracelab.parquet --serve 8081
uv run python scripts/dynamo_local.py            # Mocker workers + frontend on this machine (~4 s)
bash scripts/l1_e2e.sh                           # proxy + Mocker + scripted sessions end to end
```

Simulator arms: `native`, `proxy_rules`, `forecast_M1`, `forecast_M2` (+`_nohold`), `oracle`, `oracle_rule` (+`_noidx`), `working_set`, placement `forecast_M{1,2}_kv` (+`_size`, `_cw`), `oracle_kv` (+`_size`, `_cw`), keep-alive `forecast_M{1,2}_touch`, `oracle_touch`. Regimes: `short_tool`, `long_tool`, `interactive_long_tool`, `trace` (replay a parquet table).

## Conventions

- Results are reported with metric, horizon and split next to every number; policy claims come only from closed-loop runs.
- Every controller and the sidecar fail open. Directives carry `expires_at`.
- Reduced precision and other risky modes are opt-in only.
- Experiments are YAML under `experiments/`; each run directory carries a `manifest.json` with git SHA, config hash and input hashes.
