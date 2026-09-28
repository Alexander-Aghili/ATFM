# Local operation and integration

This guide covers the shipped launch scripts. For a CPU-only experiment without
services, use the [README quickstart](../README.md#quickstart).

## Prerequisites

- Python 3.12 and the `serve` extra. Include `dev` for tests.
- A running OpenAI-compatible upstream; the examples use port 8000.
- A canonical training parquet for the board. The examples use
  `data/tracelab/tracelab.parquet`, which is not included in a clone.
- Free local ports: 8000 for the upstream, 8081 for the board, 8799 for the proxy.

```bash
uv sync --extra dev --extra serve
mkdir -p runs/local
```

Commands below run from the repository root in separate terminals. The launcher
scripts bind the board and proxy to `127.0.0.1`.

## 1. Start an upstream

Use an existing server at `http://127.0.0.1:8000`, or install the Dynamo extra and
start the local Mocker/frontend pair:

```bash
uv sync --extra dev --extra serve --extra dynamo
uv run python scripts/dynamo_local.py up
curl --fail http://127.0.0.1:8000/v1/models
```

The launcher uses file discovery, writes logs and process IDs under
`runs/dynamo/`, and defaults to `Qwen/Qwen3-0.6B`. It can require access to model
metadata/tokenizer assets. Inspect `frontend.log` and `mocker.log` if startup
fails. A standalone check that starts and stops its own processes is:

```bash
uv run python scripts/dynamo_local.py smoke
```

## 2. Start the board

```bash
uv run python scripts/run_board.py \
  --events runs/local/events.jsonl \
  --snapshots runs/local/snapshots.jsonl \
  --train data/tracelab/tracelab.parquet \
  --serve 8081 \
  --control experiments/control_local.yaml
```

`--snapshots` is required by the parser even in HTTP mode. In HTTP mode snapshots
are exposed by the API; the file loop is not run. The CLI fits M2 and uses 256
draws over 10, 30, 120, 300, and 900 seconds. HTTP mode advances only when
`POST /tick` is called; the control loop below supplies those ticks.

Without `--serve`, the command tails the event file and appends snapshot JSONL
on its tick interval. Add `--once` for one file-mode iteration. File-mode ticks
use wall-clock time, so old event recordings are not an offline replay; use the
H1 evaluation runner for historical traces.

`experiments/control_local.yaml` attaches GDP, touch, tier, and replica
controllers. Configure its metrics URL and capacities for your actual workers.
A frontend page without worker KV metrics will not provide usable GDP capacity
or a touch eviction frontier.

## 3. Start the proxy

```bash
uv run python scripts/run_proxy.py \
  --upstream http://127.0.0.1:8000 \
  --port 8799 --window 8 \
  --events runs/local/events.jsonl \
  --trace runs/local/calls.jsonl \
  --board http://127.0.0.1:8081
```

The proxy and board must use the same event file. Harness-side tool telemetry
must reach that bus too; proxy call events alone do not describe tool progress.
Use the sidecar adapters for the relevant harness.

Send a request, replacing the model name if your upstream uses another model:

```bash
curl --fail http://127.0.0.1:8799/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-atfm-session: demo-session' \
  -H 'x-atfm-class: interactive' \
  -H 'x-atfm-tenant: demo' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
```

Classes are `interactive` and `background`; the default is `background`. Without
`x-atfm-session`, requests get generated anonymous identifiers. Additional
headers include `x-atfm-parent` and `x-atfm-deadline` (an absolute timestamp in
seconds). Consult `proxy/app.py` for metadata parsing and upstream forwarding.

## 4. Start the control loop

```bash
uv run python scripts/run_control.py \
  --board http://127.0.0.1:8081 \
  --proxy http://127.0.0.1:8799 \
  --interval 5 --log runs/local/control.jsonl
```

Each iteration ticks the board, fetches directives, applies unexpired holds and
touches, and records results. Use `--steps 10` for a bounded run. By default,
tier and replica proposals are logged; they do not move cache blocks or scale
an external cluster. The replica connector is currently virtual.

### Optional LMCache actuation

The LMCache path requires a separately running compatible controller, the
correct instance ID, and a tokenizer matching the serving model. `transformers`
is imported by this path but is not a declared project extra. One way to supply
it for the command is:

```bash
uv run --with transformers python scripts/run_control.py \
  --board http://127.0.0.1:8081 --proxy http://127.0.0.1:8799 \
  --lmcache http://127.0.0.1:9000 --lmcache-instance vllm-0 \
  --tokenizer Qwen/Qwen3-0.6B
```

With an actuator attached, touch directives become pins and tier directives
can invoke pin/move operations. Endpoint shapes and actual controller behavior
must be verified against the installed LMCache deployment. A unit-tested HTTP
adapter does not establish successful placement on real workers.

### Optional Redis transport

`RedisStreamsBus` accepts an injected client or a URL and can persist a consumer
cursor. URL construction requires the `redis` Python package, which is not in
the base dependencies. The current launch scripts use JSONL; selecting Redis
requires composing the bus in Python, not a documented CLI switch. A saved
cursor restores stream position, not the board's in-memory model/session state.

## API reference

| Service | Method and path | Purpose |
| --- | --- | --- |
| Board | `GET /healthz` | Process health. |
| Board | `GET /state` | Published prediction freshness, control admission, and stage timings. |
| Board | `POST /tick` | Ingest available events and compute a snapshot. |
| Board | `GET /snapshot` | Read snapshot quantiles and endogenous fractions. |
| Board | `POST /predict` | Predict service and next-tool duration for `session_id`, `isl`, `osl`. |
| Board | `POST /directives` | Plan once per snapshot; subsequent polls reuse the cached result. |
| Proxy | `GET /healthz` | Process health. |
| Proxy | `GET /state` | Queue/admission state and prediction outcome counters. |
| Proxy | `POST /v1/chat/completions` | Forward admitted chat requests, including streaming responses. |
| Proxy | `POST /directives` | Accept a session hold for subsequent submissions (legacy single-update endpoint). |
| Proxy | `POST /directives/batch` | Validate up to 1,024 holds, apply nonexpired updates, then schedule once. |
| Proxy | `POST /touch` | Refresh a remembered session prefix. |
| Proxy | `POST /gate` | Query a session's launch delay. |
| Proxy | `GET /session/{session_id}/prompt` | Retrieve remembered messages for placement tokenization. |

```bash
curl --fail http://127.0.0.1:8799/state
curl --fail http://127.0.0.1:8081/snapshot
curl --fail -X POST http://127.0.0.1:8081/tick
```

## Collection and end-to-end checks

The collection scripts need a working Docker daemon and network access for the
images and repositories named in the job YAML. The existing end-to-end scripts
also expect `data/tracelab/tracelab.parquet`.

```bash
uv run python scripts/collect_traces.py experiments/l1_jobs.yaml
bash scripts/l1_e2e.sh
bash scripts/l1_control_e2e.sh
```

Run one end-to-end script at a time, without separately running services on its
ports. These are local integration checks with fixed output paths. In
particular, `l1_control_e2e.sh` truncates its `runs/control/events.jsonl` before
starting; preserve any earlier recording you need.

## Troubleshooting and shutdown

| Symptom | Check |
| --- | --- |
| Missing training parquet | Supply a canonical `TraceTable` parquet and update `--train`. Corpora are not bundled. |
| `--snapshots` argument error | Supply the argument in both file and HTTP modes. |
| Snapshot time is null | Call `POST /tick` or start the control loop. |
| No holds or touches | Check worker metrics, nonempty capacity/frontier, live sessions, and configured controllers. |
| Board unavailable | Inspect board logs and prediction fallback/timeout behavior; check port 8081. |
| Missing Uvicorn | Sync with `--extra serve`. |
| Dynamo does not start | Inspect `runs/dynamo/` logs, port conflicts, and model asset availability. |
| Unexpected replay results | Historical event timestamps do not match live wall-clock ticks; use the offline runner. |

Stop foreground proxy, board, and controller processes with Ctrl-C. Stop the
Dynamo processes created by the launcher with:

```bash
uv run python scripts/dynamo_local.py down
```

## Status and limitations

The [recorded local integration check](research/2026-09-27-deploy-e2e.md) verified
event flow, board predictions, and directive polling. It issued no holds or
touches because the Mocker frontend metrics page lacked the required KV data.

Simulation includes eviction, touches, and pins. Real-worker placement
performance, capacity-driven controller effectiveness, and external replica
scaling require separate validation. Keep-alive touches consume serving
resources; pins depend on cache-layer support. Redis support is a library
integration, and the launch scripts remain JSONL-based.

The provided service launchers are local development processes. They do not
configure authentication or TLS for the control endpoints. Operating them as a
shared service requires deployment-specific access control, telemetry, and
process supervision outside these scripts.

See [control scaling decisions](development/control-scaling.md) for batching, expiry,
restart, and incremental JSONL consumption contracts.

## Prediction overload protection

The proxy bounds unfinished prediction jobs independently of its LLM admission
window. Defaults are four jobs and a 50 ms caller budget:

```bash
uv run python scripts/run_proxy.py --board http://127.0.0.1:8081 \
  --prediction-limit 4 --board-timeout 0.05
```

Python configuration uses `ProxyConfig.prediction_limit` and `board_timeout_s`.
Both must be positive; the budget must be finite. Saturated prediction admission
immediately uses the normal local service estimate and zero next-tool estimate.
It does not reject the LLM request or consume a waiting prediction queue.

Timeout and cancellation release the caller, but a running job retains capacity
until its worker finishes. Board HTTP phases receive the remaining monotonic
budget instead of a 500 ms minimum. The proxy checks the total caller deadline
as well; event-loop stalls can delay handling, so this is not a hard real-time
latency guarantee. Remote board computation may continue after disconnection.

Inspect `GET /state` -> `predictions`. `rejected` means local overload/closed
prediction admission, not an upstream HTTP rejection. `pending` counts waiting
callers; `outstanding` includes jobs abandoned by those callers; `running`
counts jobs executing predictor code. `peak_outstanding` must never exceed the
configured limit. See [counter definitions](development/load-testing.md#prediction-work-accounting).
Do not increase the limit merely to hide rejections: measure timely prediction
coverage, tail latency and board CPU together.

Application shutdown stops prediction admission, cancels work not yet started,
waits for running jobs, and closes owned clients and trace output. Injected
predictors and upstream clients remain caller-owned. A custom synchronous
predictor must eventually return; Python cannot forcibly stop its thread.

## Board computation and prediction freshness

The HTTP board runs event ingestion, forecasting, controller planning, and metrics
state updates on one dedicated control worker. Only one unfinished control job
is admitted: overlapping `/tick` or `/directives` requests receive HTTP 503 with
`Retry-After: 1`. A cancelled caller does not release the worker. A metrics scrape
that finds it busy skips that update and retries on its next scheduled scrape.
Shutdown drains the current job before closing the worker.

After a successful tick, the board atomically publishes a version containing
session tool means, service-rate parameters, and pre-encoded snapshot JSON.
`/predict` reads the last complete version while another tick runs; `/snapshot`
does not recompute quantiles on the HTTP event loop. Controller outputs are also
serialized on the worker and cached per snapshot. Mutable registry/model/RNG
state stays with that worker; configure controllers before starting requests.

`--prediction-max-age SECONDS` sets the maximum age of a prediction view. The
default is `max(1, 3 * tick_s)`: 15 seconds with the board launcher's default
five-second tick. This is an operational allowance for cadence plus calculation
and scheduling delay, not a calibrated forecast-validity guarantee. Match it to
the actual control-loop cadence and acceptable stale-state risk. Age uses a
monotonic clock starting **before ingestion**, so calculation time counts toward
the limit. `/predict` adds `prediction_version`, `prediction_age_s`, and `stale`.
An expired view returns zero estimates with `over_budget: true`, which makes the
proxy use its existing fallback. Startup version zero contains the initial
registry projection and pooled unknown-session estimate; it expires normally if
no tick succeeds. New events become visible only after a successful tick.

Board `GET /state` reports publication age/version, served/stale/unavailable
prediction counts, control admission counters, and stage timings (`count`,
`total_s`, `max_s`, `last_s`). Stages distinguish publication read, prediction
calculation/serialization, ingestion, projection, forecast, snapshot serialization,
and directive planning/serialization. Read timing includes the brief publication
lock; no prediction waits for the control worker. Times exclude network transit
and time before the event loop dispatches the handler.

This fast path applies to the built-in `LiveBoard` prediction methods. Custom
subclasses overriding those methods retain a bounded, four-job legacy prediction
runner and their caller budget; their arbitrary internal state is not converted
into immutable prediction views. Thread isolation does not remove Python GIL
contention or provide hard real-time deadlines. Forecast/controller algorithms
and their random-number order are unchanged.

## Upstream connection reuse

The proxy's owned HTTP client retains `max(20, window)` idle upstream connections
by default, so a busy admission window larger than HTTPX's usual 20-connection
keep-alive limit does not continually discard reusable connections. Override
with `--upstream-keepalive N` (`upstream_keepalive_connections` in `ProxyConfig`);
zero disables reuse. Total connection capacity is `max(100, window, keepalive)`.
Limits are set at client construction, not dynamically when a queue window changes.
The existing 600-second transport timeout and five-second idle expiry remain.
Injected upstream clients keep their caller-selected limits and ownership.

This controls proxy-to-upstream connections, separately from the load generator's
`client_keepalive_connections` for client-to-proxy traffic and from board prediction
transport. It does not change admission, holds, predictions or upstream payloads.
