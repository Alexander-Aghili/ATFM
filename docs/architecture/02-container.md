# Level 2 — Container — ATFM

> **Reading this view:** diagrams describe architectural intent. The text below identifies
> current integration boundaries; [implementation status](../status.md) separates shipped
> paths from hardware evidence, and [operations](../operations.md) gives runnable commands.

> **Diagram type**: Container
> **Scope**: the independently deployable parts of ATFM and how they exchange data with each other, the agent harness and the Dynamo pool.
> **Audience**: the engineering team building and operating ATFM.
> **Status**: draft v1.1, generated from the architecture spec (2026-09-23); diagrams moved from Mermaid C4 to reladraw on 2026-09-27; LMCache added as the placement actuator (D3 amendment); pending founder validation.

## Overview

ATFM has two paths. The request path is one hop: the harness's LLM calls go through the harness proxy, which decides when to release each call and with what priority tier, then forwards it to the Dynamo frontend unchanged apart from hints and headers. The state path is everything else: the sidecar inside the harness emits tool progress, the proxy emits call events, worker metrics are scraped, and the event bus carries all of it to the demand board, which publishes forecasts that the controllers turn into hold directives back to the proxy. Board predictions are requested under a time budget; telemetry and controller failures are intended to fall back without stopping the request path.

The simulator and evaluation tools are offline programs that reuse the same predictor, policy and controller classes against a closed-loop fleet model, while keeping simulator-specific worker state separate from live transport and metrics. Shared schemas support comparison; simulated and live observations are not identical measurements.

## Diagram

```reladraw
// ATFM container diagram (v1.1).
// Row 1 is the request path: harness -> proxy -> Dynamo. Row 2 is the event and forecast cycle,
// left to right: sidecar -> bus -> board -> controllers, with the board's predictions and the
// controllers' hold directives returning up to the proxy. Row 3: the store on the left, the
// offline simulator under the controllers, the operator at the far right.
default edge  line: (path: square, corners: rounded)
style person   fill: #1f3a5f  border: #3b6ea8  text: (color: #e8f0fa)
style external fill: #2a2a2a  border: #6a6a6a  text: (color: #dcdcdc)
style svc      fill: #2d1f4f  border: #7a5cc0  text: (color: #efe8ff)
style store    fill: #142814  border: #486544  text: (color: #e4f2e4)  badge: database
style dim      text: (color: #9a9a9a)

node proxy   "Harness proxy / [dim]FastAPI, OpenAI endpoint; index, global window, holds, tiers[/dim]" (wrap: 34)  style: svc  gap: wide
node harness "Agent harness / [dim]mini-SWE-agent, OpenHands, Harbor[/dim]" (wrap: 28)  style: external  left of proxy
node dynamo  "NVIDIA Dynamo pool / [dim]frontend + KV router + workers + planner[/dim]" (wrap: 30)  style: external  right of proxy
node lmcache "LMCache controller / [dim]pin, move, lookup by prefix tokens; CPU and disk tiers[/dim]" (wrap: 30)  style: external  right of control (gap: wide)  below dynamo

node sidecar "Tool-runtime sidecar / [dim]in-process; wraps tool subprocesses; emits progress[/dim]" (wrap: 30)  style: svc  below harness  gap: wide
node bus     "Event bus / [dim]Redis Streams (deploy); in-memory or JSONL (dev)[/dim]" (wrap: 28)  style: svc  right of sidecar
node board   "Demand board / [dim]session registry; predictors B0..M3; Monte Carlo forecast every 5 s[/dim]" (wrap: 32)  style: svc  right of bus
node control "Controllers / [dim]GDP planner, touch and tier placement, replica floor, control loop[/dim]" (wrap: 30)  style: svc  right of board

node store    "Trace and run store / [dim]trace table, snapshots, run metrics (parquet + JSON)[/dim]" (wrap: 28)  style: store  below bus  gap: wide
node sim      "Simulator and evaluation / [dim]closed-loop fleet model; forecast and serving metrics[/dim]" (wrap: 30)  style: svc  below control  gap: wide
node operator "Platform operator" style: person  right of sim

edge harness -> proxy    "chat completions (OpenAI API)"             from: right   to: left
edge proxy -> dynamo     "admitted calls + tier / OSL hints"         from: right   to: left
edge dynamo -> proxy     "KV blocks, queue depth (Prometheus)"       from: left    to: right
edge harness -> sidecar  "execute(action)"                           from: bottom  to: top
edge sidecar -> proxy    "launch gate" (size: small)   from: left  to: top  left of harness
edge sidecar -> bus      "tool.start / progress / data / end"        from: right   to: left
edge proxy -> bus        "llm.request / first_token / done" (size: small)  from: bottom  to: top
edge bus -> board        "events"                                    from: right   to: left
edge board -> proxy      "predictions for / pending calls" (size: small)  from: top     to: bottom
edge board -> control    "ForecastSnapshot per horizon"              from: right   to: left
edge control -> proxy    "HoldDirective" (size: small)               from: top     to: bottom
edge control -> lmcache  "pin / move (placement)" (size: small)       from: right   to: left
edge lmcache -> dynamo   "KV tiers under the workers" (size: small)   from: top     to: bottom
edge bus -> store        "JSONL events -> trace table"               from: bottom  to: top
edge board -> store      "forecast snapshots"                        from: bottom  to: right
edge sim -> store        "reads traces; writes run metrics"          from: left    to: right
edge operator -> sim     "runs experiments (CLI, YAML)"              from: left    to: right
edge operator -> proxy   "class weights, budgets, SLOs (YAML)"       from: top     to: top   right of dynamo and control
```

Source: `02-container.reladraw` (rendered with `npx reladraw 02-container.reladraw -o 02-container.svg`). Rendered copy: [02-container.svg](./02-container.svg).

## Legend

- **Container**: an independently deployable process, library loaded into another process, or data store.
- **Queue container**: the event bus.
- **Database container**: file-based trace and run store.
- **External system**: the agent harness and the Dynamo pool (out of scope).
- No colors, icons or line styles carry meaning in this diagram.

## Elements

| Element | Type | Technology | Responsibility |
|---|---|---|---|
| Tool-runtime sidecar | Container (library, in-process with the harness) | Python 3.12 | Wraps instrumented tool subprocesses; returns output byte-for-byte; streams the same output through parsers (pytest, build, dbt, training, fallback) that emit progress and data events; consults the launch gate for deferrable tools and spawns. Fail-open. |
| Harness proxy | Container | Python 3.12, FastAPI/ASGI | OpenAI-compatible endpoint. Classifies sessions, asks the board for per-request predictions, computes the index and its objective, keeps a global admission window, holds deferrable calls per controller directives, writes `nvext.agent_hints` tiers, logs every call. Fail-open to default hints. |
| Event bus | Queue | Redis Streams in deployment; in-memory or JSONL file for tests and replay | Transports session, tool and worker events off the blocking path. |
| Demand board | Container | Python 3.12, numpy | Session registry and the predictor ladder B0..M3 behind one interface; publishes demand samples per horizon and class and serves per-request predictions. The live CLI uses M2 with 256 draws; HTTP ticks are driven by POST /tick, normally from the control loop. |
| Controllers | Container (co-located with the board in v1) | Python 3.12 | Ground delay program: 30 s slots over 15 min, chance constraints on KV and prefill capacity evaluated on the samples, greedy ration-by-schedule (heuristic), hard caps, per-tenant fairness accounting. Tier recommendations are logged by default and can be sent through the optional LMCache actuator; replica proposals remain advisory in the launch scripts. Slot and horizon settings are configurable. |
| Trace and run store | Database (files) | Parquet, JSON | Canonical trace table (one row per LLM call plus the tool phase it launched), forecast snapshots, run directories with config, seed and metrics. |
| Simulator and evaluation | Container (offline CLI) | Python 3.12 | Closed-loop discrete-event fleet model with workers, tiers, router and engine scheduler; reuses the deployment predictor, policy and controller classes; forecast metrics (CRPS, pinball, coverage, surge lead time) and serving metrics with paired bootstrap CIs. |
| Agent harness | External system | mini-SWE-agent, OpenHands, Harbor | Runs the agent loop; executes tools via the sidecar; sends LLM calls to the proxy. |
| NVIDIA Dynamo pool | External system | Dynamo v1.5 frontend, KV router, vLLM/SGLang or Mocker workers, planner | Serves admitted calls; honours priority tiers; exposes Prometheus metrics. |
| Platform operator | Person | — | Configures budgets and SLOs; runs experiments; reads results. |

## Key relationships

| From | To | Intent | Protocol / Technology |
|---|---|---|---|
| Agent harness | Tool-runtime sidecar | Executes each tool through | Python call `execute(action, cwd, timeout)` |
| Agent harness | Harness proxy | Sends chat completions to | HTTP/JSON, OpenAI API |
| Harness proxy | Dynamo pool | Forwards admitted calls with priority tier and OSL hints to | HTTP/JSON, `nvext.agent_hints` |
| Tool-runtime sidecar | Event bus | Publishes tool.start / progress / data / end events to | Bus protocol; JSONL in launch scripts, Redis Streams when composed explicitly |
| Tool-runtime sidecar | Harness proxy | Asks the launch gate whether a deferrable tool or spawn may start | HTTP/JSON |
| Harness proxy | Event bus | Publishes llm.request / first_token / done events to | Bus protocol; JSONL in launch scripts, Redis Streams when composed explicitly |
| Demand board metrics scraper | Dynamo pool | Scrapes worker capacity and KV metrics from the configured URL | HTTP/Prometheus |
| Event bus | Demand board | Delivers session, tool and worker events to | JSONL reader in launch scripts; optional Redis Streams consumer |
| Demand board | Harness proxy | Returns expected service time and expected next-tool duration for pending calls to | HTTP/JSON, or in-process in v1 |
| Demand board | Controllers | Publishes ForecastSnapshot samples per horizon to | in-process |
| Controllers | Harness proxy | Issues HoldDirectives with release-not-before times to | HTTP/JSON |
| Harness proxy | Trace and run store | Appends per-call records to | JSONL; canonical parquet conversion is a separate path |
| Demand board | Trace and run store | Writes snapshots in file mode; serves them over HTTP in service mode | JSONL or HTTP/JSON |
| Simulator and evaluation | Trace and run store | Reads trace tables from and writes run metrics to | Parquet, JSON |
| Platform operator | Simulator and evaluation | Runs experiments and reads metrics with | CLI, YAML configs |
| Platform operator | Harness proxy | Configures admission and service settings | Launcher CLI options; full ProxyConfig in Python |

## Notable architectural decisions

- The proxy holds the backlog on purpose (global window) because ordering only sticks at the layer that holds the backlog; a per-worker window is not enforceable above the router (D1, spec 4.2).
- Priority hints carry stable tiers derived from class and deadline, never queue ranks, so a released request's hint cannot go stale (spec 4.3).
- The sidecar is a library, not a separate process, so the tool result path stays in the harness process and is byte-for-byte unchanged (D9).
- The board and controllers are one process in v1; the bus lets them be split later without changing any producer.
- The D3 amendment adds LMCache pin/move actuation and keep-alive touches as a fallback. Simulation accounts for placement and hold costs; equivalent real-worker measurements require usable worker telemetry and separate validation.
- Policy results come only from closed-loop runs (simulator or live agents); trace replay scores forecasts and calibrates the simulator (D12).

## Assumptions

- Redis Streams is available as a library bus. The shipped launch scripts use JSONL; Redis is not selected through a CLI flag.
- The configured worker-metrics scraper runs with the board service. A frontend page without worker KV metrics is insufficient for capacity-based control.
- The controllers run inside the demand board process in v1; the diagram shows them as a separate container because they are a separate package with their own lifecycle.
- The trace and run store is local files; no database is planned.

## Links to other levels

- ↑ [System Context](./01-context.md).
- Component views are not drawn: the largest container (demand board) has six components (registry, predictor ladder, forecaster, exogenous model, snapshot publisher, per-request API) documented in spec section 5.
- See also: [architecture spec v1.1](../superpowers/specs/2026-09-22-atfm-architecture-design.md).
