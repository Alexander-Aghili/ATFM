# ATFM: Agent Traffic Flow Management. Architecture and process design

Date: 2026-09-22. Status: v1, approved for implementation planning by the founder's instruction to proceed.
Companion documents: `detail.md` (research plan), `docs/research/*.md` (platform, benchmarks, demonstration ladder).

## 0. Decisions made in this document

| # | Decision | Chosen | Rejected | Why |
|---|---|---|---|---|
| D1 | Where the class-aware queue lives | Hybrid: hold queue in the harness proxy above Dynamo, with the index rank encoded into `nvext.agent_hints.priority` so the Dynamo router and the engine agree on order for everything released | Rust `request_classifier` plugin inside the router; engine-only priority | Only the proxy sees a session before its next LLM call exists (from sidecar events). Ground delay and tool-launch gating are impossible below it. Python only, testable against Mocker on a laptop. Rust plugin deferred as an optional integration. |
| D2 | Simulation strategy | Own Python discrete-event fleet simulator with pluggable policies, validated against Dynamo Mocker (CPU) and then paired real-GPU cells | DynoSim only | DynoSim runs on a virtual clock our real-time proxy cannot join (manual-clock API is an unmerged PR). We need policy sweeps over thousands of configurations and a controllable tool-duration tail. DynoSim stays as the fidelity cross-check for the Dynamo router and engine model. |
| D3 | KV tiering mechanism | Engine-native tiers through adapters: vLLM CPU offload, SGLang HiCache with `nvext.cache_control` TTL pinning, LMCache or KVCR later | KVBM | KVBM is deprecated in Dynamo v1.5.0 and removed in v1.6.0. |
| D4 | Trace format | One canonical event schema (detail.md 8.2) with adapters in (TraceLab, AgentX, sidecar logs) and out (DynoSim agentic Mooncake, AIPerf) | Per-source formats | Every experiment, real or simulated, reads the same records. |
| D5 | Forecast representation | Monte Carlo samples of fleet demand per horizon, with a shared latent backend factor | Closed-form moments | Fan-out, correlated slowdowns and heavy tails make moments misleading; samples feed CRPS, pinball loss and chance constraints directly. |
| D6 | Predictor interface | Every rung of the model ladder B0 to M3 implements one interface: given a session's observable state, return a distribution over time-to-next-LLM-call and next-call size | Separate codepaths per model | Gains are attributable only if the ladder differs by information used, not by code. |
| D7 | Language and runtime | Python 3.12, uv, pydantic models, FastAPI for proxy and board, numpy/scipy, pytest. No Rust in v1 | Rust plugin, Go sidecar | Research velocity; all Dynamo extension points we use are Python or HTTP/gRPC. |
| D8 | Hardware ladder | L0 laptop simulation, L1 laptop functional, L2 1xH100, L3 2xH100, L4 2xH100 + Mocker padding | 8xH100 nodes | Literature norm is 1 to 4 GPUs; see demonstration ladder. |
| D9 | Sidecar contract | Result path untouched; a separate state path emits `{call_id, phase, completed, total, ts}` and data events; sidecar is fail-open | Modifying tool outputs, blocking on the bus | Agents must see identical tool results with and without the sidecar. |
| D10 | Control loop safety | Every controller is fail-open: proxy forwards with default hints if the board is unreachable; ground delays have hard caps; pre-staging is advisory | Fail-closed | A forecasting bug must never stall a fleet. |

## 1. Goal and non-goals

Goal: a control layer for agent fleets on a shared Dynamo-served GPU pool that (a) forecasts fleet KV and prefill demand from the live state of in-flight sessions, (b) orders and admits LLM calls by a class-aware index, and (c) uses the forecast to delay deferrable work before it consumes resources, to pre-stage KV across memory tiers, and to set replica floors. The same code runs against the simulator, against Dynamo Mocker, and against real GPUs.

Non-goals for v1: multi-node KV transfer, a Rust router plugin, hosted-API token quotas, customer-facing packaging, strategic (weeks-scale) capacity planning.

## 2. System context

```
 agent harnesses (mini-swe-agent, OpenHands, Harbor)        Dynamo pool
 +---------------------------+                             +--------------------------+
 | harness loop              |  OpenAI-compatible HTTP     | frontend -> KV router    |
 |   LLM call ---------------+---> [ATFM proxy] ---------->|   -> workers (vLLM/SGLang|
 |   tool exec --[sidecar]---+--+                          |      or Mocker)          |
 |   subagent spawn --[gate]-+  |  events                  |   planner <- [plugin]    |
 +---------------------------+  v                          +--------------------------+
                          [event bus] ---> [demand board] ---> forecasts ---> [controllers]
                                                                              GDP / prestage / scale
```

Components and ownership:

| Component | Package | Role | Depends on |
|---|---|---|---|
| Schema | `atfm.schema` | Pydantic models for events, trace records, forecasts, directives | none |
| Sidecar | `atfm.sidecar` | Tool wrapper + progress/data parsers + harness adapters | schema, bus |
| Bus | `atfm.bus` | Event transport: in-memory (tests), JSONL file (replay), Redis Streams (deploy) | schema |
| Demand board | `atfm.board` | Session registry, predictors (ladder), Monte Carlo fleet forecaster, per-request predictions | schema, bus |
| Proxy | `atfm.proxy` | OpenAI-compatible ASGI server: classify, index, hold queue, hints, forward, log | schema, bus, board client |
| Controllers | `atfm.control` | Ground delay program, pre-staging, planner plugin | board, proxy, dynamo adapters |
| Dynamo adapters | `atfm.dynamo` | Hints encoding, HiCache pinning, worker metrics scrape, Mocker launcher, planner gRPC stub | none (HTTP/gRPC) |
| Simulator | `atfm.sim` | Discrete-event fleet model with pluggable policies; same interfaces as proxy/board | schema, board predictors |
| Traces | `atfm.traces` | Adapters in/out, synthetic session generator, perturbation injector | schema |
| Evaluation | `atfm.eval` | Forecast and serving metrics, paired comparisons, bootstrap CIs, reports | schema |
| Runners | `scripts/` | Trace generation, experiment sweeps, GPU runs | all |

## 3. Data model

### 3.1 Identity

- `session_id` (string, harness-provided or minted by the proxy from `x-atfm-session`), `parent_session_id` for fan-out.
- `tenant`, `class` in {interactive, background}, `deadline` (optional absolute time), `deferrable` (bool, derived from class unless overridden).
- `turn_index` increments per LLM call in a session.

### 3.2 Events (bus records; one pydantic model per kind, discriminated by `kind`)

| kind | fields | emitter |
|---|---|---|
| `session.start` | session_id, parent_session_id, tenant, class, deadline, t | proxy (first request) or sidecar |
| `llm.request` | session_id, turn_index, request_id, t, isl, predicted_osl, hints, held_since | proxy |
| `llm.first_token`, `llm.done` | request_id, t, osl, worker_id, prefix_hit_tokens | proxy |
| `tool.start` | session_id, turn_index, call_id, tool_name, args_hash, backend_id, t | sidecar |
| `tool.progress` | call_id, t, completed, total, phase | sidecar |
| `tool.data` | call_id, t, metric, value | sidecar |
| `tool.end` | call_id, t, exit_status, output_chars | sidecar |
| `spawn.request`, `spawn.granted` | parent_session_id, child_session_id, t | sidecar gate / proxy |
| `worker.metrics` | worker_id, t, kv_blocks_used, kv_blocks_total, queue_depth, tier_blocks{hbm,dram,ssd} | dynamo adapter |
| `perturbation` | t, kind, backend_id, active | runner |

### 3.3 Trace record (flattened, one row per LLM call; detail.md 8.2)

Every adapter produces this table; every simulator run emits it. Columns: session_id, parent_session_id, class, tenant, turn_index, t_request, t_first_token, t_last_token, isl, osl, prefix_hit_tokens, kv_blocks_by_tier, worker_id, tool_name, tool_args_hash, t_tool_start, t_tool_end, tool_exit_status, progress_events (list), data_events (list), backend_id, spawned_children, perturbation_flag, plus `source` in {tracelab, agentx, sidecar, sim}.

### 3.4 Forecast snapshot

`ForecastSnapshot{t, horizons[], targets{kv_blocks, prefill_tps}[class], samples[h][n], quantiles[h]{q10,q50,q90,q95}, endogenous_fraction[h], model_id}`. Samples are kept (n = 512 default) so downstream chance constraints and scoring use the same draws.

### 3.5 Directives (controller outputs)

- `HoldDirective{session_id, release_not_before, reason}` (ground delay; proxy enforces).
- `TierDirective{session_id, target_tier in {hbm,dram,ssd,drop}, by_time, ttl_s}` (pre-staging; Dynamo adapter translates: SGLang `cache_control` pin with TTL, `speculative_prefill`, or no-op with logging when unsupported).
- `ReplicaFloor{component, at_least, until}` (planner plugin returns `OverrideType.AT_LEAST`).

## 4. Request lifecycle and the queue decision (D1)

### 4.1 States of a session turn

```
 TOOL_RUNNING --(tool.end)--> LLM_PENDING --(proxy releases)--> LLM_QUEUED --(engine admits)--> LLM_RUNNING --(done)--> TOOL_RUNNING
      ^  elapsed a, progress w/W            held in proxy hold queue     Dynamo router + engine queue    decode; preemptible
      |  survival S(a), filter posterior     ground delay applies here    priority hint orders here
      +---------------------------------------------------------------------------------------------------(spawn)--> child session
```

The demand board's prediction target is the time from "now" until `LLM_PENDING` for every session in `TOOL_RUNNING`, plus the size of that call. The proxy's index applies at `LLM_PENDING`.

### 4.2 The three queues

| Layer | Holds a backlog when | Sees | Can do |
|---|---|---|---|
| ATFM proxy hold queue | always, by design (per-worker admission window) | class, tenant, deadline, sidecar progress, board predictions, forecast | order, hold (ground delay), gate spawns, set hints |
| Dynamo router pending heap | workers saturated (active-block tracking) | per-worker prefix overlap and load, `priority`, `strict_priority` | order by (tier, effective arrival), pick worker |
| Engine scheduler (vLLM/SGLang) | KV blocks or batch slots exhausted | its own waiting and running sets, priority value | admit order, preempt running lower-priority decodes, evict KV (SGLang priority eviction) |

Ordering sticks at the layer that holds the backlog. The proxy therefore keeps at most `W_k` requests in flight per worker (default: max batch size + small slack, measured from worker metrics) and orders everything beyond that itself. The index rank of each released request is encoded into `priority` so that whatever backlog forms below still respects the proxy's order, and vLLM's priority preemption acts on running decodes.

### 4.3 The index

For pending call i: `pi_i = w(class_i) * (1 + beta * E[T_tool_next,i]) / E[S_i]`, with an interactive slack override: if `slack_i = deadline_i - now - E[S_i] < slack_threshold`, the call is placed in the strict tier above all background work. `E[S_i]` comes from the board's service-time predictor (ISL, predicted OSL per tool type, current batch state). `E[T_tool_next,i]` comes from the per-tool duration model conditioned on the tool the harness is expected to call next (from the session's tool-transition history; default: marginal).

Priority encoding: `priority = round(clip(rank_scaled, 0, 1000))` where rank is the position in the proxy's ordered hold queue at release time; interactive-slack overrides use `strict_priority = 1`.

## 5. Demand board

### 5.1 Session registry

Keyed by session_id. For each: class, tenant, deadline, current state (4.1), current tool (name, backend_id, start time, latest progress and data events), history of (tool_name, duration) pairs and of (isl, osl) per turn, spawn count, last context length (for K_i).

### 5.2 Predictor interface (D6)

```
class Predictor(Protocol):
    def fit(self, train: TraceTable) -> None
    def resumption(self, s: SessionState, now: float, rng) -> Samples   # time until next LLM call, n draws
    def next_call(self, s: SessionState, rng) -> Samples                # (isl, osl, kv_blocks) draws
    def spawn(self, s: SessionState, rng) -> Samples                    # number of children over horizon
```

Ladder:

| id | resumption uses | how |
|---|---|---|
| B0 | nothing (current = next) | demand at t+h equals demand at t |
| B1 | aggregate time series | Dynamo Planner predictors (constant, ARIMA, Kalman) on the fleet KV series; wrapped as a Predictor that ignores sessions |
| B2 | per-tool historical duration distribution | empirical or log-normal mixture per tool_name, sampled fresh (ignores elapsed) |
| M1 | B2 + elapsed time | conditional survival: sample from the empirical distribution truncated to > a |
| M2 | M1 + progress and data events | Bayesian filter over rate r (Gamma prior updated by (w, dt) increments) and phase; remaining = (W - w)/r plus end-phase residual model; data events feed early-stop probability |
| M3 | M2 + latent backend factor | per backend_id a shared log-speed factor z_b ~ N(mu_b, sigma_b) updated from all sessions on that backend; sessions on the same backend share draws |

### 5.3 Fleet forecaster

At each tick (default 5 s) and for each horizon h in {10 s, 30 s, 2 min, 5 min, 15 min}: for n draws, for each session draw resumption time R and next-call size; include children by drawing spawn counts and their calls recursively up to depth 2; add exogenous arrivals from a nonhomogeneous Poisson process with rate estimated per class from the last 30 min. Aggregate KV blocks and prefill tokens per class. Emit `ForecastSnapshot`. Cost: O(sessions x n) per horizon; vectorized with numpy.

### 5.4 Per-request predictions

The proxy asks the board for `E[S_i]`, `E[T_tool_next,i]` and predicted OSL for each pending call via an in-process call (same process in v1) or HTTP (deploy). Cached per session for 1 s.

## 6. Controllers

### 6.1 Real-time: index scheduling

Lives in the proxy (4.2, 4.3). Preemption of long background decodes is delegated to the engine through priority; the proxy additionally may cancel and re-issue a background request when an interactive surge is forecast and the engine does not support preemption (configurable, off by default).

### 6.2 Tactical: ground delay program (GDP)

Runs every 30 s on the latest snapshot. Slots tau of 30 s over the next 15 min. Inputs: interactive demand samples I_tau (from the forecast), capacity C_tau (KV blocks, from worker metrics and any pending replica changes), deferrable sessions with forecast resumption slot tau_i^0, delay cost c_i (deadline-based: cost grows as slack shrinks; default linear), K_i.

Solve: assign each deferrable session a release slot minimizing sum c_i (tau - tau_i^0)^+ subject to P(I_tau + sum K_i x_i,tau <= C_tau) >= 1 - eps per slot, evaluated on the samples (sample-average approximation). v1 solver: greedy ration-by-schedule (sessions ordered by tau_i^0; each takes the earliest slot whose chance constraint still holds on the samples), which is exact for the equity ordering and fast. An MILP (OR-Tools CP-SAT) is a later option behind the same interface. Output: HoldDirectives with a hard cap (default 10 min) and per-tenant fairness accounting (max imposed delay).

### 6.3 Tactical: pre-staging

For each session in TOOL_RUNNING with KV resident: from the resumption samples compute q10 and q90 of R. Policy: if q90 < T_hot keep in HBM (pin with TTL = q90 + margin on SGLang); if q10 > T_cold demote to DRAM/SSD; promote back (or issue `speculative_prefill`) when q10 falls below the tier's transfer lead time. Thresholds are per-tier transfer costs measured at L2. When the engine offers no directive, the controller logs the decision so the simulator and the real run are comparable.

### 6.4 Tactical: replica floor

A Planner PROPOSE plugin (gRPC, `OverrideType.AT_LEAST`) returns the replica count needed so that forecast q90 total KV demand at horizon = scale-out lead time fits capacity. v1 targets the `virtual` connector (L0/L4) and Kubernetes at L3 only if time permits.

## 7. Sidecar

- One wrapper function `run_tool(cmd, ctx) -> result` that starts the subprocess, streams stdout/stderr to the parser chain, forwards the original output unchanged to the harness, and emits `tool.start/progress/data/end`.
- Parser chain: pytest (`collected N items`, per-test PASSED/FAILED lines, session phases collect/run/teardown), generic build (`[n/N]`, percentage patterns), dbt/Spark (rows processed, stage k/N), training monitors (loss, step), fallback (bytes and line rate only, "weak signal").
- Harness adapters: mini-swe-agent (`Environment.execute` override), OpenHands (tool executor wrapper on `TerminalTool`), Harbor (`BaseAgent.environment.exec` wrapper).
- Spawn gate: `gate(parent_session_id) -> allowed_at` consulted before subagent creation when the class is deferrable; fail-open.
- Signal coverage report: share of tool time with strong, weak, or no signal (Phase 0 checkpoint).

## 8. Simulator

Discrete-event, own heap-based loop (no SimPy dependency), deterministic under a seed.

Entities: sessions (generated from a workload spec or replayed from a trace table), workers (KV block capacity, max batch, prefill tokens/s, decode tokens/s per batch size from a profile table, tiers with capacities and transfer bandwidths), router (prefix-affinity or round-robin), engine scheduler (FCFS or priority with preemption), proxy policy (same `Policy` interface as the real proxy), controllers (same classes as deploy, driven by simulated time).

Workload spec: per class, session arrival process, turns per session, ISL growth per turn, OSL distribution, tool-duration distribution per tool name with a controllable tail weight (share of calls over 60 s), fan-out probability and children per spawn, backend assignment and slowdown schedule (perturbations), think-time distribution for interactive sessions.

Outputs: the trace table (3.3), worker metrics series, forecast snapshots if the board is attached, and the serving metrics (12 in detail.md).

Fidelity: L2/L3 paired cells replay the same trace on the simulator and on real hardware; report mean and p99 JCT and TTFT-after-tool error; target within 6%. Mocker (CPU) is the intermediate check for the Dynamo router behaviour.

## 9. Evaluation

`atfm.eval.forecast`: CRPS from samples, pinball loss at q90/q95, coverage of 80/90% bands, surge lead time and false-alarm rate against a capacity line, endogenous fraction, all sliced by class, horizon, and perturbation windows; time-block train/test splits.

`atfm.eval.serving`: TTFT after tool return p50/p95/p99, SLO attainment, background JCT and deadline hit rate, tasks/hour, max imposed delay per tenant, GPU-hours, recomputed prefill tokens, KV transfer bytes by tier, cost per task; paired comparisons on identical replays; bootstrap CIs by session or time block; Pareto frontier plots.

Experiment definitions are YAML files under `experiments/`; every run writes a directory with config, seed, git SHA, inputs hash, metrics JSON, and the trace table.

## 10. Error handling and safety

- Proxy: board timeout 50 ms then default hints; hold queue has a max hold (10 min) and a max size after which it forwards FCFS and raises an alarm; requests carry the original body unchanged except `nvext` and headers.
- Sidecar: parser exceptions are caught per line; the result path is a pass-through pipe; bus write failures are dropped with a counter.
- Board: unknown sessions are created on first sight; stale sessions expire after 2 h without events; predictors fall back down the ladder on missing features (M2 with no progress events behaves as M1).
- Controllers: every directive has an expiry; on restart nothing is enforced until the first fresh snapshot.

## 11. Testing strategy

- Unit: schema round-trips, each predictor's sampling properties (M1 truncation, M2 posterior updates on synthetic progress streams), GDP chance constraint on hand-built samples, index ordering, hold-queue windows.
- Property: GDP never violates the cap or the constraint on its own samples; sidecar output equals subprocess output byte-for-byte.
- Golden: small synthetic workloads replayed through the simulator produce fixed metrics under a seed.
- Integration (L1): proxy + Mocker + a scripted harness; events flow to the board; forecast snapshots appear; hints are present in Mocker logs.
- Hardware (L2/L3): paired sim/real cells; priority and preemption verified with vLLM `--scheduling-policy priority`.

## 12. Repository layout

```
Company/
  pyproject.toml              # package atfm, uv-managed, python 3.12
  src/atfm/{schema,bus,sidecar,board,proxy,control,dynamo,sim,traces,eval}/
  tests/
  scripts/                    # trace generation, sweeps, GPU runs
  experiments/                # YAML experiment definitions
  docs/research/, docs/superpowers/specs/, docs/superpowers/plans/
  detail.md
```

## 13. Delivery order (maps to the demonstration ladder)

1. L0 core: schema, traces (TraceLab adapter, synthetic generator), board predictors B0/B2/M1/M2, forecaster, forecast metrics. First result: H1 on TraceLab and on synthetic fleets with a regime map.
2. L0 policies: simulator with P0, P1, P3, P4, oracle variants; serving metrics; sweeps.
3. L1 plumbing: proxy with hold queue and hints against Mocker; sidecar for mini-swe-agent with the pytest parser; bus (in-memory and JSONL); coverage report.
4. L2: Dynamo adapters against vLLM on one H100; simulator calibration; Continuum baseline via SGLang pinning.
5. L3/L4: pre-staging, GDP on real hardware, planner plugin with virtual connector, headline scenarios.
