# ATFM: Agent Traffic Flow Management. Architecture and process design

> **Design specification with dated amendments.** Current launcher behavior and validation
> boundaries are summarized in [implementation status](../../status.md).
> Use [operations](../../operations.md) for executable setup instructions.

Date: 2026-09-22, revised 2026-09-23 (v1.1). Status: approved for implementation.
Companion documents: `detail.md` (research plan), `docs/research/*.md` (platform, benchmarks, demonstration ladder, first H1 results).

### v1.1 changes (external design review, 2026-09-23)

- Central claim split into three hypotheses (D11) so the sidecar earns its place independently of the demand board.
- Admission delay restated: holding has a cost, the proxy can only act on requests that have arrived, delaying earlier needs an explicit launch gate, delay budgets per class (6.2).
- Global admission window first; per-worker windows via proxy-side worker selection are a separate integration project (D1, 4.2).
- Stable priority tiers in `nvext.agent_hints`, fine ordering stays inside the proxy (4.3).
- Index objective stated; the unblocking term is a heuristic to be ablated (4.3).
- GDP is a heuristic over KV and prefill capacity, not "exact"; per-slot chance constraints do not bound overflow over the horizon (6.2).
- Tier placement and replica floors are future work with no shipped dependency; the cost of holding is measured, not assumed (D3, 6.3, 6.4).
- Policy claims come only from closed-loop runs; trace replay scores forecasts and calibrates the simulator (D12, 8, 9).
- Native baselines include Dynamo priority scheduling and the experimental ThunderAgent Program Scheduler where it runs (9).

## 0. Decisions made in this document

| # | Decision | Chosen | Rejected | Why |
|---|---|---|---|---|
| D1 | Where the class-aware queue lives | Hybrid: hold queue in the harness proxy above Dynamo with a **global** admission window, and a stable priority **tier** (not a rank) written into `nvext.agent_hints` so the router and engine keep class order for everything released | Rust `request_classifier` plugin inside the router; engine-only priority; proxy-side worker selection through the Python KvRouter bindings (a distinct integration project: reservations, forwarding, streaming, cleanup) | Only the proxy sees a session before its next LLM call exists (from sidecar events). Python only, testable against Mocker on a laptop. Per-worker windows are not enforceable above the router because the proxy does not know which worker Dynamo will pick. |
| D2 | Simulation strategy | Own Python discrete-event fleet simulator with pluggable policies, validated against Dynamo Mocker (CPU) and then paired real-GPU cells | DynoSim only | DynoSim runs on a virtual clock our real-time proxy cannot join (manual-clock API is an unmerged PR). We need policy sweeps over thousands of configurations and a controllable tool-duration tail. DynoSim stays as the fidelity cross-check for the Dynamo router and engine model. |
| D3 | KV tiering mechanism | **Amended 2026-09-27.** LMCache's controller exposes per-prefix `pin`, `move`, `lookup` and `clear` keyed by (instance, storage location, token ids), and Dynamo ships an LMCache integration. Placement decisions (which sessions' KV stays on the GPU tier, which is parked on CPU or disk) are therefore executed through LMCache (`atfm.control.lmcache.LMCacheActuator`), with the keep-alive touch as the fallback for engines without a cache layer. kvcached (per-model elastic GPU memory) is orthogonal; Dynamo KVBM is the alternative substrate. The original v1.1 text (no shipped API pins or demotes one session's KV; measure the KV side effects of holding instead) described the state before this amendment | KVBM as the only substrate; treating the keep-alive touch as the deployable form (the simulator shows its slot cost consumes the gain when interactive load saturates the window) | The forecast is our layer; the cache is theirs. Pin and move have no batch-slot cost, and the H100 study runs on vLLM plus LMCache so the result is measured on the mechanism people deploy. Caveat: the pin API is documented for LMCache's in-process mode, now deprecated in favour of multi-process mode; the actuator's endpoint shapes are configurable and must be verified against the MP controller before the study |
| D4 | Trace format | One canonical event schema (detail.md 8.2) with adapters in (TraceLab, AgentX, sidecar logs) and out (DynoSim agentic Mooncake, AIPerf) | Per-source formats | Every experiment, real or simulated, reads the same records. |
| D5 | Forecast representation | Monte Carlo samples of fleet demand per horizon, with a shared latent backend factor | Closed-form moments | Fan-out, correlated slowdowns and heavy tails make moments misleading; samples feed CRPS, pinball loss and chance constraints directly. |
| D6 | Predictor interface | Every rung of the model ladder B0 to M3 implements one interface: given a session's observable state, return a distribution over time-to-next-LLM-call and next-call size | Separate codepaths per model | Gains are attributable only if the ladder differs by information used, not by code. |
| D7 | Language and runtime | Python 3.12, uv, pydantic models, FastAPI for proxy and board, numpy/scipy, pytest. No Rust in v1 | Rust plugin, Go sidecar | Research velocity; all Dynamo extension points we use are Python or HTTP/gRPC. |
| D8 | Hardware ladder | L0 laptop simulation, L1 laptop functional, L2 1xH100, L3 2xH100, L4 2xH100 + Mocker padding | 8xH100 nodes | Literature norm is 1 to 4 GPUs; see demonstration ladder. |
| D9 | Sidecar contract | Result path untouched; a separate state path emits `{call_id, phase, completed, total, ts}` and data events; sidecar is fail-open | Modifying tool outputs, blocking on the bus | Agents must see identical tool results with and without the sidecar. |
| D10 | Control loop safety | Every controller is fail-open: proxy forwards with default hints if the board is unreachable; admission delays have hard caps; pre-staging is advisory | Fail-closed | A forecasting bug must never stall a fleet. |
| D11 | Central claim | Three separable hypotheses. **H1a (demand board):** in-flight session state (phase, elapsed time, context size) forecasts fleet demand at 30 s to 15 min better than history-based predictors. Evidence exists (first results, 2026-09-22). **H1b (sidecar):** live tool progress improves the forecast beyond elapsed time, measurable only on long-tool workloads. **H2 (controller):** a proxy that admits and orders calls on those forecasts lowers interactive latency after tool return or GPU cost at a stated backadmission delay budget, beyond native Dynamo priority scheduling and ThunderAgent. Each is tested on its own workload regime and reported with metric, horizon and split | One combined claim | A null M2-minus-M1 on sub-second tools says nothing about the sidecar; a forecast gain says nothing about a controller until a closed-loop run shows it. |
| D12 | What counts as a policy result | Only closed-loop runs: the simulator generating sessions that react to reply times, or live agents on hardware. Trace replay (TraceLab, AgentX) scores forecasts and calibrates the simulator | Policy sweeps on replayed traces | A policy changes when agents get replies, which changes when they launch tools, spawn and call again; a fixed replay cannot see that feedback. |

## 1. Goal and non-goals

Goal: a control layer for agent fleets on a shared Dynamo-served GPU pool that (a) forecasts fleet KV and prefill demand from the live state of in-flight sessions, (b) orders and admits LLM calls by a class-aware index, and (c) uses the forecast to delay deferrable work before it consumes resources, to pre-stage KV across memory tiers, and to set replica floors. The same code runs against the simulator, against Dynamo Mocker, and against real GPUs.

Non-goals for v1: multi-node KV transfer, a Rust router plugin, hosted-API token quotas, customer-facing packaging, strategic (weeks-scale) capacity planning.

### 1.1 Workload evidence from production (added 2026-09-27)

DeepSeek's sandbox platform report (DSec, arXiv 2609.22978, September 2026) measures the CPU side of the same workload ATFM manages, at production scale (one unit of about 160 CPU nodes: ~3 million sandboxes a day, ~380k concurrent, over 5,000 creations a second, jobs of up to 32k sandboxes). Three findings motivate this design directly:

- **Sessions are long-lived and mostly idle.** Median sandbox lifetime is 15 to 17 minutes with a p99 over three hours, and about 90% of sandboxes use at most 5% of their requested CPU on average; the tool-call phase is "short CPU bursts separated by periods" of waiting on the model. The sessions whose KV this design holds, evicts or pre-stages are therefore long-lived objects with sparse, bursty GPU demand, which is the regime where forecasting when a session returns is worth more than reacting when it does. This matches the heavy-tailed tool and pending-gap distributions measured on TraceLab and AgentX (section 5.2) on a fleet three orders of magnitude larger.
- **The agent loop is decoupled from the GPU pool in production.** From DeepSeek-V4.1 the rollout loop runs on the sandbox platform and calls model serving as a separate service. That is the topology this design assumes: a fleet of stateful sessions whose timing is set by tool execution elsewhere, hitting a shared serving pool. Training rollout fleets are a demand source alongside interactive coding agents.
- **Pause and resume is the operator's default lever without a forecast.** DSec preempts sandboxes with docker pause plus swap, or a microVM snapshot, and reports no resume cost. That is occupancy-triggered pausing, the behaviour the `working_set` baseline models (section 12); the simulator's first runs put it at 2.6x the background cost of demand-aware holds for the same interactive SLO, a comparison the H100 study must confirm.

DSec says nothing about inference serving, KV cache, admission control or demand forecasting; it is evidence for the workload shape and the deployment topology, not a method to adopt.

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
| Controllers | `atfm.control` | Bounded-delay admission planner, pre-staging, planner plugin | board, proxy, dynamo adapters |
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

- `HoldDirective{session_id, release_not_before, reason}` (admission delay; proxy enforces).
- `TierDirective{session_id, target_tier in {hbm,dram,ssd,drop}, by_time, ttl_s}` (pre-staging; Dynamo adapter translates: SGLang `cache_control` pin with TTL, `speculative_prefill`, or no-op with logging when unsupported).
- `ReplicaFloor{component, at_least, until}` (planner plugin returns `OverrideType.AT_LEAST`).

## 4. Request lifecycle and the queue decision (D1)

### 4.1 States of a session turn

```
 TOOL_RUNNING --(tool.end)--> LLM_PENDING --(proxy releases)--> LLM_QUEUED --(engine admits)--> LLM_RUNNING --(done)--> TOOL_RUNNING
      ^  elapsed a, progress w/W            held in proxy hold queue     Dynamo router + engine queue    decode; preemptible
      |  survival S(a), filter posterior     admission delay applies here    priority hint orders here
      +---------------------------------------------------------------------------------------------------(spawn)--> child session
```

The demand board's prediction target is the time from "now" until `LLM_PENDING` for every session in `TOOL_RUNNING`, plus the size of that call. The proxy's index applies at `LLM_PENDING`.

### 4.2 The three queues

| Layer | Holds a backlog when | Sees | Can do |
|---|---|---|---|
| ATFM proxy hold queue | always, by design (per-worker admission window) | class, tenant, deadline, sidecar progress, board predictions, forecast | order, hold (admission delay), gate spawns, set hints |
| Dynamo router pending heap | workers saturated (active-block tracking) | per-worker prefix overlap and load, `priority`, `strict_priority` | order by (tier, effective arrival), pick worker |
| Engine scheduler (vLLM/SGLang) | KV blocks or batch slots exhausted | its own waiting and running sets, priority value | admit order, preempt running lower-priority decodes, evict KV (SGLang priority eviction) |

Ordering sticks at the layer that holds the backlog. The proxy therefore keeps at most `W` requests in flight across the pool (a **global** window, default: sum of worker batch capacities plus a small slack, from worker metrics) and orders everything beyond that itself. The proxy cannot enforce a per-worker window because Dynamo picks the worker after the proxy releases the request; a per-worker window requires the proxy to own worker selection (KvRouter Python bindings or the standalone selection service), which also means owning reservations, forwarding, streaming and cleanup, and is out of scope for v1. The window is swept in the simulator; the L2 study verifies that contention actually forms at the proxy queue (Dynamo notes router priority has no effect when requests do not wait there).

### 4.3 The index, its objective, and priority encoding

Objective: minimize `sum_i w(class_i) * C_i + lambda_int * (interactive SLO misses on TTFT after tool return) + lambda_bg * (background deadline misses)`, where `C_i` is the completion time of call i, subject to a per-class delay budget (6.2). The weighted shortest-expected-processing-time rule (`w / E[S]`) is optimal for weighted completion time on one server with known sizes (the c-mu rule); the unblocking term is a critical-path heuristic borrowed from I/O-bound process scheduling and must be justified empirically against this objective.

For pending call i: `pi_i = w(class_i) * (1 + beta * E[T_tool_next,i]) / E[S_i]`, with an interactive slack override: if `slack_i = deadline_i - now - E[S_i] < slack_threshold`, the call goes to the top tier. `E[S_i]` comes from the board's service-time predictor (ISL, predicted OSL per tool type, current batch state). `E[T_tool_next,i]` comes from the per-tool duration model conditioned on the tool the harness is expected to call next (session tool-transition history; default marginal). Ablation: `beta = 0` (no unblocking term) is always run alongside.

Priority encoding: stable tiers, not ranks. `strict_priority`: 2 for interactive calls under the slack threshold, 1 for interactive, 0 for background; `priority`: a coarse per-tier bucket of the index (4 levels). Fine ordering is recomputed inside the proxy on every arrival; a released request's tier does not go stale because tiers are class and deadline facts, not queue positions. Dynamo guarantees ordering only among requests waiting in one router queue plus engine-level priority (vLLM `--scheduling-policy priority` orders waiting requests and can preempt running lower-priority ones; SGLang also evicts by priority); it does not preempt work already admitted at the router. The window keeps the backlog below the proxy small enough that this suffices.

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

### 6.2 Tactical: bounded-delay admission planner (GDP), a heuristic

What admission delay is: when a deferrable session's next call arrives at the proxy during a forecast surge, the proxy may hold it (not forward it) until a planned release time. Holding costs background completion time; it is a trade, not a free action. It is justified only against an explicit budget: each background class carries a delay budget (default 10 min per hold, and a per-session cumulative cap tied to its deadline), and the objective in 4.3 charges `w(class) * delay` for every held second.

What the proxy can and cannot do: it can only act on a request that has arrived. While a tool is running, the GDP plans provisionally (which sessions it would hold, for how long); the decision is made at arrival using the latest plan. Delaying a session before its tool completes requires a separate, explicit control: the launch gate (7), which the sidecar consults before starting a tool or spawning a child for deferrable classes. The gate controls tool launch; the hold controls LLM admission.

The plan: every 30 s on the latest snapshot, slots tau of 30 s over the next 15 min. Inputs: interactive demand samples per slot for both resources (KV blocks and prefill tokens/s), capacity per resource (from worker metrics and pending replica changes), deferrable sessions with forecast resumption slot tau_i^0, delay cost c_i, and K_i. Assign each deferrable session a provisional release slot minimizing `sum c_i (tau - tau_i^0)^+` subject to `P(I_tau + sum K_i x_i,tau <= C_tau) >= 1 - eps` per slot and per resource, evaluated on the samples. v1 solver: greedy in expected-resumption order (sessions ordered by tau_i^0, each taking the earliest slot whose constraints still hold on the samples). This is a heuristic with the Expected-resumption ordering property; it is not claimed optimal. Per-slot constraints do not bound the probability of any overflow across the horizon (the union bound gives at most 30 eps over 30 slots), so eps is chosen with that in mind and the realized overflow rate is reported.

Output: HoldDirectives with the hard cap, per-tenant fairness accounting (max imposed delay), and the KV side effects of holding measured on the backend (6.3).

### 6.3 Tactical: pre-staging (future work) and the cost of holding

Explicit placement (pin, demote, promote, speculative prefill on a resumption ETA) is deferred until one mechanism is demonstrated end to end on a real backend (D3). The design is unchanged: per-session q10/q90 of the resumption time against tier lead times. Until then the controller only logs the TierDirective it would have issued, so simulated and real runs stay comparable.

What v1 measures instead: a held session's cached prefix stays in the engine's cache subject to its eviction policy; it is not reserved capacity. The cost of a hold is therefore the combination of cache occupancy while held, evictions of other sessions' prefixes caused by it, later cache misses and recomputed prefill when it resumes, and the backadmission delay itself. All four are recorded per hold on the real backend (recomputed prefill tokens, KV hit rate, evictions, imposed delay) and reported with the policy results.

### 6.4 Tactical: replica floor (future work)

A Planner PROPOSE plugin (gRPC, `OverrideType.AT_LEAST`) returning the replica count needed so that forecast q90 demand at horizon = scale-out lead time fits capacity. Deferred until H2 has a result; it is the natural consumer of the H1a forecast gain at 5 to 15 min horizons, and it is tested with the `virtual` connector first.

## 7. Sidecar

- One wrapper function `run_tool(cmd, ctx) -> result` that starts the subprocess, streams stdout/stderr to the parser chain, forwards the original output unchanged to the harness, and emits `tool.start/progress/data/end`.
- Parser chain: pytest (`collected N items`, per-test PASSED/FAILED lines, session phases collect/run/teardown), generic build (`[n/N]`, percentage patterns), dbt/Spark (rows processed, stage k/N), training monitors (loss, step), fallback (bytes and line rate only, "weak signal").
- Harness adapters: mini-swe-agent (`Environment.execute` override), OpenHands (tool executor wrapper on `TerminalTool`), Harbor (`BaseAgent.environment.exec` wrapper).
- Launch gate: `gate(session_id, kind in {tool, spawn}) -> allowed_at` consulted before starting a tool or creating a subagent when the class is deferrable; fail-open. This is the only control that can delay work before its next LLM call exists (6.2).
- Signal coverage report: share of tool time with strong, weak, or no signal (Phase 0 checkpoint).

## 8. Simulator

Discrete-event, own heap-based loop (no SimPy dependency), deterministic under a seed.

Entities: sessions (generated from a workload spec or replayed from a trace table), workers (KV block capacity, max batch, prefill tokens/s, decode tokens/s per batch size from a profile table, tiers with capacities and transfer bandwidths), router (prefix-affinity or round-robin), engine scheduler (FCFS or priority with preemption), proxy policy (same `Policy` interface as the real proxy), controllers (same classes as deploy, driven by simulated time).

Workload spec: per class, session arrival process, turns per session, ISL growth per turn, OSL distribution, tool-duration distribution per tool name with a controllable tail weight (share of calls over 60 s), fan-out probability and children per spawn, backend assignment and slowdown schedule (perturbations), think-time distribution for interactive sessions.

Outputs: the trace table (3.3), worker metrics series, forecast snapshots if the board is attached, and the serving metrics (12 in detail.md).

Fidelity: L2/L3 paired cells replay the same trace on the simulator and on real hardware; report mean and p99 JCT and TTFT-after-tool error; target within 6%. Mocker (CPU) is the intermediate check for the Dynamo router behaviour.

## 9. Evaluation

Hypotheses are reported separately (D11), each with the metric, horizon set, split and workload named next to every number, so that "x times better" is never read without its definition.

Forecast (H1a, H1b): time-block held-out splits; TraceLab and AgentX replay for H1a; sidecar-instrumented long-tool traces (L1) for H1b, where the quantity of interest is M2 minus M1 on tools with strong progress signals.

Serving (H2): closed-loop only (D12). The first hardware study uses one backend, one model, and one control action, proxy admission, with five arms under the same closed-loop fleet and delay budgets: (1) native Dynamo tuned for queueing and engine priority, (2) proxy with class and deadline rules and no forecast, (3) proxy with the elapsed-time forecast (M1), (4) proxy with the sidecar progress forecast (M2), (5) oracle with true tool completion times. Native baselines also include the experimental ThunderAgent Program Scheduler where its configuration runs on the chosen backend: it pauses programs at tool boundaries and resumes on a working-set budget with hysteresis, so it is the closest reactive comparison. Run on both regimes: short-tool (Claude Code style) and long-tool (tests, builds, pipelines). The decisive quantity is arm 4 versus arm 3; the product comparison is arm 4 versus arms 1 and 2. Verify that contention forms at the proxy queue. Measure interactive TTFT from tool return, whole-task completion by class, deadline misses, throughput, GPU utilization, imposed delay by tenant, and the four hold costs from 6.3. Forecast scores are reported alongside as explanation, not as the result.

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

## 13. Delivery order (v1.1)

1. L0 core (done 2026-09-22): schema, TraceLab adapter, synthetic generator, predictors B0 to M3, forecaster, forecast metrics, H1 runner. H1a evidence recorded in `docs/research/2026-09-22-h1-first-results.md`.
2. L1: sidecar for mini-SWE-agent with pytest and build parsers, launch gate, bus (in-memory and JSONL), proxy with global window and tiered hints against Mocker; collect real long-tool traces on the laptop (tools run on CPU); report progress-signal coverage and an honest M2-versus-M1 forecast result (H1b).
3. L0 policies: closed-loop simulator with the five arms of section 9 and the beta ablation; calibration hooks.
4. L2: the factored closed-loop serving experiment on one H100, one backend, one model, both regimes, native Dynamo and ThunderAgent where operable (H2). Hold costs measured on the backend.
5. Later, only if H2 holds: tier placement with a demonstrated mechanism, replica floor with the virtual connector, two-worker scenarios.

### 13.1 Status at 2026-09-27

Items 1 to 3 are built and evidenced (`docs/research/`). The simulator's H2 batches (2026-09-24 to 27) were negative for admission holds and positive for forecast-driven KV placement, which changes item 4's primary question to placement; the study design is `docs/research/2026-09-27-l2-h100-study-design.md`. Every remaining v1 component was then built test-first without new experiments (`docs/superpowers/plans/2026-09-27-v1-completion.md`): the `atfm.control` package (GDP planner of 6.2 on samples with the cap and per-tenant fairness; the tier logger of 6.3; the replica floor of 6.4 behind a virtual connector; a placement *touch* controller), the control loop runtime, a Redis Streams bus, the worker-metrics scraper and board HTTP service, proxy overflow-to-FCFS with alarms and directive expiry, a `/touch` endpoint, the rows/stage/training parsers and the OpenHands and Harbor adapters of 7, and in the simulator the keep-alive touch arms, trace replay, a golden test and run provenance.

**The deployable form of placement (amends 6.3).** No shipped engine exposes pin or evict for one session's KV, but every prefix cache is an LRU: a minimal request sharing a session's prefix (`max_tokens` 1, lowest priority) makes its blocks most-recently used. The touch controller issues such touches for sessions forecast to return within a short horizon whose blocks are near the eviction frontier, within a per-second budget; its costs (a batch slot for one step, the prefix hit, and a full recompute moved earlier when the prefix was already gone) are recorded per touch. This is what the H100 study tests, with a random-touch ablation to separate the mechanism from the forecast. **Amended 2026-09-27:** with LMCache in the stack the primary actuator is its controller's `pin`/`move` (D3); the touch remains the fallback, and the study runs both.

### Prediction admission implementation update (2026-09-28)

The proxy limits all unfinished prediction jobs, including running work whose
caller timed out, to `prediction_limit` (default four). At capacity it immediately
uses local estimates. One monotonic caller deadline spans dispatch and prediction;
board HTTP phases receive the remaining budget. Job completion, not caller
abandonment, returns capacity. Counters distinguish rejection, caller outcomes
and worker lifetime. Board state ownership and synchronous tick computation
remain unchanged. See the [implementation study](../../research/2026-09-28-prediction-overload.md)
for tests, paired latency/coverage results and limitations.


### Implementation update: board isolation (28 September 2026)

A single bounded control worker now owns registry/model/RNG/controller mutation.
Prediction reads use a scalar immutable projection atomically published after a
successful tick with its version, capture timestamp and pre-encoded forecast.
New events become visible at publication; a monotonic age limit causes fallback
when ticks fail or lag. No deep copy of model or session histories is required.
Overlapping control calls receive 503; cancellation retains worker ownership.
See the [implementation/evidence record](../../research/2026-09-28-board-isolation.md)
for complexity, legacy custom predictor limits, GIL contention, and paired results.

### Implementation update: exact peer counts (28 September 2026)

Outgoing priority hints now use a per-tier histogram of queued indices. Queue
mutations maintain counts; rank queries scan distinct indices without materializing
a peer list. Strict-less-than ties, held peers and empty tiers preserve the old
rank formula. Admission heaps and hold semantics remain unchanged. See the
[profile and trial record](../../research/2026-09-28-proxy-ranking.md) for O(U)
queries, worst-case O(Q), and why a tested larger upstream pool was reverted.

### Implementation update: bounded upstream pools (28 September 2026)

Direct upstream requests with initial admission windows above 20 now balance
across 16 independent HTTPX pools with a combined 100-connection limit. Smaller
windows retain stock HTTPX by default after a measured light-load latency cost. Response close, rather than header receipt,
releases balancing occupancy. Shared verifying TLS context, stock fallback for
proxy discovery and an explicit one-shard rollback preserve operational choices.
Admission priority/holds remain separate from transport. The
[transport study](../../research/2026-09-28-sharded-transport.md) records component
and end-to-end trials, including limits and the earlier rejected shared-pool tuning.

### Implementation update: offline settings selection (28 September 2026)

The experiment layer separates target workload/hardware contexts, candidate
settings, objective/constraints and search/validation budgets. Selection freezes
before disjoint-seed comparison with the baseline. Missing/failed trials cannot
win; inconclusive results remain explicit and no settings deploy automatically.
See the [protocol](../../development/policy-tuning.md) for approximate uncertainty,
finite-search limits and the future contextual-policy boundary. Runtime defaults
remain unchanged; the initial real-HTTP adapter uses a fake worker.
