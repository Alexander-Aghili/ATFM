# Agent Traffic Flow Management: Core Idea and Experiment Plan

> **Research and product plan.** This document includes proposed capabilities.
> For the implemented system, start with the [README](README.md),
> [implementation status](docs/status.md), and [core developer guide](docs/development/core.md).

*Draft v1, September 2026*

---

## 1. Summary

Enterprises running agent fleets on self-hosted GPUs face a congestion problem that current inference stacks handle reactively. Agents alternate between short LLM turns and tool calls that range from milliseconds to hours. When many agents resume at once, they arrive at the GPU pool like aircraft arriving at a congested airport: they queue while holding memory, caches thrash, and user-facing latency suffers.

The core idea borrows from how the FAA manages air traffic. The FAA can see where every aircraft is and when it will land, projects arrival demand against airport capacity, and absorbs delay on the ground (cheap) rather than in the air (expensive). We apply the same approach to agent fleets:

1. **A demand board.** A streaming, probabilistic forecast of fleet demand for GPU memory and compute, built from the live state of every in-flight agent session (which tool is running, how far along it is, what data it is producing), not only from historical request rates.
2. **Class-aware scheduling.** Background agents whose next LLM turn will be short and will quickly launch another long tool call are served early, like I/O-bound processes in an operating system. User-facing agents get strong latency protection. Long background decodes can be preempted and offloaded.
3. **Forecast-driven control.** When the forecast predicts a surge, deferrable background work is delayed before it consumes resources (ground delay), capacity is scaled and KV state is pre-staged ahead of demand, and delay is allocated fairly across tenants.

The first target customer is an enterprise that self-hosts open models and runs both interactive and background agents on the same GPU pool. The prototype builds on NVIDIA Dynamo.

---

## 2. Problem

- **Agent workloads break request-level serving.** An agent session spans tens to hundreds of LLM calls separated by tool calls. Serving systems that treat each call independently evict KV cache during tool pauses and force recomputation or reload on resume.
- **Tool durations are heavy-tailed and dynamic.** In published traces, fewer than 3% of tool calls account for roughly two-thirds of total tool time. Durations shift with CPU contention, network load, and shared-backend slowdowns, so historical predictors degrade when the environment changes.
- **Congestion is handled on arrival.** Current systems let every agent resume immediately and deal with contention afterward. Continuum measured scheduling bubbles (waiting for GPU memory after a tool returns) at up to 58.2% of an agent program's total delay.
- **Capacity changes have lead time.** Scaling out takes tens of seconds to minutes (model loading, engine startup), and staging KV from SSD or remote storage takes seconds. Reactive control is always late for these decisions.
- **Enterprises mix latency classes on one pool.** A developer assistant (interactive) and nightly migration or CI-fix agents (background) compete for the same GPUs, with no principled way to trade their costs.

---

## 3. Core Idea

### 3.1 The aviation analogy

| Aviation | Agent fleet |
|---|---|
| Aircraft en route | Agent sessions paused on tool calls |
| ADS-B position updates refining ETA | Progress and data events refining tool completion time |
| Airport acceptance rate | HBM capacity, prefill throughput, token quotas |
| Airborne holding (expensive) | Agent resumes, waits in queue holding memory, cache thrashes |
| Ground holding (cheap) | Delay a deferrable turn, tool launch, or subagent spawn before it consumes resources |
| Ground delay program | Planned delays for deferrable workload classes during a forecast surge |
| Ration-by-schedule, collaborative decision making | Fair delay allocation across tenants; agents and tools report intent and progress |

Two insights carry over directly:

- **Delay is cheaper before departure than on arrival.** Holding a deferrable agent before it resumes costs almost nothing; letting it resume into a congested pool costs memory, cache, and latency for everyone.
- **Aggregate forecasts are sharper than individual ones.** A single tool call's completion time is noisy, but the sum over thousands of in-flight sessions is much more predictable, as long as correlated slowdowns are modeled.

### 3.2 The demand board

For each active session *i* at time *t*, maintain a predictive distribution over:

- **R_i**: time until the session's next LLM request (tool completion time plus harness overhead)
- **P_i, D_i**: prefill and decode tokens of that request
- **K_i**: KV blocks required on resumption (context length × per-token KV size)
- **N_i**: number of child sessions spawned (subagent fan-out)

For a session whose tool has been running for elapsed time *a*, with survival function *S_i*, the probability of resumption within horizon *h* is:

```
F_i(h | a) = 1 - S_i(a + h) / S_i(a)
```

Live progress and data events update this through a Bayesian filter. With reported completed work *w*, total work *W*, and an online rate estimate *r̂*, a naive remaining-time estimate is `(W - w) / r̂`; the filter maintains a posterior over the rate (and phase, for multi-phase tools) rather than a point estimate.

Fleet demand at time *t + h* is then:

```
D(t + h) = Σ_i 1{R_i ≤ h} · K_i  +  (children spawned)  +  D_exo(t + h)
```

where `D_exo` is demand from new sessions, modeled as a nonhomogeneous Poisson process. The distribution of `D(t + h)` is computed by Monte Carlo over session-level samples, with a shared latent factor per backend (CI cluster, external API, data warehouse) so that correlated slowdowns widen the aggregate distribution correctly.

Because agents can spawn more work when they land, the full model is a branching process or queueing network rather than a pure arrival process.

### 3.3 Class-aware scheduling

Each waiting LLM call gets a priority index:

```
π_i = w(class_i) · (1 + β · E[T_tool_next,i]) / E[S_i]
```

- `w(class)`: large for interactive, small for background
- `E[S_i]`: predicted LLM service time for this call (prefill + decode)
- `E[T_tool_next,i]`: predicted duration of the tool call this turn will launch
- `β`: weight on the unblocking bonus, the value of releasing long tool work sooner

This is a weighted shortest-expected-processing-time rule (the cμ rule, or a Gittins index under uncertain service time) with an agent-specific unblocking term. Interactive calls also carry a slack term: when remaining slack against the latency SLO falls below a threshold, the call jumps the queue. Long background decodes can be preempted and their KV offloaded when interactive demand arrives or a surge is forecast.

### 3.4 Three control horizons

| Horizon | Decisions | Method |
|---|---|---|
| Strategic (days to weeks) | GPU count, memory tiers, reserved vs. on-demand capacity | Two-stage stochastic program with CVaR constraint on tail latency |
| Tactical (minutes) | Scale-out, KV pre-staging, ground delays for deferrable classes | Forecast upper bounds; chance-constrained ground holding with ration-by-schedule fairness |
| Real-time (seconds) | Queue ordering, preemption, KV retention and prefetch | Class-aware index (3.3); integrate existing TTL and progress-aware policies |

### 3.5 Ground delay formulation (tactical)

Discretize the horizon into slots τ. Let `C_τ` be pool capacity, `I_τ` the forecast interactive demand (random), and `x_{i,τ}` the decision to release background session *i* in slot τ.

```
minimize    Σ_i Σ_τ c_i · (τ - τ_i^0)^+ · x_{i,τ}  +  λ · E[overflow]
subject to  P( I_τ + Σ_i K_i · x_{i,τ} ≤ C_τ ) ≥ 1 - ε    for all τ
            Σ_τ x_{i,τ} = 1                              for all i
```

where `τ_i^0` is the session's natural (forecast) resumption slot and `c_i` its delay cost (higher for tight deadlines). This adapts the stochastic ground holding problem from air traffic management (Ball, Hoffman, Odoni, and Rifkin, 2003; chance-constrained ground delay programs). In practice, released sessions are ordered by ration-by-schedule (original expected resumption order) for fairness across tenants.

---

## 4. Related Work and Positioning

| Work | What it does | Gap relative to this plan |
|---|---|---|
| InferCept (2024) | Preserve / swap / discard KV during tool interruptions | Static; assumes predictable tool calls; per request |
| Continuum (2025–26) | KV TTL from per-tool duration history | Per session; history-based; no fleet forecast |
| Ask the Tool, Don't Guess (Sept 2026) | Reads live tool progress for per-call KV decisions | Per call; no aggregation or capacity control |
| PBKV, Tokencake, KVFlow | Predictive eviction, offload, prefetch for agent workflows | Per request; no fleet-level demand or class trade-offs |
| CONCUR (2026) | Agent-level admission control, congestion-based | Reactive (TCP-style); offline batch only |
| llm-d flow control | Priority admission and tenant fairness | No forecasting |
| Dynamo Planner | Autoscaling from forecast request counts and sequence lengths (ARIMA, Kalman, Prophet) | Forecasts from history, not in-flight state |
| Predictive K8s autoscaling for LLMs (Sept 2026) | Delay-aware lookahead with UCB margin | Demand-history signal; finds predictor sophistication matters little |
| Air traffic flow management (OR literature) | Stochastic ground holding, chance-constrained and robust ground delay programs | Never applied to compute or LLM serving |

**Positioning:** no existing system aggregates streaming per-session state into a probabilistic fleet demand forecast and uses it for ground-delay-style control combined with class-aware scheduling.

---

## 5. Key Risks

| Risk | Why it matters | How the experiment tests it |
|---|---|---|
| Horizon mismatch | Memory reshuffles in ms; forecasting only pays for decisions with lead time | Evaluate forecasts at 10 s to 15 min; measure value per decision type |
| Predictor sophistication may not matter | Prior work found simple lookahead + UCB captures most gains | Model ladder isolates the value of in-flight information versus model complexity |
| Exogenous demand | New sessions are not "in the air" | Measure endogenous fraction of demand by horizon and workload |
| Agents multiply on landing | Fan-out and variable turn length | Model children and turn size; evaluate forecast of total demand, not arrivals |
| Correlated delays | Shared backends slow together; variance doesn't average out | Controlled perturbations; latent backend factor in M3 |
| Limited deferrability | Interactive work cannot be delayed | Explicit class mixes; measure background deadline hit rate |
| Better predictions may yield small gains | HPC backfilling showed slack absorbs errors | Oracle upper bounds quantify headroom |
| Platform risk | Dynamo may absorb the idea | Moat is the in-flight tool-runtime signal the planner cannot see |

---

## 6. Hypotheses

- **H1 (forecastability):** In-flight session state forecasts fleet KV and prefill demand at 30 s to 15 min horizons significantly better than history-based predictors.
- **H2 (class-aware scheduling):** The unblocking-aware index improves interactive SLO attainment and background throughput versus static priority, at equal GPU cost.
- **H3 (forecast-driven control):** Ground delays, forecast-driven pre-staging, and forecast-driven scaling improve the SLO-versus-GPU-cost Pareto frontier beyond H2 alone.

---

## 7. Experimental Platform

### 7.1 Components

| Component | Role | Built on |
|---|---|---|
| Inference backend | Serves the model | Dynamo + vLLM or SGLang |
| Harness proxy | Sits between agent harnesses and Dynamo frontend; sets `nvext.agent_hints` (`priority`, `osl`, `speculative_prefill`) per request; enforces ground delays; logs requests | New (thin service) |
| Tool-runtime sidecar | Wraps tool execution; emits progress and data events without changing what the agent sees | New; follows the result-path / progress-path separation from "Ask the Tool" |
| Event bus | Carries session and tool events to the demand board | Redis Streams or Kafka |
| Demand board | Maintains per-session posteriors and fleet forecasts | New (Python) |
| Custom router strategy | Implements class-aware queue ordering | Dynamo `KvRouter` Python bindings |
| Planner plugin | Injects forecast upper bounds as replica floor | Dynamo Planner plugin pipeline (PREDICT stage, gRPC) |
| Simulator | Fast policy sweeps on replayed traces | DynoSim / Mocker |

### 7.2 Known Dynamo gaps (need small patches or workarounds)

- No TTL cache pinning via request extension (priority only). Continuum-style baseline needs an engine-level patch.
- Prefetch hooks timed to predicted tool returns are under development; use `speculative_prefill` plus KVBM tier moves where possible.
- Priority semantics differ by backend; Dynamo normalizes, but preemption behavior must be verified per engine.

### 7.3 Hardware

- Phase 0: one 8×H100 node (or equivalent), roughly 2–3 weeks
- Phase 1: CPU only
- Phase 2: simulation, then one node for confirmation
- Phase 3: simulation, then two nodes so that planner scaling matters

### 7.4 Model

A strong open coding model sized to leave meaningful KV headroom on the node (for example Qwen3-Coder-Next or a GLM-5.x variant). Record exact checkpoint, quantization, and engine version.

---

## 8. Phase 0: Trace Generation (Weeks 1–3)

### 8.1 Workload mix

| Class | Workload | Why it matters |
|---|---|---|
| Interactive | Coding-assistant sessions (OpenHands or mini-SWE-agent) with sampled human think times | Latency-critical baseline |
| Interactive | Terminal-Bench-style tasks | Many short tool hops |
| Background | Batch SWE-bench Verified solving with full test runs | Long, progress-emitting tools (pytest, builds) |
| Background | Repo-wide migration or test-generation agents | Long sessions, many turns, fan-out |
| Background | Data-pipeline agent running dbt or Spark jobs and reading logs | Streaming data events, correlated backend slowdowns |
| Background | Fine-tuning monitor agent watching loss curves | Intermediate-data signals, early-stop decisions |

### 8.2 Log schema (one record per event)

| Field | Description |
|---|---|
| `session_id`, `parent_session_id` | Session identity and fan-out lineage |
| `class`, `tenant` | Interactive / background; tenant for fairness analysis |
| `turn_index` | Position in session |
| `t_request`, `t_first_token`, `t_last_token` | LLM call timing |
| `isl`, `osl`, `prefix_hit_tokens` | Token counts and cache reuse |
| `kv_blocks_by_tier` | KV blocks held in HBM / DRAM / SSD / remote |
| `worker_id` | Serving worker |
| `tool_name`, `tool_args_hash` | Tool identity (args hashed for privacy) |
| `t_tool_start`, `t_tool_end`, `tool_exit_status` | Tool timing and outcome |
| `progress_events[]` | `{t, completed, total, phase}` from sidecar |
| `data_events[]` | `{t, metric, value}` (loss, residual, rows processed, test pass/fail counts) |
| `backend_id` | Shared backend used by the tool (CI runner, API, warehouse) |
| `spawned_children` | Subagent spawns |
| `perturbation_flag` | Whether a controlled perturbation was active |

### 8.3 Controlled perturbations

Inject known disturbances so forecast robustness can be measured against ground truth:

- Slow the CI runner by 2× for 20-minute windows
- Add latency to one external API
- Saturate the data warehouse with synthetic load
- Kill and restart one serving worker

### 8.4 Volume

Several thousand sessions across all workloads and several hundred GPU-hours. Hold out entire time blocks (not random events) for testing, to avoid leakage across correlated periods.

### 8.5 Checkpoint

Measure sidecar signal coverage (share of tool time with strong progress signals, weak end-phase signals, or none). "Ask the Tool" reported roughly 38–51% strong coverage and 59–72% including weak signals on coding-agent traces; use this as a reference.

---

## 9. Phase 1: Forecastability, Offline (Weeks 3–6)

### 9.1 Model ladder

| Model | Information used |
|---|---|
| B0 | Constant (current = next) |
| B1 | Dynamo Planner predictors (Kalman, ARIMA) on aggregate time series |
| B2 | Per-tool historical duration distributions, aggregated (Continuum-style) |
| M1 | B2 + elapsed-time conditioning via survival hazard |
| M2 | M1 + live progress and data events through a Bayesian filter |
| M3 | M2 + latent shared-backend factor for correlated slowdowns |

Each step adds one information source, so gains can be attributed.

### 9.2 Targets

- Fleet KV blocks required, D(t + h)
- Fleet prefill tokens per second
- Per-class breakdown (interactive vs. background)

### 9.3 Horizons

10 s, 30 s, 2 min, 5 min, 15 min

### 9.4 Metrics

- **CRPS** of the full predictive distribution
- **q90 and q95 pinball loss** (upper quantiles drive capacity decisions)
- **Interval coverage** of 80% and 90% bands
- **Surge lead time:** seconds between the forecast q90 crossing capacity and actual demand crossing capacity; plus false-alarm rate
- **Endogenous fraction:** share of demand at each horizon from already-active sessions
- **Shift robustness:** all metrics restricted to perturbation windows

### 9.5 Go/no-go gate

Proceed to Phase 3 if both hold (thresholds are starting points):

- M1 or M2 reduces q90 pinball loss by at least ~20% versus B1 at 2–15 min horizons
- Endogenous fraction exceeds ~50% at 5 min for background workloads

If the gate fails, the forecast's value is limited to strategic capacity planning; continue with Phase 2 and the strategic planner only.

---

## 10. Phase 2: Class-Aware Scheduling (Weeks 4–8)

Runs in parallel with Phase 1; needs only short-horizon service-time predictions.

### 10.1 Predictors needed

- LLM service time per call (from ISL, predicted OSL per tool type, current batch state)
- Next tool duration (per-tool distribution plus arguments features)

### 10.2 Policies

| ID | Policy |
|---|---|
| P0 | Dynamo defaults: queue order, LRU eviction |
| P1 | Static priority: interactive high, background low |
| P2 | P1 + Continuum-style TTL retention (engine patch) |
| P3 | Class-aware index (section 3.3) with preemption and offload of long background decodes |
| P3-oracle | P3 with true service and tool times (upper bound) |

### 10.3 Implementation

- Harness proxy computes π_i and sets `priority` and `osl` per request
- Custom router strategy orders the queue by π_i and applies the interactive slack override
- Preemption through engine priority preemption as passed through Dynamo

### 10.4 Sweeps

β (unblocking weight), class weights, slack threshold, preemption threshold. Sweep in DynoSim; confirm top configurations on GPUs.

---

## 11. Phase 3: Forecast-Driven Control (Weeks 7–11)

### 11.1 Policies

| ID | Policy |
|---|---|
| P4 | P3 + ground delay: hold background resumptions and tool launches when forecast q90 demand exceeds capacity; release by ration-by-schedule |
| P5 | P4 + pre-staging: `speculative_prefill` and KV prefetch triggered by the forecast instead of fixed timers |
| P6 | P5 + planner integration: forecast upper bound sets the replica floor |
| P6-oracle | Perfect future knowledge (headroom) |

### 11.2 Scenarios

| Scenario | Description |
|---|---|
| Steady | Baseline mixed load |
| Interactive surge | 3× interactive arrivals for 30 min (morning standup) |
| Correlated slowdown | CI runner 2× slower for 20 min, then recovers (release wave) |
| Capacity loss | One worker fails mid-run |
| Mix 80/20 | 80% background, 20% interactive |
| Mix 50/50 | Equal mix |

---

## 12. Metrics (Consolidated)

| Category | Metric |
|---|---|
| Interactive | TTFT after tool return (p50 / p95 / p99); SLO attainment %; per-turn latency |
| Background | Job completion time; tasks per hour; deadline hit rate (e.g., nightly work done by 7 a.m.) |
| Fairness | Maximum and distribution of imposed delay per tenant |
| System | GPU-hours; recomputed prefill tokens; KV hit rate; KV transfer bytes by tier |
| Economics | Cost per completed task |
| Quality | Agent task success rate (confirm delays cause no timeouts or behavior changes) |
| Headline | Pareto frontier of interactive SLO attainment vs. GPU cost, per policy |

---

## 13. Methodology and Rigor

- At least 5 seeds per configuration
- Paired comparisons on identical trace replays
- Bootstrapped 95% confidence intervals, resampling by session or time block
- Time-block train/test splits for all learned predictors
- Every simulated result spot-checked on real GPUs; report simulator-vs-real discrepancy
- Oracle upper bounds for every learned component
- Pre-register gate thresholds before running Phase 1 evaluation

---

## 14. Timeline

| Weeks | Work |
|---|---|
| 1–3 | Phase 0: platform, sidecar, proxy, trace generation; start customer discovery |
| 3–6 | Phase 1: demand board, model ladder, forecast evaluation, gate decision |
| 4–8 | Phase 2: scheduling policies in DynoSim, GPU confirmation |
| 7–11 | Phase 3: ground delay, pre-staging, planner integration; multi-node runs |
| 12 | Write-up, dataset release prep, design-partner prototype packaging |

---

## 15. Deliverables

1. **Trace dataset:** agent sessions with class labels, tool progress and data events, KV footprints, and controlled perturbations (publishable)
2. **Paper 1 (H1):** forecasting agent fleet demand from in-flight state
3. **Paper 2 (H2/H3):** class-aware scheduling and ground delay control for agent serving
4. **Prototype:** harness proxy, tool sidecar, router strategy, and planner plugin installable on an existing Dynamo deployment

---

## 16. Customer Discovery

### 16.1 First-customer profile

- Self-hosts open models on dedicated GPUs (at least one 8-GPU node for agents)
- Runs both interactive agents (developer assistant) and background agents (CI-fix, migration, test generation, data pipelines) on the same pool
- Has visible GPU budgets and felt surge pain
- Can deploy a sidecar in its tool runtime

### 16.2 Segments

| Segment | Why | Role |
|---|---|---|
| Regulated enterprises (banks, insurers, defense, healthcare) with internal AI platforms | Must self-host; mixed agent classes; tight GPU budgets | Likeliest first design partner |
| Companies running background coding agents at scale (e.g., Stripe, Ramp, Spotify publicly report high agent PR volumes) | Best source of workload shape, deadlines, deferrability | Interviews; some evaluating self-hosting |
| Coding-agent vendors with on-prem / VPC offerings | Run serving stacks for many regulated customers; long sessions | Partnership / channel |
| GPU clouds offering managed open-model inference (CoreWeave, Nebius, Lambda, Crusoe) | Multi-tenant version of the same problem | Channel and strategic partners |
| NVIDIA Dynamo team | Co-designing `agent_hints` v1 and asking for harness feedback | Ecosystem relationship; shape the API |

### 16.3 Interview questions

1. What share of your agent traffic is background, and what are its real deadlines?
2. What happens during a surge today? Do interactive users notice?
3. What is your GPU utilization, and how long does scale-out take?
4. Do your long-running tools emit progress, logs, or metrics that could be streamed?
5. Would you install a sidecar in your tool runtime?
6. How do you allocate GPU capacity across teams or tenants today?
7. What would a 20% reduction in GPU-hours at the same SLO be worth to you?
8. Could you share anonymized traces?

The deadline and surge answers indicate whether ground delays have value for that customer; the sidecar answer indicates whether the in-flight signal is obtainable.

### 16.4 Sequencing

Start discovery calls in week 1 alongside Phase 0, let early conversations shape the workload mix, and aim for one design partner willing to share anonymized traces by the end of Phase 1.

---

## 17. Open Questions

- How much of the forecast's value survives when only weak (end-phase) signals are available?
- What is the right delay-cost model `c_i` for background work: linear, deadline-based, or learned from customer SLAs?
- Does the class-aware index interact badly with KV-aware routing (priority versus cache locality)?
- Can the same demand board govern hosted-API token quotas and spend, extending the market beyond self-hosters?
- Which signals should be proposed for standardization in `agent_hints` or tool protocols?

---

## 18. References

**Agent serving and KV management**
- Abhyankar et al., InferCept (ICML 2024)
- Li et al., Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live, arXiv:2511.02230
- Liu, Zhang, Li, Zhang, Ask the Tool, Don't Guess: Agent Tool Calls Hold Their Progress, and the Serving System Should Read It, arXiv:2609.18849
- CONCUR: High-Throughput Agentic Batch Inference via Congestion-Based Concurrency Control, arXiv:2601.22705
- SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters, arXiv:2605.00528
- PBKV: Efficient Serving for Dynamic Agent Workflows, arXiv:2605.06472
- Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications, arXiv:2510.18586
- CacheWise, arXiv:2606.16824
- AgentServeSim: A Hardware-aware Simulator for Multi-Turn LLM Agent Serving, arXiv:2606.09613
- Luo et al., Autellix (2025)

**Autoscaling and serving infrastructure**
- Decomposing Predictive Kubernetes Autoscaling for LLM Serving Under Long Startup Delays, arXiv:2609.20874
- NVIDIA Dynamo: Full-Stack Optimizations for Agentic Inference (docs.nvidia.com/dynamo/dev/digest/agentic-inference)
- NVIDIA Dynamo Planner design documentation
- DynoSim: Simulating the Pareto Frontier (NVIDIA, May 2026)
- llm-d flow control (Red Hat Developer, August 2026)

**Air traffic flow management**
- Ball, Hoffman, Odoni, Rifkin, A Stochastic Integer Program with Dual Network Structure and Its Application to the Ground-Holding Problem, Operations Research 51(1), 2003
- Chen and Sun, Stochastic Ground-Delay-Program Planning in a Metroplex, J. Guidance, Control, and Dynamics, 2018
- Glover and Ball, Stochastic Optimization Models for Ground Delay Program Planning with Equity–Efficiency Tradeoffs, Transportation Research Part C, 2013
- Distributionally Robust Ground Delay Programs with Learning-Driven Airport Capacity Predictions, arXiv:2402.11415

**Methods**
- Adams and MacKay, Bayesian Online Changepoint Detection, 2007
- Gneiting and Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation (CRPS), JASA 2007
- Gittins, Glazebrook, Weber, Multi-armed Bandit Allocation Indices
- Shapiro, Dentcheva, Ruszczyński, Lectures on Stochastic Programming
- Rockafellar and Uryasev, Optimization of Conditional Value-at-Risk, 2000
- Tsafrir, Etsion, Feitelson, Backfilling Using System-Generated Predictions Rather Than User Runtime Estimates, IEEE TPDS 2007
