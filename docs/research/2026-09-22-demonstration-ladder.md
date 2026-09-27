# Demonstration ladder: what to test, at what level of evidence and cost (2026-09-22)

> **Dated research record.** Findings and plans below retain their original scope.
> See [current implementation status](../status.md) and [current run instructions](../operations.md).

Each level is cumulative. "Claims" are what results at that level can support; "cannot" is what they cannot.

| Level | Platform | Cost | What we test | Claims it supports | Cannot show |
|---|---|---|---|---|---|
| L0 Simulation | Laptop CPU: demand board + model ladder; DynoSim/AISimulate + Mocker replay | $0 | H1 on TraceLab + synthetic background sessions: CRPS, q90/q95 pinball, coverage, surge lead time, endogenous fraction; regime map (tail weight of tool durations, background share, fan-out) locating where prediction beats observation. Policy sweeps P0-P6 in DynoSim: beta, class weights, slack, preemption thresholds; Pareto frontiers; oracle bounds | Forecast skill vs history-based baselines (real for TraceLab, synthetic for background). Relative policy rankings and sensitivity | Absolute latency numbers; KV tier effects; anything the simulator was not validated for |
| L1 Functional | Laptop: real mini-swe-agent / OpenHands sessions, tool sidecar, harness proxy, Dynamo frontend + Mocker workers; LLM reasoning from a tiny local model (vLLM, Qwen3 0.6B-1.7B) or from a hosted API | $0 GPU; API tokens if hosted | End-to-end plumbing: sidecar progress/data events, proxy queue + ground delay + `nvext.agent_hints`, event bus, demand board consuming live state. Collect real tool-progress traces (pytest, builds, dbt) since tools run on CPU regardless of where the LLM is | Signal coverage (share of tool time with progress signals); correctness of the control loop; sidecar overhead | Any GPU-side effect |
| L2 One GPU | 1xH100 (~$2-3.5/hr, 20-80 hrs) | $100-300 | Real vLLM 0.28 / SGLang under Dynamo with a 27B-32B dense model or gpt-oss-120b. Verify priority ordering and preemption, SGLang HiCache TTL pinning (Continuum baseline), HBM<->DRAM offload. Calibrate DynoSim with paired sim/real cells (mean and p99 JCT error). Generate real-model background traces with sidecar and perturbations. Single-worker H2: class-aware queue vs vLLM FCFS, static priority, Continuum | H2 on one worker; simulator fidelity; scheduling-bubble and TTFT-after-tool measurements under contention | Routing, multi-replica, fleet-scale control |
| L3 Two GPUs | 2xH100 (~100-300 GPU-hrs) | $500-2k | DP2 (two replicas of a 27B-32B) or TP2 Qwen3-Coder-Next-FP8. Full H2 + H3 on real hardware: class-aware routing + queue, ground delay, forecast-driven pre-staging across HBM/DRAM(/SSD via HiCache L3 or LMCache), scenarios: steady, interactive surge, correlated slowdown release wave, capacity loss (kill one worker). AgentX replay via AIPerf. Baselines: vLLM, SGLang, Continuum, InferCept, ThunderAgent Program Scheduler | Headline Pareto (interactive SLO vs GPU cost) at the scale most papers use; comparability with the literature | Planner autoscaling with real replicas beyond 2 |
| L4 Hybrid emulation | 2xH100 real workers + N Mocker workers behind one Dynamo frontend; planner with `virtual` connector | same as L3 | Fleet-scale routing and forecast-driven replica floor (P6) with real tokens on 2 workers and simulated load on the rest; validated against L3 where they overlap | Planner integration and scale behaviour, with stated emulation caveats | Real multi-node KV transfer, real scale-out lead times |
| L5 Node (optional) | 8xH100 single node, 2-4 days | $1-2k | 70B+/TP4 config plus 2+ replicas; MLPerf Agentic Inference numbers; multi-replica real routing | Reviewer/customer-facing "big model" result | Not needed for core claims |

Order of execution: L0 and L1 in parallel now (both free); L2 as soon as L1 plumbing works; L3 for the headline; L4 with L3 hours; L5 only on demand.

## Test list by hypothesis

H1 forecastability (L0, refined at L2/L3 with our own traces)
1. Model ladder B0-B2, M1-M3 on TraceLab interactive sessions at 10 s, 30 s, 2, 5, 15 min horizons.
2. Same on synthetic background fleets with controlled tail weight (share of >1 min tools: 3%, 10%, 30%), fan-out, and correlated backend slowdowns.
3. Regime map: crossover horizon where M2 beats "observation" (B0 with instant reaction) as a function of tail weight and background share.
4. Surge lead time and false alarms; endogenous fraction by horizon.
5. Shift robustness inside perturbation windows.

H2 class-aware scheduling (L0 sweeps, L2 single worker, L3 two workers)
6. P0-P3 + P3-oracle on SWE-bench/BFCL Poisson replay; TTFT after tool return p50/p95/p99, background JCT, tasks/hr.
7. Preemption and offload of long background decodes when interactive demand arrives.
8. Interaction with KV-aware routing (priority vs cache locality) at L3.

H3 forecast-driven control (L0, L3, L4)
9. Ground delay P4 vs P3 in surge and release-wave scenarios; fairness (max imposed delay per tenant).
10. Pre-staging P5: forecast-driven promote/demote across HBM/DRAM/SSD vs fixed TTL; recomputed prefill tokens, KV transfer bytes by tier.
11. Planner floor P6 at L4; scale-out lead time absorbed by forecast.
12. Simulator fidelity: paired sim/real cells at L2/L3, report mean and p99 error.
