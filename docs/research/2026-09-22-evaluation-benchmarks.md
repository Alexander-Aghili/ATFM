# How the agent-serving literature evaluates (2026-09-22)

Purpose: calibrate our hardware and workload plan against what published systems papers actually used. Conclusion up front: **2xH100 is inside the norm.** Most of the field demonstrates on 1-4 GPUs with an 8B-32B model, plus one 70B/TP4 config; only SAGA (64xA100) and SMetric (32xH20) are multi-node.

## 1. Paper-by-paper setup

| Paper | GPUs | Models | Workload + arrivals | Scale | Metrics | Baselines | Sim? | Code |
|---|---|---|---|---|---|---|---|---|
| Continuum (2511.02230) | 1xB200 (8B), 4xB200 (70B), 1xA100, 1xH100 (OpenHands), 8xH100 (RL) | Llama-3.1-8B/70B, Gemma-3-12B, GLM-4.5-355B | Replay of 100 mini-swe-agent SWE-bench + 100 BFCL-v4 traces; Poisson arrivals; live 500 SWE-bench Verified | 100 programs / 500 tasks | avg JCT (8x), p90/p95, jobs/s | vLLM, Autellix, InferCept+LMCache, SGLang, Dynamo 0.7 | no | github.com/Hanchenli/vllm-continuum (traces not released) |
| Ask the Tool (2609.18849) | 4xH100, single node; HBM-only and HBM+DRAM | open-weight, unverified | Replay 100 SWE-bench Verified sessions (mini-swe-agent + OpenHands) with recorded tool durations | 100 | p90 TTFT after tool: -20.7% vs LRU; oracle -27% | LRU, oracle, Continuum, CacheWise, TokenCake, InferCept | no | none |
| InferCept (ICML'24) | 1-4xA100 | GPT-J-6B, Vicuna-13B, Llama-3-70B TP4 | Synthetic augmented-LLM mix; req/s sweep | n/s | 1.6-2x load, 2x req/s | vLLM, swap/preserve/discard | no | github.com/WukLab/InferCept |
| Autellix (2502.13965) | 8xA100 single node | Llama-3.1-8B (4 replicas), 70B (2 replicas), Falcon-180B | ShareGPT, BFCL, LATS programs; Poisson | n/s | program latency 4-15x vs vLLM; multi-engine 1.5x | vLLM, vLLM-opt, MLFQ | yes, unvalidated | 3rd-party reimpl only |
| AgentServeSim (2606.09613) | validated on 1-2xB200, RTX PRO 6000 | Llama-3.1-8B TP1, 70B TP2 | 50 SWE-bench (median 47 calls) + 100 BFCL v4; Poisson 0.02-0.1 jobs/s | 150 programs; 20 paired cells | sim vs real JCT error within 6% | TTL/LRU/FCFS/PLAS/routing variants | it IS the sim | no repo |
| SAGA (2605.00528) | 64xA100, 8 nodes, IB | Llama-3-70B TP4 | SWE-bench Verified 500, WebArena 812, BurstGPT multi-tenant; Poisson | 500+812; 10 trials | TCT 1.73x/1.55x; 99.2% SLO | vLLM(+APC), SGLang, Llumnix, TRT-LLM, KVFlow | no | none |
| PBKV (2605.06472) | 8xA6000 | Qwen3-14B/32B TP2 | HoVer/LangChain, SWE-bench/AutoGen, FinanceBench/CrewAI; fixed concurrency 24-72 | 500 test traces | latency 1.85x, hit rate 2.55x | LRU on SGLang+HiCache, KVFlow | no | none |
| TokenCake (2510.18586) | 1xA100, 1-2xH20 | Qwen2.5-14B/32B/72B | Synthetic multi-agent DAGs; Poisson 0.05-1 QPS; 20 apps | 20 apps | e2e -47% | vLLM, Mooncake, Parrot | no | github.com/pku-lemonade/TokenCake |
| KVFlow (NeurIPS'25) | 1xA10G, 1xH100 | Llama-3.1-8B, Qwen2.5-32B | Synthetic 10-agent workflows; up to 64 concurrent | 64 | 1.83-2.19x | SGLang(+HiCache) | no | github.com/PanZaifeng/KVFlow |
| CacheWise (2606.16824) | **2xH200 TP2** | Qwen2.5-Coder-32B | Real Claude Code sessions (~10M tokens); deterministic replay at 10-50 concurrent sessions | 10-50 sessions | evictions -22%..-2.6x; SCT up to 3.5x | vLLM, InferCept, oracle | no | traces: github.com/cachewise-project/cachewise-coding-traces |
| Predictive K8s autoscaling (2609.20874) | 1-4xA100 pods | Qwen2.5-7B | ServeGen/Bailian fit, QPS ramps (not agentic) | 5-min ramps | TTFT SLO violations 63.5% -> 3.7% | HPA, KEDA, EWMA, KF | SimPy, calibrated to Vidur (relative only) | pending |
| CacheScout (2608.14624) | 8xRTX PRO 6000; 4xH200 | Llama-3.1-8B, Qwen3-235B-FP8 | GSM8K/MT-Bench/GAIA/SWE-bench via 6-agent AutoGen; 0.2-50 sessions/s | n/s | hit +10-18pp, TTFT -18..45% | vLLM, Continuum | no | claimed, no URL |
| HexAGenT (2605.16637) | simulated 16-20 instances | Llama-70B, Qwen3-235B | ShareGPT/BFCL/LATS chains, fixed rate | 100-400 wf | Req95/99 -20..33% | SGLang-FCFS/LLF, Autellix | sim only, unvalidated | none |
| ThunderAgent (ICML'26, 2602.13692; basis of Dynamo's Program Scheduler) | 8xH100 single node; 2x8xH100 RL; 1xRTX 5090 | GLM-4.6-355B FP8 TP8, Qwen3-235B, Qwen3-8B | Live SWE-Agent/OpenHands on SWE-bench Lite, ToolOrchestra on HLE; concurrency sweep to 96 | 300 tasks | throughput 1.48-3.58x vs vLLM | vLLM, Continuum, SGLang gateway | no | github.com/ThunderAgent-org/ThunderAgent |
| NVIDIA DynoSim blog (May 2026) | sim on M4 laptop; verified on HGX B200 and H200 | MiniMax-M2.5 TP4, Qwen3-32B TP2 | Mooncake FAST25 toolagent trace (23,608 req) | 23.6k req | TPS/TTFT/ITL Pareto; prefix reuse | round-robin | yes, no error % published | dynamo repo |
| SemiAnalysis AgentX / NVIDIA AIPerf | GB200/GB300 NVL72 (leaderboard); AIPerf runs on A100/H100 | any served model | 393 real Claude Code sessions, 98,827 requests, fixed-schedule closed-loop replay with subagents | 393 sessions | throughput vs TTFT Pareto | n/a | no | HF semianalysisai/cc-traces-weka-062126 |
| MLPerf Inference Agentic (Jul 2026) | vendor systems | Kimi K2.6/K3, Qwen3.6-35B-A3B, DeepSeek-V4-Pro | 613 real trajectories (113 DeepSWE coding + 500 Workato workflows), 30,335 turns, closed-loop one-in-flight per conversation, X-Session-ID | 613 | tokens/s/user vs system throughput; accuracy gates | n/a | no | github.com/mlcommons/endpoints/tree/main/examples/10_Agentic_Inference |

## 2. Shared building blocks (common ground for comparison)

1. **SWE-bench Verified/Lite trajectories via mini-swe-agent or OpenHands, Poisson program arrivals with a rate sweep.** Used by Continuum, AgentServeSim, Ask the Tool, SAGA, PBKV, CacheScout, ThunderAgent, ConServe. The de-facto shared workload; each paper records its own trajectories.
2. **BFCL v3/v4** (Autellix, Continuum, AgentServeSim, HexAGenT).
3. **Real Claude Code corpora:** AgentX 393 sessions (Apache-2.0; replayable with `aiperf --public-dataset semianalysis_cc_traces_weka_with_subagents --fixed-schedule`), CacheWise CATraces, TraceLab 4.3k sessions.
4. **Mooncake FAST25 toolagent_trace.jsonl** (23,608 req; DynoSim/Mocker native format).
5. **MLPerf Agentic Inference harness**: the only vendor-neutral standardized suite; no academic paper uses it yet.
6. **Exgentic agent-llm-traces v2** (10k OTel-shaped runs; no tool spans).
7. Common baselines: vLLM (+prefix caching), SGLang (+HiCache), InferCept, Autellix/PLAS, Continuum, ThunderAgent/Dynamo. Common metrics: mean/p90-p99 JCT or session completion time, TTFT after tool return, jobs/s, KV hit rate and evictions, SLO attainment.
8. Arrival models by frequency: Poisson rate sweep (most), fixed concurrency (PBKV, CacheWise, ThunderAgent, KVFlow), timestamp replay (AgentX, SMetric, MLPerf).

## 3. Forecasting-specific prior work (H1 novelty check)

No 2026 paper has aggregate fleet demand forecasting from in-flight session state as its main contribution. Nearest: Predictive Multi-Tier Memory Management (2604.26968; single author, 1 GPU, analytical cluster projections), PBKV (per-workflow next-invocation prediction), GoodServe (2605.16867; output length prediction for routing), SageServe (cloud-trace forecast autoscaling, not agentic), the K8s autoscaling paper (non-agentic demand). **Counter-evidence to take seriously:** ConServe "Observation, Not Prediction" (2606.01839; 4xA40) argues reactive placement beats prediction, and an independent replication on the AgentX corpus (github.com/gauravapiscean/agentic-kv-cache) reports LRU is hard to beat because recomputation is dominated by sub-10 s tool loops, not TTL expiry. Our H1 gate and oracle bounds are the right answer to both.

Characterization datasets for demand modelling: TraceLab (released), AgentSysBench (2608.15127, pending), SMetric (2607.08565, pending), Agentic AI Workload Characterization (2605.26297; 2xH100 NVL, data on figshare).

## 4. What this means for our plan

- **Hardware:** 2xH100 covers H2 and the real-hardware validation of H3. Use Qwen3-Coder-Next-FP8 (80B/3B active, TP2) or a 14B-32B dense model with 2 replicas (DP2) so that routing exists. Multi-replica and multi-node effects come from DynoSim + Mocker (as NVIDIA does) with paired sim-vs-real validation on the 2 GPUs, reporting mean and p99 JCT error the AgentServeSim way (target within 6%).
- **Workloads:** (i) mini-swe-agent on SWE-bench Verified + BFCL v4, Poisson arrivals, for comparability with Continuum/AgentServeSim/Ask the Tool; (ii) AgentX Claude Code corpus via AIPerf fixed-schedule replay; (iii) TraceLab for the interactive-class demand model; (iv) our own sidecar-instrumented background runs for progress signals and perturbations; (v) optionally MLPerf Agentic for a standardized number.
- **Baselines:** vLLM/SGLang defaults, Continuum (open code), InferCept (open code), ThunderAgent Program Scheduler (open, inside Dynamo). Report oracle upper bounds.
- **Cost:** 2xH100 at ~$2-3.5/GPU-hr: 100 GPU-hrs ~ $300-700; 300 GPU-hrs ~ $1-2k. The earlier $5-7k figure came from detail.md section 7.3 (8xH100 node for weeks plus two nodes for Phase 3), which the literature does not require.
