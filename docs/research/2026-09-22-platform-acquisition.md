# Platform acquisition research (2026-09-22)

> **Dated research record.** Findings and plans below retain their original scope.
> See [current implementation status](../status.md) and [current run instructions](../operations.md).

Consolidated from four web-research passes on 2026-09-22. All prices/versions are as of that date; re-check before booking.

## 1. What runs where

| Work item | Laptop (RTX 4060 8GB, CPU) | Rented 1xH100 | Rented 8xH100 node | Two nodes + IB |
|---|---|---|---|---|
| Dynamo frontend + KV router + planner + Mocker (file discovery, no etcd/NATS) | yes | | | |
| DynoSim / AISimulate trace replay (CPU-bound, uses pre-collected GPU profiles) | yes | | | |
| Custom router policy: Python `KvRouter` bindings, or Rust `request_classifier` plugin | yes | | | |
| Planner gRPC PREDICT/PROPOSE plugin with `virtual` connector | yes | | | |
| Demand board, forecast model ladder, evaluation | yes | | | |
| Sidecar + harness proxy dev against a small model (Qwen3.5-4B Q4 via llama.cpp/Ollama) | yes | | | |
| Functional test of `nvext.agent_hints` on real vLLM 0.28 / SGLang HiCache pinning | marginal | yes | | |
| Trace generation with a real coding model (Qwen3-Coder-Next-FP8), multi-worker | | | yes | |
| KV tiering (KVCR alpha, SGLang HiCache), disaggregated P/D | | | yes | |
| Planner scaling across nodes (Phase 3 P6) | | | | yes |

## 2. Compute options and cost

Market median H100 on-demand: ~$3.36/GPU-hr (Sept 20, 2026).

| Need | Recommended | $/hr | Notes |
|---|---|---|---|
| Dev box, 1xH100 | Voltage Park/Lightning $1.99; RunPod Secure PCIe $2.89 / SXM $3.49; Vast ~$1.5-1.9 | 2-3.5 | Root+Docker on Voltage Park, Lambda, Crusoe, Nebius VMs. RunPod pods are containers (no Docker-in-Docker, no K8s). |
| Cheaper dev | RunPod L40S $1.09, A100-80 $1.59; Modal A10 $1.10 ($30/mo free); Nebius preemptible H100 from $0.79 | 0.8-1.6 | For sweeps and functional tests. |
| 8xH100 node, single | Nebius on-demand $30.80/node-hr ($36 from Oct 1 2026); Crusoe `h100-80gb-sxm-ib.8x` $31.20; SF Compute market ~$16.6; Voltage Park non-IB $15.92 | 16-36 | Nebius: self-serve, per-second, 16xH100 quota without approval, free managed K8s with IB node groups. |
| Two nodes with IB + K8s | Nebius Managed K8s IB node group (2x gpu-h100-sxm); Together Instant Clusters ($3.99/GPU, managed K8s, IB) | ~64 | Lambda on-demand nodes have no inter-node IB. |

Cost estimates (median $3.36/GPU-hr): 300 GPU-hr ~$1.0k; 600 GPU-hr ~$2.0k; 2 nodes x 3 days ~$3.9k. Total plan ~$5-7k on-demand; ~$3-4k using SF Compute/Voltage Park for the single-node work. Tear the node down between runs (37-75 node-hours over 3 weeks is <25% utilization).

Credits worth applying for (free, no equity): NVIDIA Inception first (unlocks Nebius AI Lift up to $150k, AWS Activate Portfolio), Modal Startups (up to $50k), Together Accelerator ($15k), Lambda ($7.5k), Microsoft for Startups ($5k). Academic path if a university email exists: NAIRR Start-Up (2,000 GPU-hrs, 3-week decision), NSF ACCESS Explore.

## 3. NVIDIA Dynamo v1.5.0 (Sep 21, 2026)

- Install: `pip install ai-dynamo[vllm]` (Python 3.10 or 3.12), containers `nvcr.io/nvidia/dynamo/{vllm,sglang,trtllm}-runtime:1.5.0`, Helm `oci://ghcr.io/ai-dynamo/dynamo/platform:1.5.0`. Pinned: vLLM 0.28.0, SGLang 0.5.18, TRT-LLM 1.3.0rc25. CUDA 12.9+/13.0, driver 575+/580+.
- `--discovery-backend file` runs single-machine with no etcd/NATS.
- `nvext.agent_hints`: `priority` (int, router earlier-effective-arrival + forwarded to engine), `strict_priority` (router-only tier), `osl`, `speculative_prefill` (bool, max_tokens=1 prefill of predicted next prefix). v1.5.0 adds `AgentContext.input_trigger` in {user_message, tool_result, other}.
- Priority backends: vLLM `--scheduling-policy priority` (ordering only; priority KV eviction "planned"); SGLang `--enable-priority-scheduling` + `--radix-eviction-policy priority` (ordering + eviction); TRT-LLM router-only.
- TTL pinning: only SGLang `nvext.cache_control {type, ttl}` via HiCache `pin_prefix` (TTL 300-3600 s, experimental). No vLLM TTL merged (PR #38514 open, RFC #57103 open).
- Router extension: (a) Python `KvRouter` bindings: `best_worker`, `get_potential_loads`, `get_overlap_scores`, `generate`, `mark_prefill_complete`, `free`; worker selection only. (b) Rust native policies: `WorkerFilter/WorkerScorer/WorkerPicker` + `request_classifier` plugins (run before queue ordering; can prioritize and assign queue timing). No pure-Python native policy. Custom policies cannot override Dynamo's queueing/reservations/lifecycle.
- Planner: OBSERVE -> PREDICT -> PROPOSE -> RECONCILE -> CONSTRAIN -> EXECUTE. gRPC plugin proto at `components/src/dynamo/planner/plugins/proto/v1/plugin.proto`. External replica floor = PROPOSE/CONSTRAIN plugin returning `ComponentTarget` with `OverrideType.AT_LEAST`. Built-in predictors: constant, arima (default), kalman, prophet. `virtual` connector for simulation.
- Mocker: `python3 -m dynamo.mocker --model-path <tokenizer> --num-workers N --num-gpu-blocks-override ... --speedup-ratio ...`; CPU-only.
- DynoSim: driven by `pip install aisimulate` (0.12.0, experimental) + `ai-dynamo`; `aisimulate predict --stack dynamo --config <yaml>`. Inputs: Mooncake JSONL (`timestamp,input_length,output_length,hash_ids[,priority,session_id]`) or experimental `agentic_mooncake` (`request_id, wait_for[], delay, tool_wait_ms, branches, prefix_reset`). Legacy `python -m dynamo.replay` removed in v1.5.0.
- KVBM: deprecated in v1.5.0, removal in v1.6.0. Do not build on it. Replacement: engine-native offload; KVCR v0.1.0 (`pip install kvcr`, vLLM only, NIXL, "policy APIs" for eviction, router prefetch hints) for cross-node.
- ThunderAgent Program Scheduler (v1.3.0, experimental, not released): schedules turn->tool->turn programs, pauses at tool boundaries, resumes smallest-token programs first with transient priority boost. Closest existing thing to our scheduler; must be a baseline and is a positioning risk.
- Known issue v1.5.0: SGLang sidecar path returns HTTP 500 (protocol mismatch with SGLang 0.5.19).

## 4. Harnesses, benchmarks, traces, models

- OpenHands Software Agent SDK v1.49.4: `pip install openhands-sdk openhands-tools [openhands-workspace]`; `LLM(base_url=...)` via LiteLLM; `Conversation(...).run()` batchable; LocalWorkspace/DockerWorkspace/RemoteWorkspace; tool spans have `tool_call_id` + OpenTelemetry; subagents via `TaskToolSet` (parent blocks; `tool_concurrency_limit` for parallel task calls). CLI is maintenance-only.
- mini-SWE-agent v2.4.6: `mini-extra swebench --subset verified --model openai/<name> --workers N`; every action is one bash command; hook = subclass environment and override `execute()`. Easiest sidecar target.
- Terminal-Bench 4.0 (66 tasks; 2.0 = 89 tasks) via Harbor v0.23.0: `harbor run -d terminal-bench/terminal-bench@4.0.0 -a <agent> -m openai/<name> -k N`; `BaseAgent` (host loop, `environment.exec()`) is the sidecar point.
- SWE-bench Verified: `swebench` 5.0.2, `swebench eval verified -p preds.jsonl`; Epoch AI images 30 GiB total (`--namespace epoch-research`). Test output captured only after completion; sidecar must wrap `eval.sh` or tail inside the container for progress.
- Trace datasets: **TraceLab** (UW SYFI, arXiv:2606.30560, CC BY 4.0, github.com/uw-syfi/TraceLab): 4,300 sessions, 357k LLM rounds, 432k tool records, Claude Code + Codex, Sep 2025-Jun 2026; per round token counts (cached-prefix vs append), ordered timing events, per tool name, wall + internal latency, error flag. Calls >1 min are 4-5% of calls but 85-92% of tool time. Missing: progress events, KV per tier, background class, perturbations, backend ids. Others: Exgentic v2 (no tool spans), Mooncake toolagent_trace (no tool durations), swe-bench/experiments trajectories (some have durations). Continuum, Ask the Tool, AgentServeSim, SAGA released no traces. "CONCUR" could not be found by name.
- Models for 8xH100: Qwen3-Coder-Next-FP8 (80B/3B active, ~80 GB, TP2 reference, huge KV headroom) primary; gpt-oss-120b (~62 GB, one per GPU for DP experiments); Devstral 2 123B dense control; Qwen3.8-Flash-Next-FP8 (173 GB). Tight: GLM-5.3-Flash FP8 (306 GiB), Qwen3.5-397B-FP8 (~400 GB). Laptop: Qwen3.5-4B UD-Q4_K_XL (~5.5 GB) via llama.cpp/Ollama OpenAI-compatible server.
- Simulators: DynoSim (best fit, agentic format), Vidur (no sessions), LLMServingSim 2.0 (cycle-level, no agent semantics), AgentServeSim (no code), LLM-Emu (Cambridge, open, vLLM plugin replacing forward passes with profiled latency; single node), Revati (time-warp, no code).

## 5. The "Harvard virtual nodes" tool

CrystalLLM (arXiv name PrismLLM, arXiv:2605.15617, SOSP 2026; Minlan Yu's group at Harvard with Alibaba). Sandbox GPUs run real ranks, assistant GPUs host virtual ranks that replay recorded compute durations and do real NCCL communication; 0.58% iteration-time error. **Training only, not open-sourced as of today, no inference/KV/agent modeling.** Not usable here. The same hybrid idea for serving = Dynamo frontend with one real vLLM worker plus N Mocker workers, or LLM-Emu for GPU-free vLLM scheduler emulation.
