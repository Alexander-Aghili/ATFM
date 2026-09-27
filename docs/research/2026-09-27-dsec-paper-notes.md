# Notes: DeepSeek Elastic Compute (DSec), arXiv 2609.22978 (2026-09-27)

**What it is.** A systems report from DeepSeek (131 authors, submitted 19 Sept 2026) on the sandbox platform behind their agentic RL training. Four backends (function call, container, microVM, full VM) behind one SDK; composable environment layers (base image, workspace, toolkit) over overlayfs and EROFS; on-demand image loading from 3FS because sandboxes touch 4 to 13% of image bytes; virtio-pmem and DAMON-based memory reclamation for microVMs (40% lower peak, 21% lower time-integrated memory); QoS-aware CPU scheduling (latency-sensitive vs best-effort; SCHED_IDLE plus core scheduling cuts latency inflation from 45% to 17% at 50% co-located load); power-of-k placement; agent loop moved off preemptible GPU pods with pause/resume.

**Scale numbers.** One unit of ~160 CPU nodes (30k cores, ~250 TB DRAM): ~3 M sandboxes a day, ~380k concurrent, >5,000 creations a second, jobs up to 32k sandboxes, 11,266 base images and 102k workspaces in one week, 130+ TB of image artifacts.

**Workload numbers we cite.** Median sandbox lifetime 17.4 min (containers) and 15.5 min (microVMs), p99 over 3 h for both. ~90% of sandboxes average at most 5% of requested CPU. Tool-call phase described as short CPU bursts separated by waits on the model. Per-node peaks of 1,048 containers / 524 microVMs.

**What it does not contain.** Nothing on inference serving, KV cache, prefill, batching, admission control, or demand forecasting. No resume-latency or pause-cost numbers. Does not cite Dynamo, vLLM, SGLang, ThunderAgent, Continuum or ConServe.

**How it informs ATFM.**
1. Independent, production-scale evidence that the sessions whose KV we decide to hold or evict are long-lived and mostly idle: the regime where forecasting resumption beats reacting to it. Cited in spec section 1.1 and the architecture page.
2. The agent-loop / serving-pool decoupling is a stated production pattern (from DeepSeek-V4.1), which is ATFM's assumed topology; training rollout fleets are a second demand source.
3. Pause/resume is the operator's default lever without a forecast. It is what the `working_set` simulator baseline models; first simulator runs (2026-09-24 note) put it at 2.6x the background cost of demand-aware holds for the same interactive SLO. The H100 study must measure the real pause/resume and recompute cost.
4. Their CPU-side split into latency-sensitive and best-effort sandboxes is the same two-class split as our interactive/background tiers, applied one layer down.
