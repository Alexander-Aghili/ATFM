# Primary-source register

Reviewed 28 September 2026. Living documentation is a dated snapshot, not a pinned release contract.

| Key | Source | Scope |
| --- | --- | --- |
| transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) | Foundational attention formulation; not a serving benchmark. |
| mqa | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (2019) | Multi-query attention and decode bandwidth. |
| gqa | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (2023) | Grouped-query attention architecture. |
| flash | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022) | Exact tiled attention; arithmetic and IO are different costs. |
| flash3 | [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) (2024) | Hopper-oriented overlap and low-precision kernels. |
| flashinfer | [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005) (2025) | Inference attention kernels and variable layouts. |
| orca | [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) (2022) | Iteration-level scheduling; OSDI proceedings. |
| paged | [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023) | Foundational vLLM paper; historical implementation, not latest configuration. |
| v1 | [vLLM V1 guide](https://github.com/vllm-project/vllm/blob/main/docs/usage/v1_guide.md) (living documentation) | Unified token-budget scheduler; mutable main branch. |
| prefix | [Automatic Prefix Caching](https://docs.vllm.ai/en/v0.15.0/design/prefix_caching/) (v0.15.0 documentation) | Versioned hash-based cache design; not latest default flags. |
| hybrid | [Hybrid KV Cache Manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) (living documentation) | Full/sliding-window group distinctions; mutable latest docs. |
| vdisagg | [Disaggregated Prefilling (experimental)](https://docs.vllm.ai/en/v0.20.0/features/disagg_prefill/) (v0.20.0 documentation) | Connectors and phase separation; experimental in this version. |
| sglang | [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) (2023) | RadixAttention and structured programs. |
| hicache | [HiCache System Design and Optimization](https://docs.sglang.io/docs/advanced_features/hicache_design) (living documentation) | GPU/host/storage hierarchy, matching, prefetch, write-back. |
| trt | [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html) (living documentation) | Paged pools, prioritized retention, host offload; backend-specific semantics. |
| vattention | [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437) (2024) | CUDA virtual-memory alternative; distinct from 2025 sparse-attention namesake. |
| sarathi | [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) (2024) | Chunked prefills and scheduling interference. |
| distserve | [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670v3) (2024) | Disaggregation with SLO and bandwidth constraints. |
| mooncake | [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079v4) (2025 revision) | KV-centric cluster architecture and overload management. |
| preble | [Preble: Efficient Distributed Prompt Scheduling for LLM Serving](https://arxiv.org/abs/2407.00023v2) (2024) | Joint cache reuse and load balancing. |
| lmcache | [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/abs/2510.09665v2) (2025) | Cache extraction, cross-engine reuse and transfer; project technical report. |
| lmp | [Architecture and Developer Guide: multiprocess mode](https://docs.lmcache.ai/mp/architecture.html) (living documentation) | Separate cache server, L1 manager, L2 adapters, asynchronous controllers. |
| llegacy | [LMCache Controller](https://docs.lmcache.ai/kv_cache_management/index.html) (legacy documentation) | Page now marks in-process mode deprecated; important ATFM compatibility boundary. |
| lpin | [Pin the KV cache](https://docs.lmcache.ai/kv_cache_management/pin.html) (legacy documentation) | Location-scoped pin, asynchronous event ID; CPU example is not engine-HBM pinning. |
| cachegen | [CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving](https://arxiv.org/abs/2310.07240) (2023) | Compression and network-adaptive context loading. |
| cacheblend | [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444v3) (2025 revision) | Non-prefix reuse with selective recomputation; quality requires evaluation. |
| kivi | [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) (2024) | Asymmetric key/value quantization; numerical quality tradeoff. |
| mla | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) (2024) | Multi-head latent attention; architecture-specific cache representation. |
| spec | [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022) | Draft/verify acceleration with exact-distribution algorithm. |
| dynamo | [Dynamo Architecture](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/architecture) (living documentation) | Frontend, request/event planes, transfer and planner ownership. |
| drouter | [Dynamo KV-Aware Routing](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing) (living documentation) | Projected cache reuse and active load; routing is separate from transfer. |
| dplanner | [Dynamo Planner](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/planner/overview) (living documentation) | Existing predictive and load-based autoscaling; ATFM is not first to forecast. |
| llmd | [llm-d Architecture](https://llm-d.ai/docs/architecture) (v0.9 documentation snapshot) | Gateway/EPP, InferencePool, model servers and cache-aware deployment patterns. |
| continuum | [Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live](https://arxiv.org/abs/2511.02230v7) (2026 revision) | Latest inspected revision 8 September 2026; TTL and program-level scheduling. |
| thunder | [ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System](https://arxiv.org/abs/2602.13692v3) (2026 revision) | Latest inspected revision 30 June 2026; program and tool-resource scheduling. |
| tutti | [Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving](https://arxiv.org/abs/2605.03375) (2026 preprint) | GPU-centric SSD IO research; reported results not reproduced here. |
| asym | [Multi-Segment Attention: Enabling Efficient KV-Cache Management for Faster Large Language Model Serving](https://arxiv.org/abs/2606.02964) (2026 preprint) | AsymCache; position-dependent recomputation and kernel-aware eviction. |
| agentbench | [From LLM Inference to Agentic Workloads: Characterization and Implications for Serving Systems](https://arxiv.org/abs/2608.15127) (2026 preprint) | AgentSysBench; heterogeneous non-LLM bottlenecks and state. |
| mlperf | [Agentic Inference for MLPerf Inference](https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/) (8 July 2026) | Benchmark announcement, not an ATFM result or a fixed released benchmark claim. |
