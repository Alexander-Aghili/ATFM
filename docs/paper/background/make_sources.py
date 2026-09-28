"""Generate the paper's bibliography and readable primary-source registry."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = [
('transformer','Vaswani et al.','Attention Is All You Need','2017','https://arxiv.org/abs/1706.03762','Foundational attention formulation; not a serving benchmark.'),
('mqa','Shazeer','Fast Transformer Decoding: One Write-Head is All You Need','2019','https://arxiv.org/abs/1911.02150','Multi-query attention and decode bandwidth.'),
('gqa','Ainslie et al.','GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints','2023','https://arxiv.org/abs/2305.13245','Grouped-query attention architecture.'),
('flash','Dao et al.','FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness','2022','https://arxiv.org/abs/2205.14135','Exact tiled attention; arithmetic and IO are different costs.'),
('flash3','Shah et al.','FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision','2024','https://arxiv.org/abs/2407.08608','Hopper-oriented overlap and low-precision kernels.'),
('flashinfer','Ye et al.','FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving','2025','https://arxiv.org/abs/2501.01005','Inference attention kernels and variable layouts.'),
('orca','Yu et al.','Orca: A Distributed Serving System for Transformer-Based Generative Models','2022','https://www.usenix.org/conference/osdi22/presentation/yu','Iteration-level scheduling; OSDI proceedings.'),
('paged','Kwon et al.','Efficient Memory Management for Large Language Model Serving with PagedAttention','2023','https://arxiv.org/abs/2309.06180','Foundational vLLM paper; historical implementation, not latest configuration.'),
('v1','vLLM contributors','vLLM V1 guide','living documentation','https://github.com/vllm-project/vllm/blob/main/docs/usage/v1_guide.md','Unified token-budget scheduler; mutable main branch.'),
('prefix','vLLM contributors','Automatic Prefix Caching','v0.15.0 documentation','https://docs.vllm.ai/en/v0.15.0/design/prefix_caching/','Versioned hash-based cache design; not latest default flags.'),
('hybrid','vLLM contributors','Hybrid KV Cache Manager','living documentation','https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/','Full/sliding-window group distinctions; mutable latest docs.'),
('vdisagg','vLLM contributors','Disaggregated Prefilling (experimental)','v0.20.0 documentation','https://docs.vllm.ai/en/v0.20.0/features/disagg_prefill/','Connectors and phase separation; experimental in this version.'),
('sglang','Zheng et al.','SGLang: Efficient Execution of Structured Language Model Programs','2023','https://arxiv.org/abs/2312.07104','RadixAttention and structured programs.'),
('hicache','SGLang contributors','HiCache System Design and Optimization','living documentation','https://docs.sglang.io/docs/advanced_features/hicache_design','GPU/host/storage hierarchy, matching, prefetch, write-back.'),
('trt','NVIDIA','TensorRT-LLM KV Cache System','living documentation','https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html','Paged pools, prioritized retention, host offload; backend-specific semantics.'),
('vattention','Prabhu et al.','vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention','2024','https://arxiv.org/abs/2405.04437','CUDA virtual-memory alternative; distinct from 2025 sparse-attention namesake.'),
('sarathi','Agrawal et al.','Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve','2024','https://arxiv.org/abs/2403.02310','Chunked prefills and scheduling interference.'),
('distserve','Zhong et al.','DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving','2024','https://arxiv.org/abs/2401.09670','Disaggregation with SLO and bandwidth constraints.'),
('mooncake','Qin et al.','Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving','2025 revision','https://arxiv.org/abs/2407.00079v4','KV-centric cluster architecture and overload management.'),
('preble','Srivatsa et al.','Preble: Efficient Distributed Prompt Scheduling for LLM Serving','2024','https://arxiv.org/abs/2407.00023v2','Joint cache reuse and load balancing.'),
('lmcache','Cheng et al.','LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference','2025','https://arxiv.org/abs/2510.09665','Cache extraction, cross-engine reuse and transfer; project technical report.'),
('lmp','LMCache contributors','Architecture and Developer Guide: multiprocess mode','living documentation','https://docs.lmcache.ai/mp/architecture.html','Separate cache server, L1 manager, L2 adapters, asynchronous controllers.'),
('llegacy','LMCache contributors','LMCache Controller','legacy documentation','https://docs.lmcache.ai/kv_cache_management/index.html','Page now marks in-process mode deprecated; important ATFM compatibility boundary.'),
('lpin','LMCache contributors','Pin the KV cache','legacy documentation','https://docs.lmcache.ai/kv_cache_management/pin.html','Location-scoped pin, asynchronous event ID; CPU example is not engine-HBM pinning.'),
('cachegen','Liu et al.','CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving','2023','https://arxiv.org/abs/2310.07240','Compression and network-adaptive context loading.'),
('cacheblend','Yao et al.','CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion','2025 revision','https://arxiv.org/abs/2405.16444v3','Non-prefix reuse with selective recomputation; quality requires evaluation.'),
('kivi','Liu et al.','KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache','2024','https://arxiv.org/abs/2402.02750','Asymmetric key/value quantization; numerical quality tradeoff.'),
('mla','DeepSeek-AI','DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model','2024','https://arxiv.org/abs/2405.04434','Multi-head latent attention; architecture-specific cache representation.'),
('spec','Leviathan et al.','Fast Inference from Transformers via Speculative Decoding','2022','https://arxiv.org/abs/2211.17192','Draft/verify acceleration with exact-distribution algorithm.'),
('dynamo','NVIDIA','Dynamo Architecture','living documentation','https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/architecture','Frontend, request/event planes, transfer and planner ownership.'),
('drouter','NVIDIA','Dynamo KV-Aware Routing','living documentation','https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing','Projected cache reuse and active load; routing is separate from transfer.'),
('dplanner','NVIDIA','Dynamo Planner','living documentation','https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/planner/overview','Existing predictive and load-based autoscaling; ATFM is not first to forecast.'),
('llmd','llm-d contributors','llm-d Architecture','v0.9 documentation snapshot','https://llm-d.ai/docs/architecture','Gateway/EPP, InferencePool, model servers and cache-aware deployment patterns.'),
('continuum','Li et al.','Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live','2026 revision','https://arxiv.org/abs/2511.02230v7','Latest inspected revision 8 September 2026; TTL and program-level scheduling.'),
('thunder','Kang et al.','ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System','2026 revision','https://arxiv.org/abs/2602.13692v3','Latest inspected revision 30 June 2026; program and tool-resource scheduling.'),
('tutti','Qiu et al.','Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving','2026 preprint','https://arxiv.org/abs/2605.03375','GPU-centric SSD IO research; reported results not reproduced here.'),
('asym','Shi et al.','Multi-Segment Attention: Enabling Efficient KV-Cache Management for Faster Large Language Model Serving','2026 preprint','https://arxiv.org/abs/2606.02964','AsymCache; position-dependent recomputation and kernel-aware eviction.'),
('agentbench','Chang et al.','From LLM Inference to Agentic Workloads: Characterization and Implications for Serving Systems','2026 preprint','https://arxiv.org/abs/2608.15127','AgentSysBench; heterogeneous non-LLM bottlenecks and state.'),
('mlperf','MLCommons','Agentic Inference for MLPerf Inference','8 July 2026','https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/','Benchmark announcement, not an ATFM result or a fixed released benchmark claim.'),
]

def escape(s):
    return s.replace('&', r'\&').replace('_', r'\_').replace('%', r'\%')

if __name__ == '__main__':
    records = [dict(zip(['key','author','title','edition','url','scope'], row), accessed='2026-09-28') for row in SOURCES]
    (HERE/'sources.json').write_text(json.dumps(records,indent=2)+'\n')
    bib = ['\\begin{thebibliography}{99}']
    md = ['# Primary-source register', '', 'Reviewed 28 September 2026. Living documentation is a dated snapshot, not a pinned release contract.', '', '| Key | Source | Scope |', '| --- | --- | --- |']
    for r in records:
        bib.append(r'\bibitem{'+r['key']+'} '+escape(r['author'])+'. '+r'\emph{'+escape(r['title'])+'}. '+escape(r['edition'])+'. '+r'\url{'+r['url']+'}. Accessed 28 September 2026.')
        md.append(f"| {r['key']} | [{r['title']}]({r['url']}) ({r['edition']}) | {r['scope']} |")
    bib.append(r'\end{thebibliography}')
    (HERE/'references.tex').write_text('\n\n'.join(bib)+'\n')
    (HERE/'SOURCES.md').write_text('\n'.join(md)+'\n')
