# Real GPU cache validation and rental handoff

ATFM's validated hardware path is **LMCache MP 0.5.5 disk → CPU warm
prefetch**, consumed by **vLLM 0.30.0** on the next inference request. The local
RTX 4060 Laptop test uses actual model weights, CUDA inference and filesystem
KV storage. It is a compatibility check, not evidence of policy benefit.

## Run on this machine or a single rented GPU

Use Linux x86-64, Python 3.12 and a CUDA-capable NVIDIA driver compatible with
the locked CUDA 13 runtime. The measured machine uses driver 595.84, an 8 GiB
RTX 4060 Laptop, and 30 GiB host RAM. Allow substantial disk space for the
PyTorch/CUDA environment, model download and build cache (50 GiB is a practical
starting allocation). LMCache's native extension may require a C++ build toolchain.
An H100 is the next target; the H100 result is not yet measured.
The [local evidence report](../research/2026-09-28-gpu-cache.md) records successful
fresh-cache and rebuilt-environment runs.

From the repository root, with `uv` installed:

```bash
bash experiments/gpu-cache/setup.sh
.venv/bin/python -m atfm_experiments.gpu_cache --output runs/gpu-check
```

The setup isolates serving dependencies in `tmp/venvs/gpu`; the complete
[version lock](../../experiments/gpu-cache/requirements.lock) includes vLLM,
LMCache and PyTorch 2.13.0. It is a Linux/Python 3.12 package-version snapshot,
not a container digest or a portable ABI guarantee. The runner pins both model
and tokenizer to Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`.
Initial startup downloads weights and compiles kernels. Readiness allows six
minutes per service. Both services bind loopback; ports 18180, 18181 and 15555
must be free. Output directories must be new. Usage reporting is disabled.

## What a passing check proves

1. A synthetic, deterministic request performs real inference. The tokenizer
   produces 352 prompt tokens, aligned to 22 cache chunks of 16 tokens.
2. Cold inference has zero external cache hits. The runner waits for idle
   storage and completed asynchronous writes.
3. Only the runner's freshly created, idle CPU cache is cleared. Its L2 disk
   directory is unique to the run. The runner checks that CPU occupancy is zero.
4. An actual ATFM `TierDirective(action="prefetch", tier="cpu")` submits a warm
   request. A 202 response is only acceptance. ATFM polls until completion and
   requires all expected keys, then the runner verifies CPU occupancy and that
   no read/write locks remain.
5. A second real inference request must return identical generated text and
   reuse at least 351 of the 352 prompt tokens according to vLLM's external
   cache counter. vLLM's own prefix cache is explicitly disabled. The serving
   path recomputes the final token rather than reporting all 352 as hits.
6. Owned server process groups are shut down, including on exceptions. A
   `summary.json` with `passed: true` is written only after the checks and shutdown.

Artifacts include exact server commands, installed versions, model revision,
GPU/driver, Git revision and source fingerprints, request and responses,
LMCache storage snapshots, prefetch completion, Prometheus snapshots and server
logs. Keep the entire directory. A missing summary means the run is incomplete
or failed. Raw timings are diagnostic observations, not a throughput comparison.
The synthetic prompt is intentionally small; it is not an AgentX workload or
an agent-quality test. Filesystem reuse may benefit from the OS page cache;
this does not measure cold-device disk latency.

## Contract and migration from the old adapter

The core adapter validates `/version` and `/status` before its first submission:
LMCache must report 0.5.5, the configured chunk size and healthy status. Recreate
an actuator after restarting or changing the backend. Model name, world size,
cache salt and token IDs must match the serving engine. The default world size
is one; the hardware check does not validate tensor parallelism.

`POST /cache/prefetches` carries `model_name`, `world_size`, `token_ids`,
`cache_salt`, `source_tier="l2"`, `target_tier="l1"`. The acknowledged chunk count
must match the complete token chunks. `GET /cache/prefetches/{request_id}` must
report the same ID and `found_keys == total_keys == chunks * world_size`.
Partial retrieval is explicitly unsuccessful. Pending requests retain their
original immutable token snapshot and ID; retries do not resubmit pending work.
The pending map is bounded (64 jobs by default). Polling waits are bounded by
a completion window plus an in-flight HTTP request timeout; it is not a hard
end-to-end deadline including tokenization and submission.

Completion results are consumed by the upstream status endpoint. A later 404
means **unknown**, not success. Transport errors remain fail-open and increment
the actuator's error count. `release_expired` is retained as a compatibility
hook that reconciles pending work; it releases no leases because this warm API
acquires none. Completion reconciliation does not retroactively increment the
control loop's synchronous `tier_applied` count.

**Pin, unpin, promotion, demotion and GPU prefetch are unsupported.** Touch
requests become an explicit unsupported result when this actuator is installed;
they are not silently turned into GPU placement or proxy inference. The loop
logs `cache_unsupported`, `cache_pending`, `cache_partial`, and other unsuccessful
outcomes separately from applied directives. Warming is subject to ordinary
LMCache eviction, with no residency guarantee until a future request arrives.
The old assumed legacy endpoints and local pin bookkeeping have been removed.

The control launcher uses the serving endpoint's `/tokenize`, rather than a
separately loaded tokenizer with potentially different chat-template options:

```bash
.venv/bin/python scripts/run_control.py \
  --lmcache http://127.0.0.1:18181 --lmcache-model Qwen/Qwen3-0.6B \
  --lmcache-chunk-size 16 --inference http://127.0.0.1:18180
```

This launcher supports ordinary message-only chat prompts with default template
settings. The proxy's remembered messages do not capture tools, multimodal
payloads or arbitrary template overrides; those cases need an exact token-ID
source before enabling actuation. The direct token-ID probe avoids this ambiguity.
The board's GPU-targeted proposals remain unsupported; this check validates a
CPU directive, not an automatically selected end-to-end placement policy.

## What to do after renting

First run exactly the same small check, unchanged, on one H100. Preserve its
artifacts and compare correctness, environment and shutdown results with the
local evidence. Do not begin with emulated nodes or simultaneous policy changes.

Then run the existing [AIPerf public-workload preparation](local-cluster.md) to
check the replay client. Its CPU smoke caps outputs and removes delays, so it
must not be presented as a policy benchmark. A real policy study needs a
whole-session subset that fits the selected model's context, original inter-call
and branch timing, unchanged requested outputs, and a train/validation split.
Do not truncate a 55k-token trace into this runner's 2k context window.

With those inputs fixed, compare ordinary LMCache, ATFM observation-only, and
one supported mechanism at identical memory/concurrency limits. Report reused
and transferred tokens/bytes, cache occupancy, TTFT and request latency tails,
throughput, error rate and controller overhead across paired repeats. Establish
an eviction/transfer bottleneck before tuning. Use the
[policy-selection protocol](policy-tuning.md) for hardware-specific selection;
these compatibility results do not select a universal policy.

## Upstream contract sources

- [Pinned MP HTTP schemas](https://github.com/LMCache/LMCache/blob/v0.5.5/lmcache/v1/multiprocess/http_apis/schemas.py)
- [LMCache MP HTTP API](https://docs.lmcache.ai/mp/http_api.html)
- [vLLM connector in the pinned LMCache release](https://github.com/LMCache/LMCache/blob/v0.5.5/lmcache/integration/vllm/lmcache_mp_connector.py)
