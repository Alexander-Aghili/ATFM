# Paired GPU cache workloads

After the [single-request compatibility check](gpu-cache.md), run:

```bash
export CUDA_HOME="$PWD/tmp/venvs/gpu/lib/python3.12/site-packages/nvidia/cu13"
.venv/bin/python -m atfm_experiments.gpu_cache.matrix --output runs/gpu-matrix --repeats 3
```

This starts one owned vLLM/LMCache stack and runs 36 sequential cases. The six
workload shapes are 64-, 352-, 1,024- and 1,920-token prompts with up to eight
output tokens; a 1,024-token prompt with up to 128 output tokens; and a cache
pressure case with three other 1,920-token contexts between initial fill and
revisit. Four such contexts require about 840 MiB of KV data, above the 512 MiB
CPU cache budget. Actual occupancy, eviction-related metrics and lock counts
are saved; exceeding the working-set budget does not by itself prove a specific
target was evicted.

Each shape has three repeats and two modes: ordinary on-demand LMCache retrieval
(`disk`) and explicit ATFM CPU prefetch (`prefetch`). Pair order alternates by
repeat. All modes retain the same model, memory limits and engine settings.
The pressure baseline may include residual CPU hits; its mode name does not
claim every retrieved chunk came from disk. Other shapes explicitly clear idle
L1 after the initial fill. Unique hashed prompt prefixes prevent cross-case hits;
each case must independently observe zero external hits on its cold request.

The baseline and prefetch inputs have matched lengths and repeated bodies but
different isolation prefixes. Within each case the cold and revisit inputs are
identical. The runner requires identical generated text, at least prompt length
minus one external reused tokens, idle cache locks, successful prefetch completion
when requested, and CPU occupancy within the configured budget. Counter deltas
are measured per request rather than treating cumulative metrics as new hits.
Requested output limits are not claims about actual output lengths; raw responses
retain the server's token usage.

Report request latency and prefetch lead time separately, along with their sum.
The lead time includes actuator setup, validation and completion polling. The sum
does not include diagnostic metrics/status collection, artificial idle waits or
real tool execution. This experiment supplies the target in advance and is not a
prediction-policy evaluation. Non-streaming HTTP duration is not TTFT. Three
repeats do not justify tail-latency or throughput claims.

These are synthetic stress tests, not public-agent replay or task-quality evidence.
The public AgentX/Weka workload path remains in [local-cluster.md](local-cluster.md);
a faithful replay requires whole sessions fitting the chosen context and original
timing/output semantics. Do not truncate traces to make them fit this 2,048-token
engine. Persistent network storage and OS page caching also limit interpretations
of disk timings on a rental.

The output includes per-case requests, responses, cache status, metrics and results,
plus shared server logs and a source/environment manifest. A final summary is
written only after the owned servers stop successfully. Partial per-case results
survive a later failure; a missing summary does not constitute a passing matrix.
The runner never stops or terminates a rented Pod; manage billing separately.
