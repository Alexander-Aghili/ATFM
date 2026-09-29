# Public tool-call and agentic serving workloads

The public-workload follow-up uses Qwen/Qwen3-4B-Instruct-2507 at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, a 131,072-token serving window,
two sequences, eager execution, 55% GPU memory utilization, LMCache MP with
24 GiB CPU cache, and filesystem L2. The model's pinned configuration supports
262,144 positions natively; no context extension or input truncation is applied.
This differs from the 0.6B compatibility probe and must be reported separately.

## Complete Weka sessions

The source is the public
[SemiAnalysis Weka corpus](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126).
The local file SHA-256 is `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`.
None of its 393 complete roots fit a 40,960-token window; 82 fit 131,072.
Selection is a small, feasibility-driven sample, not representative sampling.

| Case | Root ID | Requests | Subagent groups | Largest recorded input + output |
| --- | --- | ---: | ---: | ---: |
| Short branch | bbdcb12440a7ab3496b9fac8b5f9824b1672 | 21 | 1 | 54,926 |
| Sequential | 5c5e408b76e5e22747853915f67be3c491a4 | 24 | 0 | 93,675 |
| Multiple branches | 07dd40536557a1d6440a923557c3129dc929 | 119 | 3 | 118,677 |

The first two have short recorded spans within their topology class; the last
offers three groups and 37,635 recorded output tokens. The initially considered
six-group root had 231,817 output tokens and was replaced before execution to
fit the rental budget without shortening any requests. All roots are copied unchanged, including
nested requests, timing, output sizes and block hashes. The selector saves source
and individual hashes; replay rejects modified selected files.

AIPerf 0.13.0 owns reconstruction, scheduling and branch execution. Replay uses
the fixed trace schedule without input/output synthesis caps or delay removal.
`ignore_eos:true` requests the recorded output length rather than stopping early
on EOS. The client tokenizer revision matches the server. Each case is bounded
to 1,200 seconds, with a 2,700-second aggregate replay deadline. A timed-out,
errored, truncated, incomplete or wrong-count replay is not a pass. Requested
token lengths still need comparison with actual exported server/tokenizer counts.

Weka reconstructs content from KV block hashes, not original tool arguments or
tool results. These selected roots have no `input_types` annotations, so they
cannot support tool-name-specific findings. Subagent groups and recorded waits
exercise agentic serving structure; they are not evidence of real tool execution.
The baseline uses ordinary LMCache; ATFM actuation is disabled. There is no
policy-benefit or official AgentX benchmark claim.

```bash
.venv/bin/python -m atfm_experiments.gpu_cache.trace_selection --output runs/public-selection
uv venv --python 3.12 tmp/venvs/aiperf
uv pip sync --python tmp/venvs/aiperf/bin/python experiments/local-cluster/client-lock.txt
export CUDA_HOME="$PWD/tmp/venvs/gpu/lib/python3.12/site-packages/nvidia/cu13"
.venv/bin/python -m atfm_experiments.gpu_cache.replay \
  --source runs/public-selection --output runs/public-replay
```

The selector also downloads the pinned BFCL samples described below and hashes
both their complete source files and selected rows.
Both serving processes are owned and cleaned up. The Pod itself must be stopped
separately. Keep raw AIPerf exports, server metrics, source manifests and failed
attempts alongside any summary.

## BFCL tool-call API samples

Three cases each from `simple_python`, `multiple` and `parallel` in
[BFCL V4](https://github.com/ShishirPatil/gorilla/tree/58f57e9124ea981403792dd51e00a6577e621fae/berkeley-function-call-leaderboard/bfcl_eval/data)
exercise real OpenAI-compatible tool schemas and generated `tool_calls`.
The selector takes the first three JSONL rows of each category from that pinned
commit, retaining source hashes and IDs in `bfcl/manifest.json`; selected arrays
are saved as `bfcl/{category}.json`. Question text and function descriptions remain unchanged.
The adapter converts BFCL's `dict`/`float`/`list` schema types to JSON Schema and
normalizes dots in function names to underscores, rejecting name collisions.

The server enables the Hermes tool parser. Requests use temperature zero, seed
seven and a 512-token output limit. Checks require nonempty tool calls, offered
function names, JSON-object arguments and no output truncation. Raw responses and
usage are saved. No generated call is executed, and these checks do not validate
argument semantics or replace BFCL's official evaluator. They are API integration
samples, not an official BFCL score or evidence that the model completes tasks.

The [synthetic matrix](gpu-workloads.md) remains a separate diagnostic option.
