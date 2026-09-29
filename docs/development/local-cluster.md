# Local cluster preparation before GPU rental

This recipe runs **AIPerf 0.13.0 against Dynamo 1.5.0 Mocker workers** on CPU.
It reuses the public AgentX/Weka corpus already used for ATFM forecasting.
It is a serving integration smoke, not an ATFM policy evaluation, a faithful
workload benchmark, or an LMCache hardware test. No paid resources are created.

## Run

Run from the repository root on Linux with Python 3.12. Keep this recipe separate
from other local Dynamo stacks: the installed file-discovery backend is shared.
Use an unused HTTP port. Initial execution may download the Qwen tokenizer/config;
Mocker does not load model weights or run GPU inference.

```bash
uv sync --extra dev --extra serve --extra dynamo
uv pip install --python .venv/bin/python 'ai-dynamo==1.5.0'
uv venv tmp/venvs/aiperf
uv pip sync --python tmp/venvs/aiperf/bin/python experiments/local-cluster/client-lock.txt
.venv/bin/python -m atfm_experiments.local_cluster \
  --source data/agentx/traces.jsonl --workers 2 --out runs/local-cluster/two
```

The input is the public `semianalysisai/cc-traces-weka-062126` corpus, available
from [Hugging Face](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126).
Keep downloaded traces under ignored `data/`; do not collect private local agent
history. The existing file is approximately 1.8 GB. Selection streams one JSONL
root at a time, hashes the entire input, and chooses the smallest nonempty root
by total recorded input plus output tokens, with trace ID as the tie breaker.
It retains that root and all its nested requests unchanged in `trace.json`.

For a quick repeat with four logical workers, reuse the saved root:

```bash
.venv/bin/python -m atfm_experiments.local_cluster \
  --source runs/local-cluster/two/trace.json --workers 4 --out runs/local-cluster/four
```

The repeat manifest hashes the selected-root file as its source; keep the first
manifest to retain the full-corpus provenance chain. Outputs must be new directories.
The client environment is isolated from ATFM, with a complete package-version
snapshot in `client-lock.txt`. This is a Linux/Python 3.12 recipe, not a claim of
cross-platform reproducibility or a GPU dependency lock.

## What the smoke changes and checks

AIPerf owns Weka parsing, prompt reconstruction, streaming requests and branch
execution. The smoke removes recorded delays, caps requested output at 16 tokens,
and uses the selected root's original request count. The client has a 180-second
process timeout. These changes deliberately shorten the test; they disqualify it
as an AgentX benchmark submission or evidence about tool-return forecasting.
Weka hash-based prompt reconstruction also does not recreate original task semantics.

A pass requires AIPerf's completed, uncancelled export, positive request count,
no request errors and no errored/truncated branches. Inspect the branch statistics
and request count as well as the process exit status. Output tokenization warnings
are retained: Mocker output is simulated text, and client-tokenized counts can
differ from the requested count. This smoke does not assert token-count fidelity.

Artifacts include the source/selection hashes, package versions, ATFM source
fingerprints, runner hashes, exact CLI, worker/frontend logs and AIPerf exports.
`result.json` explicitly marks hardware, LMCache and ATFM control as untested.
The runner owns and cleans up the processes it starts, including failed startup.
A missing result or a failed result is not successful evidence.

## Hardware experiment contract

The first rental should run a small model on one H100 with sufficient host RAM:

1. Establish ordinary vLLM inference and streaming before enabling cache control.
2. Pin a compatible vLLM/LMCache release or container digest and record the model,
   tokenizer revision, drivers, GPU, host RAM, cache sizes and transfer settings.
3. Prove cold fill, supported transfer completion and subsequent cache reuse.
   Record observed residency, lock state and reused tokens; test leases only
   if the selected backend actually exposes them.
4. Replay a representative, dependency-aware workload subset with original timing
   semantics. Do not reuse the smoke's delay/output overrides for policy results.
5. Compare normal LMCache policy, ATFM in observation-only mode, then one enabled
   ATFM mechanism, at identical resource limits. Keep validation workloads separate.

The next local stage is now implemented: [real GPU validation and rental
handoff](gpu-cache.md) pins vLLM 0.30.0 and LMCache MP 0.5.5 and verifies CPU
warming followed by real inference reuse on an RTX 4060. It replaces the old
assumed legacy pin/move contract. MP warm prefetch holds no leases; pinning,
unpinning and direct GPU placement are explicitly unsupported. The H100 study
and representative policy comparisons remain future measurements.

## Upstream references

- [Dynamo local Mocker](https://docs.nvidia.com/dynamo/knowledge-base/concepts/simulation/local-mocker)
- [AIPerf Weka replay](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/weka-trace.md)
- [LMCache legacy pin API](https://docs.lmcache.ai/kv_cache_management/pin.html)

These links track upstream documentation; the executable recipe above checks its
installed versions. Simulated inference timings are not calibrated H100 timings.
