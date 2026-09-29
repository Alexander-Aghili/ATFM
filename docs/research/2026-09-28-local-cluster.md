# Local serving preparation: public AgentX replay on Dynamo Mocker

Implementation: `2d6a978` (process lifecycle), `321159e` (experiment runner).
Runs occurred on 28 September 2026 local time (29 September UTC).
[Run instructions and backend contract audit](../development/local-cluster.md).

## Observed results

| Configuration | Requests | Child branches completed | Request errors | Truncated branches | Workers serving requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two Mocker workers | 20 | 4 | 0 | 0 | 2 |
| Four Mocker workers | 20 | 4 | 0 | 0 | 4 |

Dynamo 1.5.0 frontend logs confirm two/four registered members and distinct
serving worker IDs respectively. AIPerf 0.13.0 reported complete, uncancelled
exports. Worker processes shut down after each run. The full test suite passed
567 tests with three skips and six existing warnings; a subsequently added
client-process timeout test passed in the nine-test local-cluster module run.
All tracked Python functions remain within 20 physical lines.

The source is the existing public AgentX/Weka corpus (393 roots). Streaming
selection retained the smallest root by total recorded input/output tokens,
`e3dbfa26bddbcb04c48a73dab3fa353d45f4`, including its nested requests. The original
corpus SHA-256 is `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`.
The selected root contains 20 inference requests. Both final runs used its saved
JSON file; the initial selection manifest preserves the full-corpus hash chain.

An exploratory 32-request cap repeated/cut the workload and reported two truncated
children. It was rejected as a recipe for complete replay. A duration-only trial
also repeated the workload. The final command uses the selected root's request
count and checks errors, cancellation and truncation in the exported results.
Only the corrected final two/four-worker runs are archived as passing evidence.

## Interpretation

This establishes that an existing public workload can traverse an existing
benchmark client and a multi-worker simulated serving runtime locally. It tests
streaming and branch handling without inventing another load generator.

It does not establish task success, real inference throughput, H100 capacity,
LMCache compatibility, or ATFM control benefit. Delays were removed, requested
outputs capped at 16 tokens and Mocker timing accelerated. AIPerf token-count
mismatch warnings remain in the logs; synthetic response tokenization is not
validated. These runs are not faithful AgentX benchmark submissions. Unit tests
ran concurrently; no timing comparison between worker counts is claimed.

The local host exposes an NVIDIA RTX 4060 Laptop GPU with 8,188 MiB memory.
This opens a possible small-model hardware compatibility step, subject to backend
version/memory requirements. No GPU inference or LMCache package was installed
for these CPU runs. The existing LMCache adapter still requires the contract
repairs described in the guide before policy testing.

## Evidence

[Archived artifacts](results/local-cluster-2026-09-28/manifest.json) contain source
fingerprints, selection provenance, exact commands, results, frontend/worker logs,
AIPerf exports and full-suite test output. SHA-256 values are over uncompressed
bytes. The two-worker source was uncommitted at execution but the runner hashes
identify its content; the four-worker run records commit `321159e`. Both used
the final runner implementation. Runtime defaults and policy settings are unchanged.

Subsequent work completed the separate [real GPU cache compatibility check](2026-09-28-gpu-cache.md). It does not change the CPU-only scope of the runs above.
