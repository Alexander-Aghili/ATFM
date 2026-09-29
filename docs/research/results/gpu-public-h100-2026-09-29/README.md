# H100 public-workload evidence

Read the [study report](../../2026-09-29-public-gpu-workloads.md) for outcomes,
configuration, workload limitations and the difference between accepted cases
and rejected attempts.

## Readable results

- [Short-branch verdict](short-branch/result.json) and [AIPerf profile](short-branch/profile.json).
- [Sequential verdict](sequential/result.json) and [AIPerf profile](sequential/profile.json).
- [40-case BFCL summary](tools-summary.json): 39 call-shape checks passed;
  semantic accuracy was not scored.
- The flagged case's [request](parallel_9-request.json), [response](parallel_9-response.json)
  and [pinned reference](parallel_9-reference.json).
- [Selected public roots and source hashes](selection.json), [hardware](hardware.json),
  [shared-stack manifest](shared-manifest.json) and [tool-matrix manifest](tools-manifest.json).

## Complete archives

`public-initial-evidence.tar.gz` contains the initial configuration failures,
request-bounded truncation, reporting-regression attempt, two accepted seeded
cases and the invalid quota-exhausted large case. It includes the selected public
roots, BFCL inputs, raw AIPerf exports, metrics, logs and failure snapshots.
An archive that contains accepted cases is not an overall passing-suite verdict.

`public-tool-evidence.tar.gz` contains the isolated 40-case tool matrix: selected
source inputs, source hashes, every request/response, summaries and serving logs.
The separately retained `parallel_9-reference.json` records the official answer's
source URL and SHA-256; it was fetched after observing the call-shape discrepancy.

Archives preserve original files, including null-filled log tails from the disk
quota failure. Model weights and generated KV binary objects are excluded.
[SHA-256 checksums](sha256.json) cover the committed evidence.

Sources: [SemiAnalysis Weka dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)
and the [pinned BFCL data](https://github.com/ShishirPatil/gorilla/tree/58f57e9124ea981403792dd51e00a6577e621fae/berkeley-function-call-leaderboard/bfcl_eval/data).
