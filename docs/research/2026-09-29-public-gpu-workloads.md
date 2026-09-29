# Public agentic and tool-call GPU workloads: 29 September 2026

A detailed [results PDF](gpu-results-report/atfm-gpu-results-2026-09-29.pdf) consolidates these results, compatibility checks, and caching tradeoffs with nine visualizations. See the [report source and build instructions](gpu-results-report/README.md).

This follows the [H100 compatibility check](2026-09-29-h100-cache.md).
The purpose is to exercise public workload structure on real vLLM/LMCache,
validate the replay machinery, and establish a serving baseline. ATFM control
is disabled. These runs do not establish a policy benefit or task accuracy.

**Outcome:** two complete traces passed (45 requests total); 39/40 BFCL
call-shape checks passed. The larger 119-request trace first exhausted workspace
quota, then reached its retry time limit with 92 records retained. It is not a
passing complete-trace result. Both Pods were stopped within the approved
aggregate compute allowance.

## Workload provenance and interpretation

The [reproduction guide](../development/gpu-public-workloads.md) records model,
source revisions, selection, commands and validation rules.

- Three complete roots from the public [SemiAnalysis Weka corpus](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)
  cover a short branch, a sequential session and multiple subagent groups.
  Their recorded totals are 164 requests, 12,282,816 input tokens and 51,786
  output tokens. The largest recorded input plus output is 118,677 tokens.
- [BFCL V4](https://gorilla.cs.berkeley.edu/leaderboard.html) supplies actual
  tool schemas and questions. The initial nine cases cover single functions,
  selection among functions and parallel calls. The expanded matrix adds
  parallel selection among multiple functions and uses ten cases per category.

Weka reconstructs prompt content from cache hashes; it does not contain the
original tool arguments or results. Its `in` values are cache-block-derived
counts, not exact target-tokenizer measurements. These selected roots lack
`input_types` annotations. Original source models are all replayed through one
Qwen model. We preserve recorded sizes and end-to-start delays without synthesis
caps, but closed-loop duration changes with the new model's service times.
The selected roots are a small feasibility sample, not representative sampling
or an official AgentX evaluation. The session run at `676bb9f` did not set an
AIPerf global seed, so its reconstructed text is not byte-reproducible from the
selected roots alone. The subsequent recipe pins seed 7 before any paired
policy study. BFCL requests already use seed 7 and temperature zero.

BFCL checks validate emitted names, JSON-object arguments, non-truncation and
at least two calls in parallel cases. They do not execute tools or grade argument
semantics. The first ten rows per category are a convenience sample. There is no
official BFCL accuracy score here.

## Serving environment

The original stopped Pod could not restart because its host had no free GPU.
A replacement H100 NVL 94GB was created at 03:40:16 UTC at the same approved
$3.19/hour compute rate, within the remaining aggregate two-hour budget.
Both Pods retain chargeable persistent storage while stopped.

The replacement runs Qwen/Qwen3-4B-Instruct-2507 at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, with a 131,072-token context,
two sequences, eager execution and 55% GPU memory utilization. LMCache MP uses
24 GiB CPU L1, 16-token chunks, LRU and filesystem L2. The workspace is a network
filesystem, not a local SSD. Native environments live on the container disk;
model downloads, cache files and evidence live in the workspace.

The pinned stack is vLLM 0.30.0, LMCache 0.5.5, PyTorch 2.13.0+cu130 and
AIPerf 0.13.0. vLLM's own prefix cache is disabled. The Hermes parser supplies
the OpenAI-compatible tool-call interface. GPU DCGM telemetry was unavailable;
server metrics and client request exports remain available. The server did not
enable per-response prompt-cache details, so external reuse must be read from
vLLM metrics rather than an absent client cached-token field.

## Replay issues caught by validation

1. Explicit fixed-schedule replay failed before inference: AIPerf's validator
   did not recognize timing inside nested Weka roots. Closed-loop replay uses
   the loader's recorded end-to-start delays instead of absolute timestamps.
2. A request-count stop condition admitted repeated roots while their children
   were still running. The short-branch run exported exactly 21 requests but
   reported three truncated children. The branch checks correctly rejected it.
   The run was interrupted through process cleanup, and the stopping condition
   changed to `--num-sessions 1` so one complete root tree can drain.
3. The first session-bounded run completed all 21 requests and its one child,
   but a missing expected-count argument in the harness stopped result handling.
   The raw complete export is retained. The fix has a per-case reporting
   regression test, and the full suite was restarted with seed 7.
4. The first large shared-stack case exceeded the 100 GB workspace quota.
   AIPerf reported `OSError(122, Disk quota exceeded)` at 04:20:38 UTC; log
   files also contained null-filled tails. About 73 GiB of L2 cache, 18 GiB
   of package-download cache and 7.6 GiB of model weights filled the volume.
   The shared-filesystem `df` output showed host-wide space and did not
   reveal the Pod quota. This partial case is invalid; the earlier completed
   cases remain separate accepted evidence. Metrics were captured on the
   container disk before interruption, and generated caches were reclaimed.
5. Combined run verdicts now include tool-call failures, not only trace cases.
   A regression test covers propagation of a failed tool case into its summary.

Failed and interrupted attempts are retained separately. A matching request
count alone is insufficient evidence that a public trace completed faithfully.

Generated KV binary files from stopped attempts were removed to preserve storage
headroom. Their logs, request/response records, metrics and provenance remain.

## Accepted seeded results

The accepted rerun uses commit `4bcc012` and reconstruction seed 7. Each root
runs once, with no request-count cutoff. Latencies below are per-request
observations from one session, not confidence intervals or cross-policy gains.
Server usage, rather than retokenized response text, is used for output totals.

| Case | Completed requests | Child groups completed | Server output tokens | Replay duration | TTFT p50 / p95 | External hit/query tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short branch | 21/21 | 1/1 | 3,949/3,949 | 143.93 s | 880.54 / 3,877.61 ms | 714,560 / 820,262 (87.11%) |
| Sequential | 24/24 | 0/0 | 10,202/10,202 | 292.73 s | 1,255.72 / 3,240.73 ms | 1,466,560 / 1,560,190 (94.00%) |

The short branch's request latency p50/p95 was 4,281.04/6,378.09 ms. Its final
CPU cache occupied 16,529,227,776 bytes (15.39 GiB), with no read or write locks.
Server prompt usage was 820,262 tokens versus 818,752 recorded block-derived
input tokens; reconstructed text, tokenization and chat framing change that count.

The sequential session's request latency p50/p95 was 10,066.22/17,003.67 ms.
Server prompt usage was 1,560,190 tokens versus 1,557,440 recorded input tokens.
Its final CPU occupancy was 19,329,712,128 bytes (18.00 GiB), with no read/write
locks. LMCache logged eviction after crossing its 80% L1 watermark at 04:08:35,
04:09:47 and 04:11:46 UTC. Cases share a serving stack and cache in the documented
order; they are not independently reset cold-cache trials.

The 16-token chunks and eager execution are compatibility settings, not a tuned
H100 throughput configuration. Larger chunk sizes and CUDA graphs should be
separate controlled factors before claiming an optimized serving baseline.

Local validation after the final runner fix: **603 passed, 3 skipped**, with
six existing warnings. The core ATFM policy code is unchanged.

The large case also logged 28–55 ms client event-loop delays and L1 batch
allocation pressure before the quota failure. These diagnostics motivate
separate client scheduling, cache-transfer and capacity measurements; they do
not by themselves identify the dominant latency cost.

## Expanded BFCL API matrix

The isolated matrix ran at `1c33fa7`: **39/40 call-shape checks passed**. All
40 requests returned saved responses. Single-function and multiple-function
selection cases passed 10/10 each; parallel calls passed 9/10; parallel selection
among multiple functions passed 10/10. Each category uses the first ten rows of
the pinned public source. Tools were not executed and semantic accuracy was not
scored.

`parallel_9` emitted one `find_movie_showing` call containing arrays for two
movies and times. Its schema permits arrays, but our parallel check requires
at least two calls. The pinned BFCL reference also contains two separate calls.
This is a confirmed call-shape discrepancy; it is not an official BFCL evaluation
or a demonstration that a real tool would reject the batched request. Keep the
raw response and reference alongside the verdict.

## Isolated large-session retry

After removing the 18 GiB package-download cache and stopped-run KV files, the
large root was retried alone at `f441de1`, with the same 24 GiB L1 and seed 7.
This is a fresh-cache retry, not a continuation of the invalid shared-stack case.
An outer `timeout --signal=INT --kill-after=30s 1110s` bounds the entire process,
including startup, to preserve time for evidence download and supervised Pod
shutdown within the original aggregate compute allowance. The trace is not
shortened and its recorded delays are not capped.

The supervisor returned exit code **124** after 1,110 seconds. The retained
JSONL contains **92 distinct, non-cancelled request records** and 30,433 server
output tokens, versus the complete trace's 119 requests and 37,635 output tokens.
No finalized complete-trace profile was produced. These retained records are a
partial subset, not an accepted latency distribution or a complete workload.
The owned serving processes stopped. The retry logged **19 L1 batch-allocation
warnings and no disk-quota errors**; reclaiming installation and old KV files
resolved the storage failure but did not remove CPU-cache pressure.

## Evidence

The [evidence guide](results/gpu-public-h100-2026-09-29/README.md) links readable
verdicts, pinned inputs, raw request/response and AIPerf archives, hardware
metadata and checksums. Failed attempts are retained alongside accepted cases.

## Capacity finding

The isolated retry reproduced L1 allocation warnings at 04:36:19 and 04:36:53
UTC without the previous quota exhaustion. One batch requested 4,153 blocks of
2,359,296 bytes and was short by 1,301 blocks. Preserve these warnings as
capacity diagnostics; a failed cache allocation is not automatically a failed
inference request.

The recorded LMCache layout is 147,456 bytes (144 KiB) per cached token. Thus
100,000 tokens occupy about 13.73 GiB of KV tensors, and two such contexts require
about 27.47 GiB before additional overhead. This exceeds the configured 24 GiB
CPU tier, whose LRU eviction watermark is 80%. High aggregate cache-hit rates do
not imply that overlapping long contexts can all remain resident.

The next controlled measurements should separate cold prefill, L2 load, CPU-to-GPU
transfer, and client scheduling time, then vary L1 capacity and chunk size with
the same pinned roots. A policy comparison should follow with fixed hardware,
cache settings and workload seeds, including warming cost and unsuccessful
prefetches. These runs do not establish which cost dominates or an ATFM speedup.

## Shutdown and rental accounting

All three evidence archives were downloaded and their SHA-256 values matched
the remote copies before stopping the replacement Pod. It was confirmed
`EXITED` by **04:47:58 UTC**; a subsequent check confirmed both Pods `EXITED`.
The original and replacement created-to-confirmed-stop windows total about
**117.4 minutes**, below the approved 120-minute aggregate compute allowance.
At the quoted $3.19/hour this is approximately **$6.24 compute**, not a provider
invoice. Storage, tax and billing adjustments are excluded. Both persistent
volumes remain allocated and chargeable. See the [rental outcome](results/gpu-public-h100-2026-09-29/rental-outcome.json).
