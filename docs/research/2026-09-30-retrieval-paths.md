# Retrieval paths (stage C): cold prefill vs L2 load vs warmed L1: 30 September 2026

Stage C of the [GPU results report](gpu-results-report/README.md) plan, run with
`atfm_experiments.gpu_cache.retrieval` at revision `0552406` (see the
[runbook](../development/gpu-public-workloads.md#retrieval-paths-stage-c)). One
Qwen3-4B-Instruct-2507 request with one output token per trial; three seeded,
shuffled blocks per GPU (seed 20260930), each on a fresh LMCache/vLLM pair with
48 GiB CPU tier, 16-token chunks and its own filesystem L2 directory.

**Outcome:** all 72 trials passed with the expected reuse (cold: zero external
hits; `l2` and `l1`: every token but the last served externally). A context
warmed into CPU memory ahead of time beat full recomputation at every length on
both GPUs, by 2.5-12.6x on the H100 and 1.8-21.6x on the A100. Loading the same
context from the filesystem tier on demand cost about the same wall time on both
GPUs and was slower than recomputing on the H100 at all lengths but 32k; on the
A100 it won from 32k tokens up. Reuse therefore pays only when the transfer is
moved off the request's critical path, and the break-even depends on the
hardware. Compute cost about 1.3 USD.

## Medians of three blocks

Client time for the one-token request; speedup against cold in parentheses.
"Warm" is the ATFM `TierDirective` prefetch from L2 into L1 that precedes the
`l1` request and is not included in its client time.

| H100 SXM (AP-IN-2) | 2,048 | 8,192 | 32,768 | 98,304 |
| --- | --- | --- | --- | --- |
| Cold | 0.27 s | 0.34 s | 4.26 s | 8.67 s |
| L2 on demand | 0.29 s (0.93x) | 0.95 s (0.35x) | 3.54 s (1.20x) | 10.43 s (0.83x) |
| L1 warmed | 0.09 s (3.0x) | 0.13 s (2.5x) | 0.34 s (12.6x) | 0.87 s (10.0x) |
| Warm time | 0.29 s | 0.87 s | 3.42 s | 9.77 s |

| A100 SXM (US-WA-1) | 2,048 | 8,192 | 32,768 | 98,304 |
| --- | --- | --- | --- | --- |
| Cold | 0.18 s | 0.71 s | 4.90 s | 24.77 s |
| L2 on demand | 0.35 s (0.52x) | 1.07 s (0.66x) | 3.62 s (1.35x) | 10.72 s (2.31x) |
| L1 warmed | 0.10 s (1.8x) | 0.17 s (4.3x) | 0.48 s (10.3x) | 1.15 s (21.6x) |
| Warm time | 0.25 s | 0.96 s | 3.94 s | 8.49 s |

L1 bytes after warming were 0.28, 1.12, 4.50 and 13.50 GiB, exactly 147,456
bytes (144 KiB) per token.

## Where the time goes

vLLM's own histograms place the cost of reuse in queue time, not prefill: with
external hits, `request_prefill_time` was 0.04-0.06 s at every length, while
queue time grew with the bytes to load (10.2-10.5 s at 98k from L2). Cold
prefill scaled with length and GPU: 7.0 s at 98k on the H100 against 22.8 s on
the A100. L2 load and L1 warm times were similar on the two hosts, consistent
with a host-side (filesystem and CPU-to-GPU) limit rather than GPU compute.

Implications for ATFM:

- Warming ahead of a predicted return is the only reuse path that wins at every
  length on both machines. At 98k it needs roughly 8.5-10 s of lead time.
- Whether to reload or recompute on demand is hardware-dependent: prefill speed
  and host transfer speed differ by about 3x between these Pods. A controller
  should calibrate both on its own hardware rather than use a fixed threshold.
- Warm time at 98k (9.8 s) exceeds H100 cold prefill (8.7 s), so a late warm
  on that GPU is worse than doing nothing.

## Limits

One Pod per GPU type and three blocks per cell; hosts with the same GPU have
differed by more than 1.5x in earlier rounds. The OS page cache was not dropped,
so L2 reads may come from host memory rather than the device; true cold-disk
loads can only be slower. vLLM's GPU prefix cache is disabled, so a GPU-resident
condition is not measured. Single isolated requests do not show contention
between warming and serving. On the H100, cold 32k requests spent about 3 s in
the queue before 1.1 s of prefill (one block: 1.9 s total); the cause, possibly
the preceding trial's asynchronous store, is not isolated.

## Evidence

[results/gpu-retrieval-2026-09-30](results/gpu-retrieval-2026-09-30/): per-GPU
`summary.json`, `trials.jsonl` (every trial with metric deltas), step logs, full
archives (manifests, per-trial LMCache status snapshots, server logs),
`hardware.json` and `sha256.json`. Both Pods were terminated; the account had
no Pods afterwards.
