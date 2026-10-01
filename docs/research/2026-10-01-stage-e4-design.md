# Stage E4 design: a cache-tier headline metric and a three-arm bracket: 1 October 2026

Follows [stage E2](2026-10-01-stage-e2.md) and the partial E3. The framing borrows from
the CacheBlend write-up "How CacheBlend improved the top tokenomics metric by 25x"
(Tensormesh/LMCache): name a concrete workload pattern where the default cache fails,
explain why, and report one cache metric before and after alongside TTFT.

## The failure case

**Agent fleet with long tool gaps.** Several agents share one CPU (L1) tier. While one
agent waits on a tool for 10 s or more, the others' traffic pushes its context out of L1
under LRU, so its next call must load the context from the filesystem tier (L2) on the
critical path. Forecast-driven warming moves that load before the return.

## Headline metric: CPU-tier share of reused chunks

LMCache 0.5.5 logs one line per lookup, `Prefetch request completed (L1+L2): k/n retained
keys (a L1, b L2) ... external_request_id=...`. vLLM's lookups carry a `chatcmpl-` id and
ATFM's warms an empty one, so `stage_e.tier_hits()` counts only vLLM's. The metric is
`a / (a + b)` summed over a run: the share of reused chunks already in CPU memory when the
request arrived. Also reported: requests that needed any L2 load.

Recomputed from the existing evidence (`results/*/tier-hits.json`):

| Run | L1 share of hits | Lookups needing L2 |
| --- | ---: | ---: |
| E2 direct-1 | 31.8% | 83 / 99 |
| E2 atfm-q50-2 | 81.2% | 29 / 95 |
| E2 atfm-q50-3 | 75.5% | 24 / 92 |
| E2 direct-4 | 33.7% | 78 / 97 |
| E3 A direct-1 | 14.0% | 82 / 99 |
| E3 B proxy-1 (other host) | 20.4% | 80 / 99 |

On one Pod, in a bracketed order, warming raised the L1 share from about 33% to 76-81%
and cut requests waiting on an L2 load from about 80 to 24-29. This is the mechanism the
TTFT gain needs (stage C: L2 loads are paid in vLLM queue time). The proxy-only run sits
near direct, but it ran on another host, so it is suggestive only.

## E4 plan

`experiments/gpu-cache/plans/stage-e4.txt`: on each of two Pods on separate hosts,
direct, proxy, atfm-q50, atfm-q50, proxy, direct (fleet case, 24 GiB L1, held-out training
table sha256 `2a6ef4a7...`, identical to E2). The symmetric bracket cancels linear drift for
every pair of arms. Report per arm: L1 share of hits, lookups needing L2, TTFT p50/mean,
vLLM queue time. About 2.9 h per Pod at 3.49 USD/h, about 22 USD in total.

Decision rule: warming is credited with the gain if, on both Pods, atfm beats proxy on
both the L1 share and median TTFT while proxy stays within the direct runs' spread.

Status: designed and unit-tested; not launched (awaiting run approval).
