# Stage E design: forecast-driven LMCache warming on real GPUs

Date: 2026-09-30. Status: approved direction ("build and unit-test first, then ask
before renting"); details below are the implementation contract.

## Question

Does ATFM, predicting when an agent session will return and warming its KV from
the LMCache filesystem tier into CPU memory ahead of that return, lower request
TTFT and session duration against ordinary LMCache on the same Pod, after its
own costs (proxy hop, wasted warms, transfer contention)?

Stage C measured the mechanism: a context warmed ahead of time beat recompute at
every length (2.5-21.6x), on-demand L2 loads did not reliably, and warming 98k
tokens takes about 8.5-10 s. Stage E tests whether forecasts arrive early and
accurately enough to exploit that in a live replay.

## Arms (paired within one Pod, seeded random order, >=2 runs each)

| Arm | Path | Control |
| --- | --- | --- |
| `direct` | AIPerf -> vLLM | none (round 2/3 baseline) |
| `proxy` | AIPerf -> ATFM proxy -> vLLM | proxy only, no board: isolates the hop's cost |
| `atfm` | AIPerf -> ATFM proxy -> vLLM | board + control loop issuing prefetch directives to LMCache |

Workload: the 119-request multi-branch Weka root at a 24 GiB CPU tier, where
round 3 showed pressure (89.5% hit ratio, allocation warnings). A secondary
workload may shrink L1 on the sequential root to force evictions. Primary
outcomes: request TTFT distribution and replay duration. Secondary: warms issued,
completed, too late (request arrived first) and unused; warmed bytes; vLLM queue
time; external hit ratio.

## Components

1. **PrefetchPlanner** (`atfm.control.prefetch`). Per session not in
   `llm_running`, with context `n` tokens and resumption quantiles (q10, q50,
   q90) in seconds from now: lead(n) = overhead_s + n * bytes_per_token /
   warm_bytes_per_s. Issue `TierDirective(action='prefetch', tier='cpu')` when
   q10 <= lead(n) + interval_s + margin_s (start now or be late), unless q90 <
   lead(n) (the warm cannot finish before even a late return; stage C shows a
   late warm is worse than none on fast GPUs). At most one warm per session turn,
   an in-flight byte budget, and a per-plan cap. Calibration constants come from
   stage C per host (bytes_per_token 147,456; warm rate ~1.4 GB/s; overhead).
2. **Gap phase for proxy-only sessions** (`SessionRegistry(gap_after_done=True)`).
   Without tool events a finished call leaves `llm_pending` with the pooled
   post-tool gap, which is ~0 when trained on AgentX (gaps are labelled
   `__gap__` tools). With the flag, `llm.done` starts a synthetic `__gap__` tool
   phase so the elapsed-conditioned `__gap__` duration model applies; the next
   `llm.request` records it in tool history.
3. **Asynchronous actuator** (`LMCacheConfig.wait_for_completion=False`). Submit
   the prefetch and return `submitted`; `release_expired` reconciles completion
   each step. Completion outcomes are counted for the evidence log. The fixed 2 s
   synchronous wait blocked the loop while real warms take 3-10 s.
4. **Held-out training table.** Build the board's `--train` TraceTable from the
   AgentX/Weka corpus with the three selected roots (and their children) removed.
5. **Proxy fixes.** Read OSL from `max_completion_tokens` (AIPerf's field) before
   `max_tokens`.
6. **Replay arms.** `replay --arm {direct,proxy,atfm}` starts the proxy, board and
   control loop as owned processes; AIPerf targets the proxy with
   `--session-header x-atfm-session` and `--server-metrics <vllm>/metrics`.
   Proxy call traces, board events and the control log join the evidence.

## Local smoke finding (2026-09-30)

A local run with fake vLLM/LMCache and the real board, proxy and control loop
(held-out table, 3 sessions x 6 turns) issued 18 warms, all completed, 15 before
the session's next request and 3 for sessions that never returned. The
`__gap__` model's q10 is ~0.5 s, so a q10 trigger warms right after every call,
when the context is usually still resident; eviction happens later in the gap.
The planner therefore also supports `trigger='q50'` and `rewarm_after_s`, and
stage E compares `atfm-q10` with `atfm-q50` (re-warm after 30 s).

## Out of scope for this step

GPU-tier placement (vLLM prefix cache stays disabled), holds/GDP and touches
(not configured in the `atfm` arm), parent/child session linkage, an oracle
arm (possible follow-up: the same planner fed true next-arrival times).

## Risks

Forecast error on three roots from one corpus; warming competing with on-demand
loads for host bandwidth; the proxy's message memory differing from the served
prompt (checked by comparing prefetch keys with external hits).
