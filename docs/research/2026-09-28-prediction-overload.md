# Bounded prediction work and deadline handling

The implementation bounds unfinished prediction jobs and falls back immediately
at capacity. The paired trials below show lower client p95 in all six pairs,
but **lower timely prediction coverage**. This is overload containment, not a
claim of improved scheduling quality or a completed solution to board latency.

## Problem and implementation

The old proxy submitted every prediction to a four-thread executor whose waiting
queue was unbounded. A 50 ms caller timeout did not stop running synchronous work;
the configured board HTTP timeout was at least 500 ms. Timed-out work could
therefore continue occupying scarce threads while more work was submitted.

`proxy/prediction.py:PredictionRunner` now owns admission, the executor, a
monotonic deadline, and cross-thread accounting. `prediction_limit` defaults to
four and limits all unfinished futures, not merely awaiting callers. Admission
and bookkeeping are O(1); unfinished prediction storage is O(limit). A future's
completion callback releases its slot. Cancelling an awaiting caller cannot
release a still-running worker's slot.

A full or closed runner immediately returns the same fallback used for other
prediction failures. Jobs expired at dispatch skip computation. Deadline-aware
board HTTP phases use the remaining caller budget; the async caller also checks
the deadline before using a result. Legacy synchronous predictor interfaces
remain supported; their two-call path checks expiry before the second estimate.
Python threads cannot be killed safely. Blocking custom predictors must eventually
return, and event-loop stalls can delay timeout handling. Remote board computation
can continue after its caller disconnects. Synchronous board tick/planning work
and mutable-state isolation are deliberately outside this change.

The application lifespan stops admission and drains the executor before closing
its owned board/upstream clients and trace output. Injected clients remain
caller-owned. The load harness now composes that lifecycle instead of duplicating
proxy resource cleanup. Queue formulas and fallback estimates are unchanged;
which predictions arrive in time can change, so scheduling behavior is not
claimed identical under overload.

## Paired HTTP trials

Baseline source: `670a1fa`. Measured candidate source: `40602f3`.
A subsequent bookkeeping correction (`c4bfc78`) makes caller outcomes and the
pending decrement atomic; it does not change admission rules. The measurements
remain attributed to the measured revision, rather than relabeled as final-code timings. Both used the same Python
environment and installed dependencies, the same host, and separate worker,
board and proxy processes for each case. Sizes were 256 and 1,024 sessions, with
seeds 7–9 paired across versions and alternating pair order. Trials ran
sequentially. CPU affinity, frequency scaling and unrelated host activity were
uncontrolled. These are finite local workloads, not sustained capacity estimates.

Settings: three sequential calls per session; session starts over two seconds;
burst-aligned tool returns; 64 fake-worker slots with 5 ms service; proxy window
64; 50 ms prediction budget; 128 forecast draws; ten slots; periodic control on;
control sleeps 500 ms after each step. Client keepalive was disabled by the
existing harness defaults. The shell and children used a soft descriptor limit
of 8,192. Both versions used four prediction threads; the candidate also capped
unfinished jobs at four.

Ranges are the minimum and maximum of three per-run p95s or CPU totals, **not**
confidence intervals. Prediction fractions pool counts within each version/size.

| Sessions | Version | Client p95 (s) | Predictions used | Caller timeout | Immediate rejection | Proxy CPU (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | Before | 0.503–0.825 | 58.42% | 41.58% | 0% | 2.252–2.509 |
| 256 | After | 0.427–0.557 | 52.00% | 11.02% | 36.98% | 2.169–2.309 |
| 1,024 | Before | 2.311–3.465 | 32.26% | 67.74% | 0% | 6.790–8.646 |
| 1,024 | After | 1.851–2.926 | 18.98% | 9.33% | 71.69% | 6.353–7.212 |

All 23,040 planned calls completed successfully. No session, transport, probe,
tool-publication or control errors were reported during the workload. Every
candidate run recorded an exact peak of four outstanding jobs and zero outstanding,
running and pending work at final observation. Its accepted/completed counts
balanced. The baseline did not measure outstanding jobs; its missing peak must
not be read as zero.

The improvement in p95 coexists with reduced use of forecasts. In the largest
cases, most requests immediately bypass prediction rather than waiting for a
result that may expire. Proxy CPU fell in five pairs and rose in one; the data
supports neither a universal CPU speedup nor better policy decisions. Latency
includes the load generator, HTTP, scheduling, forecasting and fake service.
It does not measure GPU execution, cache movement or real-model serving.

### Shutdown finding

All baseline cases logged `NameError: _close_proxy is not defined` during
application shutdown: that helper was defined after the module's blocking
`main()` call. The candidate removes this harness cleanup path and uses the
proxy application lifespan; its logs contain no shutdown errors. These baseline
failures occurred after workload summaries were collected and are retained,
not silently repaired. All child processes were reaped; recorded exit codes are
`-15` following the harness's SIGTERM, not successful-zero exit codes. A new
real-HTTP test assertion checks shutdown logs as well as process termination.

## Evidence and reproduction

[Archived artifacts](results/prediction-overload-2026-09-28/) include all twelve
case directories, configurations, summaries, process observations, requests,
proxy/event/control traces, logs and shutdown records. Large text inputs are
gzip-compressed. `manifest.json` records uncompressed byte lengths and SHA-256
hashes for 240 files; `comparison.csv` preserves all individual pairs.

`executed-driver.txt` is the exact driver used, with its original absolute paths.
To rerun, use fresh checkout/output locations and the same environment:

```bash
# Create isolated source checkouts; run each with the same existing interpreter.
git worktree add --detach /tmp/atfm-before 670a1fa
git worktree add --detach /tmp/atfm-after 40602f3
# Set PYTHONPATH to the chosen checkout's src and experiments/src.
# Run sequentially, alternating before/after order across seeds 7, 8, 9.
ulimit -Sn 8192
python -m atfm_experiments.load --sessions 1024 --patterns burst \
  --config headroom.json --out runs/prediction-overload-new
```

`headroom.json` contains `{"seed":7,"worker_slots":64,"worker_service_s":0.005,
"proxy_window":64}`. Repeat at 256 sessions and with seeds 8 and 9, using fresh
output directories. Install serving/test dependencies as described in the load
guide. The original baseline was a `git archive` beneath the working repository;
its environment's `git_commit` therefore reflects the enclosing repository, not
the archived source. The source revision above and recorded per-file hashes
identify the actual code. Do not use that metadata field alone to reproduce it.

## Validation and next step

The final full suite passed **442 tests with five skips** and six existing
warnings. All 17 overload-focused tests pass. Coverage includes burst admission,
retained permits after caller timeout/cancellation, cancellation before dispatch,
expiry before execution, capacity recovery, submission failure, shutdown ownership,
configuration validation, remaining HTTP budgets, legacy predictor behavior and
atomic caller accounting. Real HTTP lifecycle tests now also reject shutdown
errors in server logs. All tracked Python functions remain within 20 physical
lines. Existing queue and seeded simulation tests pass.

The next experiment should isolate synchronous board work and serve bounded,
versioned read-only prediction state. Measure useful predictions per attempted
request alongside proxy latency and snapshot age. Raising the worker limit alone
may replace immediate fallback with more stale work; it requires new paired
measurements rather than being an automatic improvement.
