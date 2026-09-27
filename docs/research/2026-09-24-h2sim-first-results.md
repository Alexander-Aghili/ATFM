# H2 in the closed-loop simulator: first six-arm results (2026-09-24)

**Status: synthetic workload, unvalidated engine model. Only the relative ordering of arms is meaningful; no absolute number here transfers to an H100.** The simulator (`src/atfm/sim`, branch `l0-sim`, reviewed and fixed on 2026-09-24) exists to answer *where each arm queues work and what a hold costs*, so that the H100 study (step 4 of the agreed sequence) measures the right things. Runner: `scripts/run_h2sim.py`, configs `experiments/h2sim_regimes.yaml`, `h2sim_short_tool.yaml`, `h2sim_loaded.yaml`; per-arm logs under `runs/h2sim/<name>/log_<arm>_<seed>.parquet`.

## Setup

- One engine: 8000 KV blocks, batch 8, prefill 20k tok/s, decode 40 tok/s, priority scheduling (Dynamo-style `priority` field honoured, head-of-line order kept). KV is an LRU with prefix reuse; a session whose blocks were evicted recomputes its whole prefix.
- Programs: synthetic (`atfm.sim.programs`), interactive sessions at 120/h (6 turns, 20 s think time, bash tools ~3 s), background at 240/h (10 turns, deadline 1800 s). `long_tool` regime: background tools bash 3 s / pytest 180 s (strong signal, 5% spawn) / build 400 s (weak signal). `short_tool`: bash 2 s / pytest 20 s. 3600 s of arrivals, 3 seeds, paired bootstrap on per-session metrics (native is the reference).
- Arms: `native` (engine priority only, no proxy window), `proxy_rules` (window 8, class tiers, no holds), `forecast_M1` / `forecast_M2` (rules + board forecast + ground-delay hold when the 90th-percentile forecast of interactive demand over the next 30 s exceeds free capacity or free slots, hold capped at 600 s, the core default `max_hold_s`, and re-consulted at expiry so a hold lasts only while the constraint binds), `forecast_M2_nohold` (forecast only in the index, never holds), `oracle` (sees the event heap: perfect knowledge of arrivals in the next slot, same hold cap), `working_set` (ThunderAgent-style: pause background programs when the working set exceeds 6000 blocks, offload their KV, resume with hysteresis).
- Cost columns are measured, not assumed: `hold_s` is time under a policy hold (window queueing is separate), `hold_kv_block_s` is the held session's resident blocks times hold time until eviction, `evictions_caused` are displacement evictions attributed to held sessions, `evictions_to_admit` are the admitting request's own evictions, `recomputed_prefill_tokens` is prefix recompute after eviction, `caps` counts holds that ran to the 600 s cap.

## long_tool regime (`h2sim_long_tool`, 3 seeds)

| arm | TTFT after tool p95 (s) | SLO (2 s) sessions | background JCT mean (s), diff vs native [95% CI] | deadline hit | recomputed prefill tokens | hold KV block-s | mean hold s (bg) | caps |
|---|---|---|---|---|---|---|---|---|
| native | 0.437 | 0.999 | 1297 | 0.728 | 9.98e6 | 0 | 0 | 0 |
| proxy_rules | 0.422 | 0.999 | +0.0 [-0.05, +0.09] | 0.728 | 9.99e6 | 0 | 0 | 0 |
| forecast_M1 | 0.340 | 0.999 | +17.0 [14.8, 19.1] | 0.725 | 1.03e7 | 5.6e5 | 1.9 | 0 |
| forecast_M2 | 0.327 | 1.000 | +16.2 [14.2, 18.1] | 0.725 | 1.03e7 | 5.5e5 | 1.8 | 0 |
| forecast_M2_nohold | 0.453 | 0.998 | +0.0 [-0.05, +0.08] | 0.728 | 1.00e7 | 0 | 0 | 0 |
| oracle | 0.357 | 0.999 | +8.4 [7.2, 9.6] | 0.728 | 1.01e7 | 3.3e5 | 1.0 | 0 |
| working_set | 0.040 | 1.000 | +3847 [3720, 3968] | 0.123 | 1.58e7 | 0 | 423 | 1508 |

## short_tool regime (`h2sim_short_tool`, 3 seeds)

| arm | TTFT after tool p95 (s) | SLO (2 s) sessions | background JCT mean (s), diff vs native [95% CI] | recomputed prefill tokens | hold KV block-s | mean hold s (bg) |
|---|---|---|---|---|---|---|
| native | 0.312 | 1.000 | 105 | 2.99e6 | 0 | 0 |
| proxy_rules | 0.194 | 0.998 | +0.0 [-0.04, +0.05] | 2.99e6 | 0 | 0 |
| forecast_M1 | 0.040 | 1.000 | +20.2 [17.5, 22.7] | 3.30e6 | 1.57e6 | 2.0 |
| forecast_M2 | 0.040 | 1.000 | +19.8 [17.2, 22.1] | 3.21e6 | 1.60e6 | 2.0 |
| forecast_M2_nohold | 0.230 | 0.998 | +0.0 [-0.04, +0.05] | 2.99e6 | 0 | 0 |
| oracle | 0.130 | 0.998 | +11.4 [9.6, 13.1] | 3.17e6 | 9.7e5 | 1.1 |
| working_set | 0.651 | 0.990 | +16.8 [14.8, 18.8] | 6.23e6 | 0 | 1.4 |

## loaded long_tool regime (`h2sim_long_tool_loaded`, 3 seeds)

Rates 3x interactive (360/h) and 2.5x background (600/h) on a smaller engine (6000 blocks, batch 6, window 6, working-set budget 4500). The window binds here: native's session SLO drops to 0.85 and background deadlines are mostly missed by every arm.

| arm | SLO (2 s) sessions, diff vs native [95% CI] | TTFT after tool p95 (s) | background JCT mean (s), diff vs native [95% CI] | deadline hit, diff | recomputed prefill tokens | hold KV block-s | mean hold s (bg) | caps |
|---|---|---|---|---|---|---|---|---|
| native | 0.851 | 2.68 | 3310 | 0.190 | 4.79e7 | 0 | 0 | 0 |
| proxy_rules | 0.867, +0.014 [-0.002, +0.028] | 3.01 | -35 [-90, +17] | +0.057 | 3.82e7 | 0 | 0 | 0 |
| forecast_M1 | 0.941, **+0.088 [+0.074, +0.101]** | 2.16 | +790 [736, 842] | -0.071 | 4.33e7 | 4.2e6 | 157 | 1218 |
| forecast_M2 | 0.946, **+0.093 [+0.080, +0.105]** | 2.12 | +789 [739, 837] | -0.068 | 4.31e7 | 4.2e6 | 157 | 1193 |
| forecast_M2_nohold | 0.841, -0.013 [-0.028, +0.001] | 3.23 | -35 [-97, +10] | +0.053 | 4.14e7 | 0 | 0 | 0 |
| oracle | 0.923, +0.070 [+0.054, +0.083] | 2.51 | +1141 [1090, 1190] | -0.099 | 4.10e7 | 2.2e6 | 119 | 805 |
| working_set | 0.929, +0.076 [+0.063, +0.089] | 2.29 | +2082 [2035, 2136] | -0.075 | 4.66e7 | 0 | 368 | 1096 |

Under load the forecast arms buy 9 points of interactive SLO for a 24% increase in background completion time, 7 points fewer background deadlines met, 4.2 million held KV block-seconds and about 1200 holds per seed that ran to the 600 s cap (held calls then still wait at the window: median proxy queue 579 s for held calls against 1 s for unheld background calls). Rules alone (`proxy_rules`) and the class-aware index without holds are within noise of native on the SLO and slightly better for background. The working-set baseline reaches a similar SLO at 2.6x the background cost of the forecast arms.

## Reading

1. **The two base regimes are underloaded for the SLO metric; the loaded regime is where the arms separate.** Session SLO attainment at a 2 s TTFT bound is 0.99 to 1.00 for every arm, so the SLO column cannot rank anything. The tail metric that does move is TTFT after a tool returns (p95): forecast holds cut it 22% (long_tool) and 8x (short_tool) against native. In the loaded regime the forecast arms gain 9 SLO points over native with non-overlapping CIs, and the ordering forecast > working_set ≈ oracle > rules ≈ native holds on every seed.
2. **Holds are what buy the tail, and they are not free.** The `forecast_M2_nohold` ablation, identical to `forecast_M2` except that it never holds, is indistinguishable from `proxy_rules`: the class-aware index alone does nothing for the tail in these regimes. The holds cost background sessions 16 to 20 s of job completion time (1.3% in long_tool, 19% in short_tool), 3 to 10% more recomputed prefill, and 0.5 to 1.6 million held KV block-seconds. This is the trade the H100 study has to measure with real KV pressure; the simulator's KV model is the least validated part.
3. **M1 and M2 do not separate here.** Their TTFT, JCT and hold costs are within noise of each other. That is expected: the hold decision keys on interactive demand over the next 30 s, and the interactive class in these programs runs only ~3 s bash tools, so tool-progress prediction (M2's edge, shown in H1b) never enters the hold rule. Separating M1 from M2 in closed loop needs a regime where interactive sessions run long tools with progress signals, or a hold rule that targets background resumption. This is a design gap to close before the H100 study, not a negative result for M2.
4. **The oracle is not an upper bound on the forecast arms.** It holds less (lower cost) but gets a worse tail than the forecast arms in both regimes. Its rule is a different heuristic (occupancy of the next slot from the true event heap) rather than a quantile of a demand forecast, so the comparison says the GDP-lite rule is more aggressive than perfect knowledge of the next 30 s, not that the forecasts beat perfect information. A like-for-like oracle (same rule, true demand instead of forecast samples) is the right upper bound and is deferred.
5. **The working-set baseline behaves like ThunderAgent-style pausing should, and it is disastrous for background work under long tools.** In long_tool it pauses background programs whenever the working set exceeds budget, offloads their KV, and the pause/resume cycle triples background JCT (+3847 s) and drops deadline hits from 0.73 to 0.12 while recomputing 58% more prefill; 1508 holds ran to the 600 s cap. It gets the best interactive tail (0.040 s) because paused programs free the batch. In short_tool it is worse than native on every column. This is the case for demand-aware holds over occupancy-triggered pausing, in the simulator's own terms; the H100 study must confirm the recompute and JCT costs are real.
6. **Where each arm queues.** `queue_proxy_share` is 1.0 for every proxy arm and 0 for native: with window = batch size, all waiting happens at the proxy and the engine never queues. That is the intended design (the proxy is the only place holds can be applied), and it means the engine-side priority field only matters when the window is larger than the batch.

## What changes in the H100 study design

- Use the loaded regime's rates as the starting point, not the base rates; the base rates never stress a 2 s SLO.
- Add an interactive-long-tool regime (interactive sessions running 60 to 300 s tools with progress output) so M1 and M2 can separate in closed loop.
- Record on the real engine exactly the columns measured here: hold seconds, held KV block-seconds, evictions attributed to holds, recomputed prefill after resumption, and caps.
- Replace the heap-reading oracle with a same-rule oracle fed true demand.

## Second batch (2026-09-27): interactive long tools, a like-for-like oracle, a 60 s cap

Three changes since the first batch: an `interactive_long_tool` regime in which interactive sessions also run 120 s test suites (strong signal) and 300 s builds (weak signal), so that *when* interactive sessions return is what the hold rule must predict; an `oracle_rule` arm that runs the same GDP-lite rule and index as the forecast arms on the true first-call demand per horizon read from the event heap; and `max_hold_s` as a config field, set to 60 s for these runs. Loaded rates (300 interactive and 480 background sessions/h on the 6000-block, batch-6 engine).

### interactive_long_tool, cap 60 s, 3 seeds (`h2sim_interactive_long`)

| arm | SLO sessions, diff vs native [95% CI] | TTFT after tool p95 (s) | background JCT diff (s) | deadline hit, diff | held KV block-s | mean hold s (bg) | share of bg calls held | caps |
|---|---|---|---|---|---|---|---|---|
| native | 0.867 | 2.54 | 2309 | 0.320 | 0 | 0 | 0 | 0 |
| proxy_rules | 0.887, **+0.020 [+0.006, +0.038]** | 2.72 | +48 [-5, +100] | **+0.080** | 0 | 0 | 0 | 0 |
| forecast_M1 | 0.882, +0.015 [0.000, +0.030] | 2.79 | +275 [228, 321] | -0.015 | 3.2e6 | 42 | 0.42 | 2802 |
| forecast_M2 | 0.877, +0.011 [-0.005, +0.025] | 2.79 | +293 [244, 340] | -0.012 | 3.3e6 | 43 | 0.42 | 2845 |
| forecast_M2_nohold | 0.868, +0.001 [-0.014, +0.017] | 2.96 | +60 [16, 115] | +0.068 | 0 | 0 | 0 | 0 |
| oracle (heap, occupancy rule) | 0.874, +0.008 [-0.007, +0.022] | 2.84 | +962 [899, 1031] | -0.192 | 2.2e6 | 32 | | 2303 |
| oracle_rule (same rule, true demand) | 0.877, +0.010 [-0.004, +0.026] | 2.75 | +1003 [941, 1075] | -0.195 | 2.4e6 | 35 | 0.62 | 2596 |
| working_set | 0.893, +0.026 [+0.010, +0.039] | 2.47 | +611 [557, 676] | -0.060 | 0 | 58 | | 684 |

Reading, and it is a negative result for the hold lever in this regime:

1. **Holds do not buy the interactive tail here, and perfect information does not rescue them.** `oracle_rule` holds 62% of background calls with the true demand in hand and gains 1 SLO point for +1003 s of background completion time and 19 points fewer deadlines. The forecast arms hold 42% of calls for a similar, statistically marginal gain at a third of that cost. Whatever the forecast arms lose against the oracle is forecast error; here they lose nothing, because the rule itself is not the right lever.
2. **Why.** Interactive calls wait 2.5 s at p95 at the proxy in every proxy arm, and that wait is interactive-on-interactive: the window of six is filled by interactive calls with large contexts (4000 tokens growing 800 per turn, 150 output tokens at 40 tok/s). Holding background calls frees slots that other interactive calls take before the held call would have run, so the tail does not move. The hold lever pays when deferrable work is what occupies the pool (the loaded long_tool regime, where it gained 9 points); it cannot pay when interactive load alone saturates the window. A rule that recognises capacity freeing up within the slot (v2 below) reduces the needless holds; it cannot create capacity.
3. **Rules alone are the best trade in this regime.** Class tiers and the index with no holds gain 2 SLO points and 8 points of deadlines at +48 s background time (CI includes zero). The no-hold forecast ablation is again within noise of rules.
4. **M1 and M2 still do not separate** (+0.015 vs +0.011, overlapping intervals), for a new reason: under this load the rule's decision is almost always "hold" (2800 caps per seed at a 60 s cap), so the quality of the forecast never enters. The GDP-lite v1 test compares forecast demand with *instantaneous* free capacity, which under load is zero at every moment.
5. **The working-set baseline gets the best SLO** (+0.026) at +611 s background time, by pausing whole programs and letting the window drain; its holds are 58 s on average but it holds fewer calls.

### GDP-lite v2

The rule now counts, as capacity for the slot, the running requests expected to finish inside it (`freeing_capacity`: start + E[S] for the forecast arms, the engine's true end for `oracle_rule`), so that under load the decision depends on the forecast rather than collapsing to "always hold". Reruns of both regimes with v2 are `h2sim_interactive_long_v2` and `h2sim_long_tool_loaded_cap60_v2`.

#### interactive_long_tool with v2 (`h2sim_interactive_long_v2`, cap 60 s, 3 seeds)

| arm | SLO diff vs native [95% CI] | bg JCT diff (s) | deadline diff | held KV block-s | mean hold s (bg) | caps |
|---|---|---|---|---|---|---|
| proxy_rules | +0.020 [+0.006, +0.038] | +48 [-5, +100] | +0.080 | 0 | 0 | 0 |
| forecast_M1 (v2) | +0.001 [-0.015, +0.015] | +73 [28, 119] | +0.053 | 4.2e5 | 6.6 | 460 |
| forecast_M2 (v2) | +0.008 [-0.008, +0.023] | +83 [41, 133] | +0.053 | 7.0e5 | 10.7 | 756 |
| forecast_M2_nohold | +0.001 [-0.014, +0.017] | +60 [16, 115] | +0.068 | 0 | 0 | 0 |
| oracle_rule (v2, true completions) | -0.004 [-0.020, +0.015] | +878 [822, 948] | -0.191 | 1.2e5 | 2.0 | 95 |
| oracle (heap) | +0.008 [-0.007, +0.022] | +962 [899, 1031] | -0.192 | 2.2e6 | 32 | 2303 |
| working_set | +0.026 [+0.010, +0.039] | +611 [557, 676] | -0.060 | 0 | 58 | 684 |

v2 does what it was meant to do on the cost side and nothing on the benefit side: the forecast arms now hold a sixth as often (460 to 756 caps per seed against 2800), their background cost falls from +275/+293 s to +73/+83 s, their deadline rate rises 5 points instead of falling, and their held KV block-seconds drop 5 to 8x. Their SLO gain stays within noise of zero. So (b) stands: in this regime the interactive tail is not something admission of background calls can move, with any forecast.

**An unexpected attribution.** `oracle_rule` under v2 holds almost nothing (95 caps, 2 s mean hold, 1.2e5 held block-seconds) and still costs +878 s of background completion and 19 points of deadlines. Its holds cannot explain that; the only other thing it does differently from `proxy_rules` is the index term: like `oracle`, it uses the *true* duration of the tool each turn will launch as E[next tool]. Serving first the sessions about to vanish into long tools (the index's intent) starves the rest when that knowledge is exact. The heap `oracle` arm shows the same +962 s. This says the index's next-tool term, as weighted, is harmful to background completion, and that the forecast arms are protected from it only because their pooled or tool-conditioned estimate is coarse. A diagnostic arm (`oracle_rule` with the pooled E[next tool]) would isolate it; it is the next simulator run after the KV placement arms.

### long_tool loaded, cap 60 s (`h2sim_long_tool_loaded_cap60`)

Same regime and rates as the first-batch loaded run, hold cap 60 s instead of 600 s, `oracle_rule` added.

| arm | SLO sessions, diff vs native [95% CI] | TTFT after tool p95 (s) | background JCT diff (s) | deadline hit, diff | held KV block-s | mean hold s (bg) | caps |
|---|---|---|---|---|---|---|---|
| native | 0.853 | 2.68 | 3310 | 0.190 | 0 | 0 | 0 |
| proxy_rules | 0.867, +0.014 [-0.002, +0.028] | 3.01 | -35 [-90, +17] | +0.057 | 0 | 0 | 0 |
| forecast_M1 | 0.852, -0.001 [-0.016, +0.015] | 3.23 | +124 [68, 180] | +0.017 | 3.7e6 | 35 | 2764 |
| forecast_M2 | 0.843, -0.010 [-0.024, +0.004] | 3.14 | +136 [81, 187] | +0.018 | 3.7e6 | 35 | 2768 |
| forecast_M2_nohold | 0.841, -0.013 [-0.028, +0.001] | 3.23 | -35 [-97, +10] | +0.053 | 0 | 0 | 0 |
| oracle (heap) | 0.852, -0.001 [-0.015, +0.013] | 3.06 | +1150 [1085, 1213] | -0.129 | 1.8e6 | 25 | 2392 |
| oracle_rule | 0.842, -0.011 [-0.024, +0.004] | 3.17 | +1162 [1099, 1226] | -0.130 | 1.9e6 | 26 | 2585 |
| working_set | 0.870, +0.017 [+0.002, +0.032] | 2.62 | +505 [445, 566] | -0.021 | 0 | 58 | 4049 |

**The first-batch gain was the long cap.** With holds limited to 60 s the forecast arms' +9 SLO points vanish (-0.001 and -0.010, intervals spanning zero) while they still pay +124 to +136 s of background time and 3.7 million held KV block-seconds; the same-rule oracle with true demand does no better (-0.011 at +1162 s). At a 600 s cap the "hold" was in effect a ten-minute deferral of most background work, which is what moved the tail. Read together with the interactive-long regime: in this simulator, short forecast-driven holds do not improve the interactive SLO under either load pattern; long deferrals do, at a cost comparable to pausing; rules alone give a small consistent gain; the working-set baseline has the best SLO at moderate cost in both.

**What this changes.** H2 as stated ("a proxy that admits, orders and holds calls against the forecast improves the interactive tail at a stated cost") is not supported by the simulator for short holds. Two candidate reasons remain open and the v2 reruns test the first: (a) the v1 rule collapses to "always hold" under load, so the forecast never enters the decision; (b) the interactive tail in these regimes is set by service-time variance and interactive-on-interactive queueing, which no admission policy on background can fix. If v2 holds selectively and the tail still does not move, (b) stands and the lever the forecast should drive is not admission but KV placement (keep the returning sessions' KV resident, evict the held ones), which is what the H100 study can measure and the simulator's engine cannot yet.
