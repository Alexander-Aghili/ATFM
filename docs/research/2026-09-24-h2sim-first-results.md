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
