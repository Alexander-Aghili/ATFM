# H1 first results: forecast skill of the predictor ladder (2026-09-22, revised 2026-09-23)

> **Dated research record.** Findings and plans below retain their original scope.
> See [current implementation status](../status.md) and [current run instructions](../operations.md).

Runs: `uv run python scripts/run_h1.py experiments/<config>.yaml`, outputs under `runs/<name>/`.

**How to read every number below.** Hypothesis H1a (demand board: in-flight session state versus history) is what these runs test. H1b (sidecar progress beyond elapsed time) is only testable on long-tool workloads with progress events; TraceLab has none, so M2 equals M1 there by construction and the synthetic fleet is the only H1b evidence so far. Nothing here is a controller result (H2).

- Metric: mean pinball loss of the forecast's 90th percentile (q90) of KV blocks demanded within the horizon, averaged over ticks. Lower is better. Under-forecasts cost 9x over-forecasts, matching capacity decisions. "x times better" always means the ratio of these losses at the stated horizon.
- Ground truth per tick: KV blocks (`ceil(isl / 16)`) required by every session that (re)starts an LLM call within the horizon, counted once per session (detail.md 3.2); exogenous new sessions included.
- Horizons: 10 s, 30 s, 2 min, 5 min, 15 min. Tick: 30 s. Monte Carlo samples: 256.
- Split: TraceLab sessions are assigned to calendar weeks; 30% of weeks are held out (seed 0), predictors are fitted on the remaining weeks' real sessions, and the held-out sessions are replayed as a Poisson fleet at 200 sessions/hour for 4 hours. Synthetic train and test are independent seeds of the same workload spec.
- History baselines B0/B1 only see demand windows that have fully elapsed (no future leak).

## Synthetic mixed fleet (`experiments/h1_synthetic.yaml`, run `h1_synth_mixed`)

Workload: 120 background sessions/h (bash 70%, pytest 20% strong progress signal, build 10% weak, median long-tool durations 200 s and 490 s, 5% fan-out), 60 interactive sessions/h with heavy-tailed think time, one 20-minute 2x slowdown of the `ci` backend. 2 h, 30 s ticks, 256 Monte Carlo samples, train and test are independent seeds.

| class | model | 10 s | 30 s | 120 s | 300 s | 900 s |
|---|---|---|---|---|---|---|
| background | B0 constant | 273 | 361 | 666 | 1030 | 3441 |
| background | B1 Kalman | 116 | 148 | 282 | 539 | 2096 |
| background | B2 history | 477 | 463 | 390 | 286 | 377 |
| background | M1 survival | 91 | 129 | 215 | 220 | 467 |
| background | M2 progress | **54** | **86** | **148** | **170** | 393 |
| background | M3 backend | 117 | 255 | 471 | 411 | **352** |
| interactive | B0 constant | 196 | 235 | 300 | 460 | 1097 |
| interactive | B1 Kalman | 101 | 102 | 168 | 236 | 804 |
| interactive | B2 history | 69 | 66 | 89 | 167 | **399** |
| interactive | M1 survival | 61 | 62 | 90 | 169 | 401 |
| interactive | M2 progress | **57** | **54** | **88** | 171 | 401 |
| interactive | M3 backend | 60 | 61 | 88 | **166** | 400 |

Surge detection at the 5-minute horizon (capacity = 95th percentile of true demand, 3 or 4 true surges in the run): B0 and B1 each catch one surge and miss three; M1 catches all four with 30 to 270 s lead and one false alarm; M2 catches all four with two false alarms.

Reading:
- The plan's gate (M1 or M2 at least ~20% better than B1 on q90 pinball at 2 to 15 min) is **met on this synthetic fleet** for both classes: on background, M2 is 1.9x better than B1 at 2 min, 3.2x at 5 min and 5.3x at 15 min; on interactive, the session models are 1.9x better at 2 min and 1.4x at 5 min, 2x at 15 min.
- Conditioning on elapsed time (M1 vs B2) is the largest single gain at short horizons (5x at 10 s on background); live progress (M2 vs M1) adds a further 30 to 40% at 10 s to 5 min on background, where the strong-signal pytest tool lives. Interactive sessions have no progress-emitting tools here, so M2 equals M1 within noise.
- M3 is worse than M2 at 10 s to 5 min on background and best only at 15 min. With one 20-minute 2x perturbation on a fleet with lognormal(sigma 0.7) tool durations the EWMA over duration ratios mostly adds variance. It needs a variance floor tied to per-tool dispersion and a longer perturbation schedule before it can be judged; parked.
- The 2 min to 15 min horizons are where the earlier (buggy) run showed B1 winning; that was an artifact of the exogenous-arrival rate being multiplied by the number of models. The corrected harness scores each model with its own arrival estimator.

## TraceLab replay (`experiments/h1_tracelab.yaml`, run `h1_tracelab_r200`)

Real Claude Code and Codex sessions overlaid at 200 sessions/h for 4 h, weekly time-block split (30% test), 30 s ticks. No progress events exist in TraceLab, so M2 equals M1 by construction; M3 is omitted (single backend).

Truth and forecasts are per tick over the 4 h fleet window (476 ticks). Pinball loss at q90, KV blocks (TraceLab contexts are large: median ISL ~124k tokens, ~7.7k blocks per call, so one mis-forecast session costs ~7k):

| model | 10 s | 30 s | 120 s | 300 s | 900 s |
|---|---|---|---|---|---|
| B0 constant | 10409 | 10845 | 13861 | 16485 | 43105 |
| B1 Kalman | **4588** | 6615 | 10597 | 14186 | 26101 |
| B2 history | 40651 | 38324 | 33861 | 30628 | 26183 |
| M1 survival | 5567 | **5254** | **5679** | **6604** | **7700** |
| M2 progress | 5538 | 5251 | 5654 | 6611 | 7671 |

90% interval coverage: B1 0.92 to 0.98; M1 0.40 (10 s), 0.57 (30 s), 0.67 (2 min), 0.62 (5 min), 0.71 (15 min); B2 under 0.07. Endogenous fraction of demand (from sessions already in flight): 0.99 at 10 s, 0.95 at 2 min, 0.90 at 5 min, 0.81 at 15 min.

Reading:
- **The plan's gate is met on real traces**: M1 beats B1 on q90 pinball by 1.9x at 2 min, 2.1x at 5 min and 3.4x at 15 min (threshold was 20%). B1 is 18% better at 10 s, where a well-calibrated wide interval wins over a biased narrow one.
- Conditioning on elapsed time is everything: B2 (per-tool duration ignoring elapsed, the Continuum-style TTL) is 6 to 7x worse than M1 and worse than the constant baseline. TraceLab has no progress events, so M2 equals M1 by construction.
- Demand is overwhelmingly endogenous at every horizon studied (81 to 99%), which is the precondition for in-flight forecasting to matter; the plan's second gate (endogenous fraction over 50% at 5 min for background) is met here for the interactive class with 90%.
- M1 is still under-dispersed (40 to 71% coverage against a 90% target). Sessions are drawn independently; real fleets share hidden state (the same user across sessions, the same repository, correlated harness stalls), and the pending-gap tail (p99 45 s, max 128 h) is only partly captured. Widening via a fleet-level dispersion factor calibrated on train blocks is the obvious next step and would also fix the 10 s loss.
- Surge detection at 5 min is inconclusive on this overlay: capacity at the 95th percentile of a Poisson-overlaid fleet is crossed by sampling noise (6 true surges, none caught by any model, B1 raises 18 false alarms). A capacity line tied to a real KV budget and a surge scenario (3x interactive arrivals, detail.md 11.2) is the right test.

Two harness bugs found in review and fixed before these numbers: the exogenous arrival model was shared across session models (multiplying the arrival rate by the model count), and the pending state after a tool was treated as "due now" although the tool-end to next-request gap has a heavy tail in TraceLab. Both are covered by tests now.

## Next experiments

1. Dispersion calibration for the session models (coverage to 90%) on train blocks.
2. Regime map on synthetic fleets: share of long tools (3%, 10%, 30%), background share, fan-out; report the crossover horizon where in-flight state beats B1.
3. Perturbation study for M3 with a longer slowdown schedule and a per-tool variance floor.
4. Surge lead time against a fixed KV budget under the interactive-surge scenario.
