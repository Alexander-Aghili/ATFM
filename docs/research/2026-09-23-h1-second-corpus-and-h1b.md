# H1a on two corpora with seed and rate sweeps, dispersion calibration, and the first per-session H1b result (2026-09-23)

Reads with the same conventions as `2026-09-22-h1-first-results.md`: pinball loss at q90 of KV blocks demanded within the horizon, truth counts each session once per horizon, held-out splits, history baselines only see elapsed windows.

## H1a: TraceLab, 3 seeds x 3 overlay rates

Held-out calendar weeks (30%), replayed at 100, 200 and 400 sessions per hour for 4 h; mean over seeds 0 to 2. Ratio of the Kalman baseline's loss to the elapsed-time model's loss (values above 1 favour in-flight state):

| sessions/hour | 10 s | 30 s | 2 min | 5 min | 15 min |
|---|---|---|---|---|---|
| 100 | 1.08 | 1.25 | 1.63 | 1.96 | 3.00 |
| 200 | 1.03 | 1.50 | 2.11 | 2.53 | 3.96 |
| 400 | 1.01 | 1.70 | 2.55 | 3.31 | 5.09 |

M1 is never worse than B1 at any cell, and the advantage grows with fleet density and horizon. 95% CI half-widths over the three seeds are 10 to 25% of the means for M1 (`runs/sweep_tracelab.csv`). M2 equals M1 within noise on TraceLab (no progress events).

## H1a: AgentX (SemiAnalysis Claude Code corpus), second corpus

393 sessions with 1,697 real sub-agent groups, timestamps relative to each trace start, so the split holds out whole families (root plus children) and the overlay shifts a family together. Seed 0, 200 sessions/hour, 4 h:

| model | 10 s | 30 s | 2 min | 5 min | 15 min |
|---|---|---|---|---|---|
| B0 constant | 29550 | 36019 | 46024 | 69955 | 196398 |
| B1 Kalman | 15222 | 28636 | 49519 | 62677 | 98710 |
| B2 history, elapsed ignored | 247688 | 236921 | 202717 | 174850 | 139947 |
| M1 survival | **13552** | **16146** | **15035** | **15209** | **14915** |

M1 beats B1 by 1.1x at 10 s, 1.8x at 30 s, 3.3x at 2 min, 4.1x at 5 min and 6.6x at 15 min. Interval coverage for M1 is 70 to 88% before calibration; 90 to 99% of demand is endogenous.

Seed-and-rate sweep (3 seeds x 100, 200, 400 sessions per hour; `runs/sweep_agentx.csv`), ratio of B1 loss to M1 loss:

| sessions/hour | 10 s | 30 s | 2 min | 5 min | 15 min |
|---|---|---|---|---|---|
| 100 | 1.10 | 1.39 | 2.20 | 2.97 | 2.73 |
| 200 | 1.21 | 1.64 | 3.11 | 4.27 | 3.34 |
| 400 | 1.49 | 1.97 | 3.69 | 5.43 | 3.58 |

M1 wins every cell on this corpus too, and here it wins at 10 s as well. Caveat: the 15-minute cells have 95% CI half-widths of 50 to 60% of the mean across seeds (10% or less at the other horizons), because 393 families give only a few hundred held-out sessions and the 15-minute demand is dominated by a handful of very long sub-agent sessions; the 15-minute ratio should be read as "about 3x" rather than a precise number.

Two adapter facts matter for anyone reusing this corpus: the inter-request gap (`think_time`) is unlabeled tool-or-human time, so tool phases are `__gap__`; and sub-agent groups become child sessions with a spawn recorded on the parent's phase, which is the first real fan-out data in the project.

## Dispersion calibration

Per-horizon variance inflation fitted out-of-sample within train (predictor on half the train weeks, factor on the other half replayed as a fleet at the test's rate, ticks bounded to the fleet window, fitted per class with incremental arrivals). On TraceLab at 200/h the factors are 1.57, 1.16, 1.10, 1.10, 1.35 across horizons; held-out 90% coverage rises from 40 / 57 / 67 / 63 / 73% to 65 / 65 / 72 / 70 / 86%, still short of 90 at short horizons, and q90 pinball worsens by 5 to 20% at 10 to 30 s. Reading: the remaining miss is systematic error on sessions and tools not seen in training, not under-dispersion; widening intervals cannot fix bias. Three earlier calibration attempts returned factors of exactly 1 for three distinct bugs (in-sample fit, un-overlaid sparse calendar ticks, and fitting on an empty class), each now covered by a test; the lesson is that a calibration that reports "nothing to do" must be treated as a failure until proven otherwise.

## H1b: per-session remaining-time prediction on collected long tools

Fleet totals over eight sequential sessions cannot discriminate predictors, so H1b is scored per session: at every 15 s inside each tool phase longer than 30 s, each predictor's distribution of the remaining time is scored against the truth (`atfm.eval.resumption`), leave-one-job-family-out. Collection: `runs/collect/l1_long_events.jsonl` (two cmake builds of fmt with percent markers, 96 s each; two pipelines with linear `processed k/N` output, 330 to 335 s; two pipelines printing only `stage k/4`, 265 s). Mean q90 pinball on remaining seconds:

| job | B2 (history, no elapsed) | M1 (elapsed) | M2 (progress, learned curve) |
|---|---|---|---|
| fmt build, percent markers | 0.0 | 1.4 | 14.1 |
| pipeline, linear progress | 2.5 | 4.1 | **2.1** |
| pipeline, four stage markers | 7.0 | 8.9 | **3.3** |

Reading:
- With a linear signal M2 is near exact: median within 2 to 4 s of the truth at every offset of a 335 s tool while M1 runs 40 to 70 s low, and even four stage markers halve M1's error.
- The build is a genuine M2 failure mode: cmake's percent markers are not linear in time. M2 now learns a per-tool progress curve from training phases, but with one other build in the training fold there is no curve to learn and linear extrapolation misleads it.
- B2's zeros come from a degenerate collection: each job repeats with an identical duration, so ignoring elapsed time and predicting the other run's duration is perfect. Nothing here says B2 generalizes; the TraceLab and AgentX fleets show it is the worst predictor by far when durations vary.
- The numpy test-suite job has not yet produced a valid phase (pytest exited before running); diagnosis in progress.

### Collection 2: varied durations (`experiments/l1_jobs_varied.yaml`, `runs/collect/l1_varied_events.jsonl`)

18 sessions, 16 phases longer than 30 s (cmake builds of fmt at -j1 and -j2: 94 to 184 s; pipelines with linear or staged output at four sizes and rates: 78 to 345 s), 99% of tool time with a strong signal. Scored per session every 15 s inside each long phase, holding out one job family at a time (the same tool at other sizes and rates stays in training, which is the realistic case). Mean q90 pinball on remaining seconds, 176 observation points per model:

| job | B2 (history, no elapsed) | M1 (elapsed) | M2 (progress, learned curve) |
|---|---|---|---|
| fmt build -j1 (94 to 184 s, percent markers) | 52.6 | 34.6 | **28.8** |
| fmt build -j2 | 8.7 | 59.2 | **3.8** |
| pipeline, linear, size a | 26.3 | 26.2 | **0.5** |
| pipeline, linear, size b | 14.3 | 16.2 | **1.4** |
| pipeline, linear, size c | 21.3 | 21.4 | **0.8** |
| pipeline, linear, size d | 8.9 | 11.4 | **1.7** |
| pipeline, staged (4 markers), size a | 25.0 | 25.0 | **4.8** |
| pipeline, staged, size b | 69.0 | 39.0 | **19.8** |
| **all phases** | 34.0 | 28.1 | **10.0** |

CRPS over all phases: B2 67.0, M1 59.7, M2 17.8.

**H1b verdict: gate passed on the long-tool regime.** Live progress cuts the remaining-time error 2.8x over elapsed time alone on q90 pinball (3.4x on CRPS): 10 to 30x on tools whose output is linear in work, 2 to 5x on tools that only report stage boundaries, and 1.2x to 15x on cmake builds once a progress curve can be learned from another build of the same project. The earlier degenerate collection (identical durations per job) is superseded. Remaining limits: eight job families, one machine, no human-in-the-loop tools, and the numpy test suite still to be added as a large real suite.

## Files

- `experiments/h1_agentx.yaml`, `scripts/sweep_h1.py`, `runs/sweep_tracelab.csv`, `runs/h1_agentx_r200/`
- `experiments/h1_tracelab_cal.yaml`, `runs/h1_tracelab_r200_cal/calibration.json`
- `src/atfm/traces/agentx.py`, `src/atfm/board/calibrate.py`, `src/atfm/eval/resumption.py`, `src/atfm/board/predictors/progress.py` (`ProgressCurve`)
