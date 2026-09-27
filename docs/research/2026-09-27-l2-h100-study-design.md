# L2 study design: two H100s, placement first (2026-09-27)

Written after the simulator's second and third batches (`2026-09-24-h2sim-first-results.md`). The primary question has changed: not whether forecast-driven admission holds help (they do not, at the proxy's cap, in any loaded regime, with or without true demand), but whether forecast-driven KV placement does on a real engine.

## Question

On a real Dynamo pool, does keeping the KV of sessions forecast to return soon resident, at the expense of sessions forecast to return late, reduce TTFT after tool return for interactive sessions without a background cost, and by how much relative to rules-only admission and to a pause-and-resume working set?

## The mechanism problem, and the deployable form of placement

No shipped engine exposes "evict this session's KV" or "pin this session's KV" (spec D3). What every engine with prefix caching does expose is *recency*: the cache is an LRU over blocks, and a request that hits a prefix refreshes it. The deployable form of placement is therefore a **keep-alive touch**: for each session the board forecasts to return within a short horizon and whose KV is at risk (its blocks are older than the engine's eviction frontier), the proxy issues a minimal request sharing the session's prefix (max_tokens 1, or the engine's prefix-only endpoint where one exists) so the blocks become most-recently used. Sessions forecast to return late get no touch and age out first. The forecast is converted into LRU order, which is what the simulator's placement arm does directly.

Costs of a touch, all measured: one batch slot for one step, the prefix hit's attention cost, the request-path overhead, and, if the prefix was already evicted, a full recompute that would have happened at resumption anyway (the touch merely moves it earlier). The policy's budget is touches per second per worker; its knobs are the horizon (touch sessions returning within T) and the age threshold (touch only blocks older than A).

The simulator gets the same arm before the hardware study (`forecast_touch`: every tick, move-to-end the KV of sessions with predicted return under T, charged one prefix-hit request each), so that the simulator's placement result and the deployable mechanism's result can be compared on the same programs.

## Arms

| arm | admission | placement | what it isolates |
|---|---|---|---|
| native | engine priority only | engine LRU | baseline |
| rules | proxy window, tiers, index without next-tool term | engine LRU | the proxy's ordering alone |
| touch_M1 / touch_M2 | rules | keep-alive touches from M1 / M2 return-time forecasts | the forecast, by predictor |
| touch_oracle | rules | touches from the true next call time (scripted sessions make this known) | forecast error |
| working_set | rules | pause background programs at tool boundaries over budget, resume with hysteresis | ThunderAgent-style reactive baseline |
| touch_random | rules | touches of random sessions at the same rate as touch_M2 | the touch mechanism without the forecast |

Admission holds are not an arm; the simulator has closed that question for this workload family. The `touch_random` arm is the ablation that a reviewer will ask for: it separates "touching helps" from "touching the right sessions helps".

## Workload

Two H100s (one Dynamo frontend and KV router, two vLLM workers with prefix caching on, one model in the 7 to 8B class at bf16 so that KV pressure is reachable with tens of concurrent sessions). Background class: mini-SWE-agent sessions on the scripted Docker jobs already used for H1b (fmt builds, pipelines, the numpy suite), through the sidecar, at a rate that keeps KV at 85 to 95% occupancy. Interactive class: replayed interactive programs from TraceLab prompt sizes and think times, with tool phases executed as real sleeps of the traced duration, at the loaded regime's rate ratio. Three seeds, 60 minutes each, paired by seed exactly as in the simulator. The simulator is calibrated first by running the same programs through it with the engine's measured prefill and decode rates and comparing native-arm TTFT distributions.

## What is recorded

Exactly the simulator's columns, from the real engine: per call, queue time at proxy and at engine, TTFT, prefix hit tokens and recomputed tokens (vLLM exposes prefix cache hit metrics per request; where per-request attribution is missing, the delta of the worker's cached-token counter around the call); per session, KV residency over time from the worker metrics; per arm, touches issued, touches that hit, touches that missed (recompute moved earlier), and touch slot-seconds. Plus the serving metrics of the simulator: session-weighted SLO at 2 s, TTFT after tool p95, background completion time and deadline rate, tasks per hour, GPU-hours.

## Decision rule

The study supports the placement claim if touch_M2 or touch_M1 beats rules on session-weighted SLO with a paired interval clear of zero in two of three seeds' pooled bootstrap, with background completion time within the rules arm's interval, and if touch_random does not. It supports the forecasting claim specifically if touch_M2 beats touch_random by more than touch_oracle beats touch_M2. Any other outcome is reported as is.

## Budget

Two H100s for about 30 hours of wall time including calibration and six arms times three seeds times one hour, plus setup. Cloud spot pricing at the time of writing makes this a few hundred dollars; the 8xH100 budget in the original plan is not needed.
