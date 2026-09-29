# Finding settings for a workload and hardware context

ATFM needs a repeatable way to choose settings, not a claim that one pool count,
admission window or prediction budget is universally optimal. The implemented
first step is an **offline, constrained selection and validation protocol** in
the experiments package. Runtime defaults are unchanged. It evaluates a declared
finite set and can retain the baseline, reject every candidate, or report an
inconclusive comparison. It never deploys its result.

## Define what optimal means

Let x describe the hardware, backend/software version and workload distribution,
and theta describe the policy settings. The conceptual target is

`theta*(x) = argmin_theta E[cost(theta, x)]`, subject to service/resource constraints.

The cost could be job completion latency, CPU per successful request, or another
explicit serving objective. Constraints could include latency SLOs, error rate,
resource limits, fairness and sufficient timely prediction use. The objective and
constraints must be declared before looking at results. A configuration that
saves CPU by missing useful predictions is not automatically better. Without a
chosen objective or tradeoff, retain a Pareto frontier rather than inventing a
weighted score.

A finite noisy study can establish the **best observed feasible candidate under
its evaluation protocol**. It cannot prove a global optimum over all settings,
hardware and future workloads. The current adapter measures local nonstreaming
HTTP with a fake worker; it cannot establish GPU/cache benefit, real job goodput,
streaming performance or a production SLO.

## Separate settings from evaluation conditions

| Input | Current support | Why it is separate |
| --- | --- | --- |
| Candidate settings | Admission window, upstream pool shards, prediction concurrency and caller budget | These are what the search may change. |
| Workload contexts | Named `LoadConfig` scenarios: arrivals, sessions, burst pattern, tool delays, worker slots/service and control parameters | The optimizer cannot win by quietly reducing offered work or giving itself a faster worker. |
| Hardware/software | Captured host, interpreter/dependency versions and source hashes | Repeat the study on each target environment; do not pool unlike hosts as interchangeable repetitions. |
| Objective | Worst-context mean run p95, or worst-context mean proxy CPU per successful request | Explicit choice; no automatic conversion between latency and CPU. |
| Constraints | Optional maximum run p95, maximum proxy CPU/request and minimum timely prediction use | Every run in every declared context must pass configured bounds. |
| Health | Complete successful requests and no client/session/control/probe/tool-publication errors | A failed or missing trial cannot win by producing fewer measurements. |
| Budget | Maximum trials and planned requests, including reserved validation | Invalid/over-budget designs fail before starting servers. |

For each context c and independent run block r, the adapter measures run-level
p95 `L[c,r]`, proxy CPU/request `U[c,r]`, and prediction-use fraction `Q[c,r]`.
The latency objective is `J = max_c mean_r L[c,r]`; the CPU objective substitutes
U. This is a mean of run p95 values, not a pooled request p95. The Pareto report
minimizes worst-context mean L and U while maximizing the smallest context mean
Q. A frontier point can still be a poor operational choice if the declared suite
omits an important workload. The current runner selects one configuration across
the suite, not a different setting for each request. Building a contextual map
requires validating that entire selector, not merely stitching together winners.

Proxy CPU comes from the harness's final process diagnostic and includes differing
numbers of control interactions over the trial duration. It excludes board,
worker and GPU CPU/cost. The trial includes cold HTTP setup; model fitting and
server startup precede the measured workload. The constraints are observed trial
checks, not probabilistic guarantees of future compliance. Supply actual limits;
the smoke example's generous values are not recommended production targets.

## Protocol

1. **Predeclare** contexts, candidates, baseline, objective, constraints, disjoint
   search/validation seeds, statistical settings and budget in a versioned plan.
2. **Search** every candidate with identical seeds within each context. Candidate
   order is randomized within context/seed blocks, and block order is randomized.
   Each full-stack case starts fresh processes. Cases run sequentially on the host.
   Captured source/environment fingerprints must match before and after each case;
   a changed Git HEAD alone is permitted when the captured sources are identical.
3. **Check feasibility** before ranking. Preserve all errors and raw measurements.
   Missing/duplicate blocks, unavailable/nonfinite metrics and evaluation exceptions
   make a candidate infeasible. Exact objective ties prefer the baseline.
4. **Freeze selection** in `selection.json` before any validation trial. Preserve
   the feasible Pareto frontier even when one candidate is chosen by the objective.
5. **Validate** only the selected candidate and baseline on disjoint seeds, with
   the same contexts and independently randomized blocks. Validation never selects
   a runner-up after the chosen candidate disappoints.
6. **Report a scoped decision.** Export a recommendation only if the selected
   candidate passes holdout constraints and the improvement/regression checks.
   Applying it or testing a different policy family is separate work.

The four tunable settings are an initial policy family. Fixed workload/control
parameters may also need study, but must be promoted into an explicit settings
schema before being searched. The evaluator is injectable through
`TuningRun(plan, directory, evaluator=...)`; an adapter accepts a `LoadConfig` and
output directory and returns the existing load-summary schema. Additional
metrics/objectives require explicit adapter and scoring changes. This is not yet
a generic optimizer for arbitrary RL policies or arbitrary backend APIs.

## Holdout uncertainty and regression checks

Validation builds matched candidate/baseline matrices with contexts as rows and
seeds as columns. A bootstrap resamples whole paired seed columns, using the same
resample across contexts. Individual HTTP calls within an agent run are correlated
and are not treated as independent experimental replications.

Each resample computes two relative changes:

- Overall objective change: `J_candidate / J_baseline - 1`.
- Worst-context change: `max_c(mean(candidate[c]) / mean(baseline[c]) - 1)`.

For confidence 1-alpha, each one-sided percentile bound uses level 1-alpha/2,
allocating the nominal error budget across the two checks. Recommend the candidate
only when the first upper bound is strictly below `-min_improvement` and the
second is at most `max_context_regression`. The second check prevents a gain on
a slow workload from concealing damage to a faster one. Hard feasibility checks
also apply to every validation run, including the baseline.

These are approximate paired percentile-bootstrap bounds with a Bonferroni
allocation, not exact confidence guarantees. With three validation blocks they
are particularly coarse: three is a smoke-test minimum, not evidence of adequate
statistical power. Use more independent blocks and representative traces for a
serious decision; the current plan permits up to 30 seeds per phase. More bootstrap
resamples do not create more observations. Shared-host load, temporal drift and
nonrepresentative scenarios can invalidate the experiment despite narrow bounds.
Validation seeds test repeated realizations of the declared contexts; they do not
automatically test unseen workload distributions or new hardware.

| Result | Meaning |
| --- | --- |
| `no_feasible_candidate` | Every search candidate violated a requirement or lacked complete evidence. |
| `baseline_retained` | The baseline won search and passed its holdout checks. |
| `validation_failed` | Selected candidate or baseline failed holdout feasibility. |
| `inconclusive` | Feasible comparison did not establish sufficient improvement without an excessive context regression. |
| `validated_candidate` | The selected candidate passed this plan's empirical holdout checks. |

Only the last status populates `recommendation`. `selection.selected` is merely
the search winner. `deployed` is always false. An inconclusive result is useful;
it prevents noisy rankings from becoming automatic configuration changes.

## Run and inspect

```bash
uv sync --extra dev --extra serve
uv run python -m atfm_experiments.load.tune \
  --plan experiments/tuning/smoke.json --out runs/tuning-smoke
```

The [smoke plan](../../experiments/tuning/smoke.json) uses two small workload
contexts and two candidates, three seeds per phase, at most 24 full-stack trials
and 384 requests. It exercises the workflow, not a meaningful capacity search.
For a real study, copy it and replace contexts, candidate settings, limits and
run budgets with your target deployment's requirements. `upstream_pool_shards=1`
forces stock HTTPX; null retains the runtime's automatic window-based selection.
The proxy-discovery fallback still applies. The root environment record flags
configured HTTP proxies because these can make a requested shard count ineffective.

Output directories must be new. Every case retains the usual load-harness raw
artifacts. Root outputs are:

- `plan.json`, `environment.json`: complete design and host/source provenance.
- `search-schedule.json`, `validation-schedule.json`: exact randomized execution order.
- `trials.json`: incrementally saved metrics, violations and exceptions for all attempts.
- `selection.json`: frozen search choice, context summaries and Pareto frontier.
- `result.json`: holdout bounds, decision and optional recommendation.

The CLI exits nonzero for no feasible search candidate or failed validation;
inconclusive/baseline-retained are valid completed studies. An interruption can
leave raw artifacts and partial `trials.json`, but no final decision. Automatic
resume is not implemented; never interpret a partial study as a recommendation.

## Which search algorithm next?

The first implementation enumerates a small explicit candidate list (maximum 32).
This avoids importing a search framework before the measurement contract is sound.
It is exhaustive over that list only and can be expensive. Three useful extensions
can share the same evaluator and independent validation protocol:

| Method | Suitable next use | Limitation for ATFM |
| --- | --- | --- |
| Random search | More knobs or broad discrete ranges; transparent search baseline | May spend many trials in poor regions; define priors/ranges explicitly. |
| Constrained Bayesian optimization | Expensive evaluations where prior results can guide proposals toward useful, feasible regions | Noise, discontinuities, hardware/context shifts and categorical choices need modeling. |
| Successive halving / Hyperband | Allocate more evaluation budget to promising candidates | Short serving trials may reverse rankings by hiding queue growth, cache warmup or burst behavior. Establish a valid fidelity relationship first. |

Random search is a well-established baseline for hyperparameter search;
its published neural-network results do not establish ATFM-specific efficiency.
[Bergstra and Bengio, 2012](https://jmlr.org/beta/papers/v13/bergstra12a.html).
Constrained Bayesian optimization models objective and constraint behavior to guide
expensive experiments. [Gardner et al., 2014](https://proceedings.mlr.press/v32/gardner14.html).
Hyperband allocates resources and stops less promising configurations early;
applying its budget notion to serving trials needs separate validation.
[Li et al., 2018](https://www.jmlr.org/beta/papers/v18/16-558.html).
The selection of finite enumeration for this first release is an engineering
choice, not a claim that it is the most sample-efficient algorithm.

## From offline settings to an adaptive policy

Eventually learn a mapping `pi(context) -> settings`, where context includes
observed arrival/burst behavior, service distribution, hardware, resource pressure
and backend version. First collect trustworthy offline evidence across those
contexts. Then compare a static baseline, a simple contextual lookup and a learned
selector on held-out traces and distributions. A policy that changes decisions
also changes subsequent arrivals, cache state and observations; offline replay
alone may not reproduce that feedback.

Live adaptation needs bounded exploration, a validated fallback, resource/SLO
monitoring, minimum dwell time, limited adjustment size and rollback. Shared
serving resources mean per-request A/B assignments are not automatically
independent experiments. Avoid switching settings on every noisy observation.
Retune when the workload, hardware, backend or relevant software changes; retain
those fingerprints with the result. None of this live adaptation is enabled by
the offline workflow. The next step is to define representative contexts and an
actual service objective, then improve search efficiency against this evaluator.

Before substituting hardware measurements for the fake-worker adapter, complete the [local cluster preparation and backend contract checks](local-cluster.md). Its accelerated smoke results are not policy-selection measurements.
