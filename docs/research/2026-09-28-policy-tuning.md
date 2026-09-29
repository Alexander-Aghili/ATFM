# A repeatable method for context-specific policy selection

Date: 28 September 2026. Implementation: `4b81425`; final source/environment
checks and paired-noise tests: `d283acd`. Runtime defaults are unchanged.

## Question and decision

The useful target is a way to discover settings for a declared hardware and
workload context. There is no justified universal admission window, pool count
or prediction budget. The [method guide](../development/policy-tuning.md)
defines the objective, constraints, search space, uncertainty and limitations.
It compares finite enumeration, random search, constrained Bayesian optimization
and successive halving using primary sources.

The first implementation lives entirely in the experiments package. It enumerates
an explicit bounded candidate list, checks feasibility, records a Pareto frontier,
freezes one selection, and validates it against a baseline on disjoint seeds.
It supports a worst-context latency or CPU objective, health checks, per-run
latency/CPU/prediction-use bounds, paired run-block bootstrap checks, and explicit
inconclusive, baseline-retained and failure outcomes. It never deploys a setting.
Source/environment fingerprints are checked before and after each case; these
checks cover captured metadata, not every possible source of host interference.

The finite list is a transparent first proposal mechanism, not a claim of search
optimality. Future proposal algorithms can reuse the evaluator and independent
validation boundary. Learning a context-dependent policy requires evaluating the
whole selector on held-out contexts/traces, not combining individually favorable
configuration results after inspecting their validation data.

## Verification

- Full suite at `d283acd`: **555 passed, 3 skipped**, six existing warnings.
- The 20-physical-line function limit remains enforced.
- Tests cover seed leakage, budgets, invalid plans and metrics, health failures,
  complete/duplicate blocks, ties, Pareto filtering, objective selection,
  frozen selection, no holdout runner-up selection, context regressions,
  paired shared noise, environment/source changes and the real proxy adapter.
- The final real-HTTP smoke completed **24 trials and 384 successful requests**
  across two contexts, two candidates and three seeds per phase, with no recorded
  request, session, control, probe or tool-publication errors.

The smoke returned `validated_candidate` for `partitioned`, with `deployed=false`.
This is a functional workflow result only. The settings change both admission
window and pool count, so it does not isolate either mechanism. The tiny fake
worker, three holdout blocks and local nonstreaming traffic cannot establish a
production setting, GPU benefit, statistical power or capacity. Documentation
builds overlapped part of this final smoke run; its timing results must not be
used as a controlled performance comparison. No default was changed from it.

The [raw evidence](results/policy-tuning-2026-09-28/README.md) preserves the plan,
schedules, frozen selection, final result, raw case artifacts, source fingerprints,
execution log and test output. A manifest hashes all 489 evidence files using
uncompressed bytes, preserving errors and measurements for independent inspection.

## Next useful work

1. Declare a deployment objective, meaningful constraints and representative
   workload contexts; add real backend/job metrics where needed.
2. Establish measurement repeatability on the target host with independent
   blocks and an untouched validation set.
3. Expand candidate proposals only once trial costs justify it; constrained
   Bayesian optimization is a candidate, not an implemented dependency.
4. Compare a fixed setting with a simple contextual selector before adding live
   adaptation, bounded exploration and rollback.
