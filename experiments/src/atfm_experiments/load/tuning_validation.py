"""Held-out paired run-block bootstrap; never use validation to select a runner-up."""
import numpy as np

from .tuning_metrics import aggregate, objective_key


def validate(plan, selected, rows):
    names = list(dict.fromkeys([plan.baseline, selected]))
    results = aggregate(plan, rows, names, plan.validation_seeds)
    if not all(r['feasible'] for r in results.values()):
        return dict(status='validation_failed', aggregates=results)
    if selected == plan.baseline:
        return dict(status='baseline_retained', aggregates=results)
    baseline, candidate = (_matrix(plan, rows, name) for name in (plan.baseline, selected))
    intervals = _intervals(plan, baseline, candidate)
    improved = intervals['objective_change_upper'] < -plan.min_improvement
    stable = intervals['worst_context_change_upper'] <= plan.max_context_regression
    return dict(status='validated_candidate' if improved and stable else 'inconclusive',
                aggregates=results, **intervals)


def _matrix(plan, rows, candidate):
    indexed = {(r['context'], r['seed']): r['metrics'][objective_key(plan)]
               for r in rows if r['candidate'] == candidate}
    return np.asarray([[indexed[name, seed] for seed in plan.validation_seeds] for name in plan.workloads])


def _intervals(plan, baseline, candidate):
    rng = np.random.default_rng(plan.order_seed)
    samples = rng.integers(baseline.shape[1], size=(plan.bootstrap_samples, baseline.shape[1]))
    before, after = baseline[:, samples].mean(axis=2), candidate[:, samples].mean(axis=2)
    objective = after.max(axis=0) / before.max(axis=0) - 1
    context = (after / before - 1).max(axis=0)
    return dict(confidence=plan.confidence, validation_blocks=baseline.shape[1],
                objective_change=float(candidate.mean(axis=1).max() / baseline.mean(axis=1).max() - 1),
                objective_change_upper=float(np.quantile(objective, (1 + plan.confidence) / 2)),
                worst_context_change_upper=float(np.quantile(context, (1 + plan.confidence) / 2)),
                method='paired seed-block percentile bootstrap, two Bonferroni bounds; approximate, not a guarantee')
