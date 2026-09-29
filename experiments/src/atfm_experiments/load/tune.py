"""Evaluate declared settings, freeze a candidate, then independently validate it."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from urllib.request import getproxies

from .__main__ import run_case
from .config import LoadConfig
from .runtime import provenance, write_json
from .tuning_config import TuningPlan
from .tuning_metrics import aggregate, measurements, pareto, select, violations
from .tuning_validation import validate


class TuningRun:
    def __init__(self, plan, directory, evaluator=run_case):
        self.plan, self.directory, self.evaluator = plan, directory, evaluator
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / 'plan.json', plan.model_dump())
        environment = provenance() | {'http_proxy_configured': any(k != 'no' and v for k, v in getproxies().items())}
        write_json(directory / 'environment.json', environment)
        self.rows = []

    def run(self):
        search = self.phase('search', self.plan.search_seeds, list(self.plan.candidates))
        results = aggregate(self.plan, search, self.plan.candidates, self.plan.search_seeds)
        selected = select(self.plan, results)
        selection = dict(selected=selected, aggregates=results, pareto=pareto(results))
        write_json(self.directory / 'selection.json', selection)
        if selected is None:
            decision = dict(status='no_feasible_candidate')
        else:
            names = list(dict.fromkeys([self.plan.baseline, selected]))
            validation = self.phase('validation', self.plan.validation_seeds, names)
            decision = validate(self.plan, selected, validation)
        recommendation = selected if decision['status'] == 'validated_candidate' else None
        result = dict(selection=selection, validation=decision, recommendation=recommendation, deployed=False)
        write_json(self.directory / 'result.json', result)
        return result

    def phase(self, phase, seeds, names):
        order = _schedule(self.plan, seeds, names)
        write_json(self.directory / f'{phase}-schedule.json', order)
        rows = [self.evaluate(phase, context, seed, candidate) for context, seed, candidate in order]
        return rows

    def evaluate(self, phase, context, seed, candidate):
        cfg = self.plan.workloads[context].model_dump() | self.plan.candidates[candidate].model_dump() | {'seed': seed}
        directory = self.directory / phase / context / str(seed) / candidate
        row = dict(phase=phase, context=context, seed=seed, candidate=candidate, directory=str(directory))
        try:
            summary = self.evaluator(LoadConfig.model_validate(cfg), directory)
            row['metrics'] = measurements(summary)
            row['violations'] = violations(summary, row['metrics'], self.plan.constraints)
        except Exception as exc:
            row.update(metrics=None, violations=['evaluation_error'], error=f'{type(exc).__name__}: {exc}')
        self.rows.append(row)
        write_json(self.directory / 'trials.json', self.rows)
        print(f'{phase} {context} seed={seed} candidate={candidate}: {row["violations"] or "feasible"}', flush=True)
        return row


def _schedule(plan, seeds, names):
    rng = random.Random(plan.order_seed)
    blocks = [(context, seed) for seed in seeds for context in plan.workloads]
    rng.shuffle(blocks)
    schedule = []
    for context, seed in blocks:
        candidates = list(names)
        rng.shuffle(candidates)
        schedule.extend((context, seed, candidate) for candidate in candidates)
    return schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = TuningPlan.model_validate_json(args.plan.read_text())
    result = TuningRun(plan, args.out).run()
    print(result['validation']['status'])
    if result['validation']['status'] in ('no_feasible_candidate', 'validation_failed'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
