"""Selection must not conceal missing trials, failed requests or holdout regressions."""
import json

import pytest
from pydantic import ValidationError

from atfm_experiments.load.tune import TuningRun, _schedule
from atfm_experiments.load.tuning_config import TuningPlan
from atfm_experiments.load.tuning_metrics import aggregate, measurements, pareto, select, violations
from atfm_experiments.load.tuning_validation import validate


def plan(**updates):
    values = dict(workloads={'burst': {'sessions': 4, 'turns': 2, 'pattern': 'burst'}},
                  candidates={'base': {'proxy_window': 4}, 'other': {'proxy_window': 8}}, baseline='base',
                  search_seeds=[1, 2, 3], validation_seeds=[11, 12, 13], bootstrap_samples=100)
    return TuningPlan.model_validate(values | updates)


def summary(p95=1., cpu=8., used=6):
    return dict(maximum_requests=8, requests_attempted=8, requests_ok=8, client_duration_s={'p95': p95},
                final_observations={'proxy': {'metrics': {'cpu_s': cpu, 'predictions': {'used': used, 'attempted': 8}}}},
                client_errors={}, session_errors=[], drain_deadline_reached=False, cancelled_sessions=0,
                tool_publish_errors=0, control_errors=0, control_http_errors=0, probe_errors=0)


@pytest.mark.parametrize('updates', [{'baseline': 'missing'}, {'validation_seeds': [3, 4, 5]},
                                     {'search_seeds': [1, 1, 2]}, {'search_seeds': [True, 2, 3]},
                                     {'max_trials': 1}, {'max_requests': 1}, {'objective': 'unknown'},
                                     {'workloads': {'../escape': {}}}, {'workloads': {'x': {'profile_proxy': True}}},
                                     {'candidates': {'base': {'proxy_window': 4}, 'other': {'proxy_window': 4}}}])
def test_invalid_or_leaking_plans_are_rejected(updates):
    with pytest.raises(ValidationError):
        plan(**updates)


def test_randomized_blocks_are_reproducible_complete_and_adjacent():
    p = plan(workloads={'a': {}, 'b': {}})
    schedule = _schedule(p, p.search_seeds, list(p.candidates))
    assert schedule == _schedule(p, p.search_seeds, list(p.candidates))
    assert len(schedule) == len(set(schedule)) == 12
    for start in range(0, len(schedule), 2):
        left, right = schedule[start:start+2]
        assert left[:2] == right[:2] and left[2] != right[2]


@pytest.mark.parametrize('field', ['p95', 'cpu'])
def test_nonfinite_metrics_cannot_win(field):
    with pytest.raises(ValueError):
        measurements(summary(**{field: float('nan')}))


def test_constraint_and_failure_checks_are_explicit():
    p = plan(constraints={'max_p95_s': .5, 'min_prediction_use': .9})
    s = summary()
    assert violations(s, measurements(s), p.constraints) == ['p95_s', 'prediction_use']
    s['requests_ok'] = 7
    s['probe_errors'] = 1
    assert 'incomplete_or_failed_requests' in violations(s, measurements(s), p.constraints)
    assert 'probe_errors' in violations(s, measurements(s), p.constraints)


def rows(p, phase='search', other=.5, cpu=2., coverage=.5):
    seeds = p.search_seeds if phase == 'search' else p.validation_seeds
    return [dict(candidate=candidate, context=context, seed=seed, phase=phase, violations=[],
                 metrics=dict(p95_s=1. if candidate == 'base' else other,
                              cpu_s_per_request=1. if candidate == 'base' else cpu,
                              prediction_use=1. if candidate == 'base' else coverage))
            for context in p.workloads for seed in seeds for candidate in p.candidates]


def test_pareto_preserves_tradeoffs_and_objective_is_explicit():
    p = plan()
    metrics = aggregate(p, rows(p), p.candidates, p.search_seeds)
    assert set(pareto(metrics)) == {'base', 'other'}
    assert select(p, metrics) == 'other'
    cpu = plan(objective='cpu')
    assert select(cpu, aggregate(cpu, rows(cpu), cpu.candidates, cpu.search_seeds)) == 'base'


def test_missing_duplicate_or_failed_blocks_are_not_feasible():
    p = plan()
    for records in [rows(p)[:-1], rows(p) + [rows(p)[-1]]]:
        metrics = aggregate(p, records, p.candidates, p.search_seeds)
        assert not metrics['other']['feasible'] and select(p, metrics) == 'base'
    records = rows(p)
    records[-1]['violations'] = ['failed']
    assert not aggregate(p, records, p.candidates, p.search_seeds)['other']['feasible']


@pytest.mark.parametrize('other,status', [(.5, 'validated_candidate'), (1., 'inconclusive'), (2., 'inconclusive')])
def test_heldout_paired_bounds_can_reject_a_search_winner(other, status):
    p = plan()
    result = validate(p, 'other', rows(p, 'validation', other))
    assert result['status'] == status
    assert result['validation_blocks'] == 3 and result['confidence'] == .95


def test_worst_context_guard_prevents_hiding_a_regression():
    p = plan(workloads={'slow': {}, 'fast': {}})
    records = rows(p, 'validation')
    for record in records:
        value = (10. if record['context'] == 'slow' else 1.) if record['candidate'] == 'base' else 2.
        record['metrics']['p95_s'] = value
    result = validate(p, 'other', records)
    assert result['objective_change'] < 0 and result['worst_context_change_upper'] > 0
    assert result['status'] == 'inconclusive'


def test_selection_is_frozen_before_holdout_and_regression_does_not_pick_runner_up(tmp_path):
    calls = []
    def evaluate(cfg, directory):
        heldout = cfg.seed >= 11
        if heldout:
            assert json.loads((tmp_path / 'run/selection.json').read_text())['selected'] == 'other'
        calls.append((cfg.seed, cfg.proxy_window))
        return summary(p95=(2. if heldout else .5) if cfg.proxy_window == 8 else 1.)
    result = TuningRun(plan(), tmp_path / 'run', evaluate).run()
    assert result['selection']['selected'] == 'other'
    assert result['validation']['status'] == 'inconclusive' and not result['deployed']
    assert len(calls) == 12
    with pytest.raises(FileExistsError):
        TuningRun(plan(), tmp_path / 'run', evaluate)


def test_evaluator_crash_is_retained_and_never_promoted(tmp_path):
    def fail(cfg, directory):
        raise RuntimeError('injected failure')
    result = TuningRun(plan(), tmp_path / 'failed', fail).run()
    assert result['validation']['status'] == 'no_feasible_candidate'
    trials = json.loads((tmp_path / 'failed/trials.json').read_text())
    assert len(trials) == 6 and all(r['violations'] == ['evaluation_error'] for r in trials)


async def test_load_adapter_forwards_candidate_settings_to_real_proxy(tmp_path, monkeypatch):
    from atfm_experiments.load.server import _proxy_app
    from atfm.proxy.transport import ShardedTransport
    monkeypatch.setattr('atfm.proxy.transport.getproxies', lambda: {})
    settings = dict(upstream_pool_shards=2, proxy_window=3, prediction_limit=7, prediction_budget_s=.123)
    cfg = plan().workloads['burst'].model_copy(update=settings)
    app = _proxy_app(cfg, tmp_path, {'worker_url': 'http://worker', 'board_url': None})
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.client._transport, ShardedTransport)
        assert len(app.state.client._transport.pools) == 2
        assert app.state.cfg.window == 3 and app.state.cfg.prediction_limit == 7
        assert app.state.cfg.board_timeout_s == .123


def test_equal_objective_prefers_baseline_even_when_its_name_sorts_last():
    p = plan(baseline='other')
    results = aggregate(p, rows(p, other=1.), p.candidates, p.search_seeds)
    assert select(p, results) == 'other'


def test_failed_holdout_never_exports_a_recommendation(tmp_path):
    def evaluate(cfg, directory):
        result = summary(p95=.5 if cfg.proxy_window == 8 else 1.)
        if cfg.seed >= 11 and cfg.proxy_window == 8:
            result['requests_ok'] = 7
        return result
    result = TuningRun(plan(), tmp_path / 'holdout-failure', evaluate).run()
    assert result['selection']['selected'] == 'other'
    assert result['validation']['status'] == 'validation_failed'
    assert result['recommendation'] is None
