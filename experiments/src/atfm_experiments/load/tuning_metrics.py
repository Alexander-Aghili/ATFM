"""Fail-closed feasibility checks and explicit multiobjective search summaries."""
import math
from statistics import mean


METRICS = ('p95_s', 'cpu_s_per_request', 'prediction_use')


def measurements(summary):
    proxy = summary['final_observations']['proxy']['metrics']
    predictions = proxy['predictions']
    result = dict(p95_s=summary['client_duration_s']['p95'],
                  cpu_s_per_request=proxy['cpu_s'] / max(summary['requests_ok'], 1),
                  prediction_use=predictions['used'] / max(predictions['attempted'], 1))
    for key, value in result.items():
        if value is None or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid metric {key}')
    if not predictions['attempted'] or not 0 <= predictions['used'] <= predictions['attempted']:
        raise ValueError('prediction coverage is unavailable or invalid')
    if result['p95_s'] <= 0 or result['cpu_s_per_request'] <= 0:
        raise ValueError('latency and CPU measurements must be positive')
    return result


def violations(summary, metrics, constraints):
    reasons = []
    if summary['requests_ok'] != summary['maximum_requests'] or summary['requests_attempted'] != summary['maximum_requests']:
        reasons.append('incomplete_or_failed_requests')
    errors = ('client_errors', 'session_errors', 'drain_deadline_reached', 'cancelled_sessions', 'tool_publish_errors',
              'control_errors', 'control_http_errors', 'probe_errors')
    reasons.extend(key for key in errors if summary[key])
    checks = [('p95_s', constraints.max_p95_s, lambda a, b: a > b),
              ('cpu_s_per_request', constraints.max_cpu_s_per_request, lambda a, b: a > b),
              ('prediction_use', constraints.min_prediction_use, lambda a, b: a < b)]
    reasons.extend(key for key, bound, fails in checks if bound is not None and fails(metrics[key], bound))
    return reasons


def aggregate(plan, rows, candidates, seeds):
    results = {}
    for candidate in candidates:
        selected = [r for r in rows if r['candidate'] == candidate]
        contexts = {name: _context([r for r in selected if r['context'] == name], seeds) for name in plan.workloads}
        feasible = all(c['feasible'] for c in contexts.values())
        score = max(c[objective_key(plan)] for c in contexts.values()) if feasible else None
        results[candidate] = dict(feasible=feasible, score=score, contexts=contexts)
    return results


def _context(rows, seeds):
    complete = len(rows) == len(seeds) and {r['seed'] for r in rows} == set(seeds)
    feasible = complete and all(not r['violations'] for r in rows)
    values = {key: mean(r['metrics'][key] for r in rows) if feasible else None for key in METRICS}
    return dict(feasible=feasible, runs=len(rows), **values)


def objective_key(plan):
    return 'p95_s' if plan.objective == 'latency' else 'cpu_s_per_request'


def select(plan, results):
    eligible = [name for name, result in results.items() if result['feasible']]
    return min(eligible, key=lambda name: (results[name]['score'], name != plan.baseline, name)) if eligible else None


def pareto(results):
    vectors = {name: _vector(result) for name, result in results.items() if result['feasible']}
    return [name for name, vector in vectors.items() if not any(_dominates(other, vector)
            for other_name, other in vectors.items() if other_name != name)]


def _vector(result):
    contexts = result['contexts'].values()
    return (max(c['p95_s'] for c in contexts), max(c['cpu_s_per_request'] for c in contexts),
            -min(c['prediction_use'] for c in contexts))


def _dominates(left, right):
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))
