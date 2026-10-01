"""Summarize stage E runs: serving outcomes per arm and, for ATFM arms, whether warms landed before the return.

python -m atfm_experiments.gpu_cache.stage_e <run dir>... > summary.json
Each run dir is one replay step (e.g. runs/round/stage-e-a/atfm-q10-2) with one replayed case inside.
"""
import json
from pathlib import Path
import re
import statistics
import sys

METRICS = ('vllm:external_prefix_cache_hits_total', 'vllm:external_prefix_cache_queries_total')


def arm(step):
    return re.sub(r'-\d+$', '', step)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def directives(run):
    return [(step['t'], d['session_id']) for step in lines(run / 'control.jsonl')
            for d in step.get('tier', []) if isinstance(d, dict) and d.get('action') == 'prefetch']


def requests(run):
    out = {}
    for e in lines(run / 'events.jsonl'):
        if e.get('kind') == 'llm.request':
            out.setdefault(e['session_id'], []).append(e['t'])
    return {sid: sorted(ts) for sid, ts in out.items()}


def outcomes(run):
    steps = lines(run / 'control.jsonl')
    return dict(steps[-1].get('cache_outcomes') or {}) if steps else {}


def warm_timing(warms, reqs):
    leads, missing = [], 0
    for t, sid in warms:
        later = [r for r in reqs.get(sid, []) if r > t]
        if later:
            leads.append(later[0] - t)
        else:
            missing += 1
    return dict(issued=len(warms), before_next_request=len(leads), no_next_request=missing,
                lead_s_p50=statistics.median(leads) if leads else None)


def counter(path, name):
    return sum(float(line.rsplit(' ', 1)[1]) for line in Path(path).read_text().splitlines() if line.startswith(name + '{'))


def serving(case):
    report = json.loads((case / 'aiperf/profile_export_aiperf.json').read_text())
    hits, queries = (counter(case / 'after-vllm.txt', m) - counter(case / 'before-vllm.txt', m) for m in METRICS)
    result = json.loads((case / 'result.json').read_text())
    return dict(passed=result['passed'], requests=result.get('requests'), elapsed_s=result['elapsed_s'],
                ttft_s={k: report['time_to_first_token'][k] / 1e3 for k in ('p50', 'p90', 'p95', 'max', 'avg')},
                latency_s={k: report['request_latency'][k] / 1e3 for k in ('p50', 'p95', 'avg')},
                hit_ratio=hits / queries if queries else None)


def summarize(run):
    run = Path(run)
    warnings = sum('Failed to batched allocate' in line for line in (run / 'lmcache.log').read_text(errors='replace').splitlines())
    case = next(path.parent for path in sorted(run.glob('*/result.json')))
    out = dict(step=run.name, arm=arm(run.name), case=case.name, l1_warnings=warnings, **serving(case))
    if (run / 'control.jsonl').exists():
        out['warms'] = dict(**warm_timing(directives(run), requests(run)), outcomes=outcomes(run))
    return out


if __name__ == '__main__':
    json.dump([summarize(path) for path in sys.argv[1:]], sys.stdout, indent=1)
    print()
