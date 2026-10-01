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
# One line per LMCache lookup; vLLM's carry a chatcmpl request id, ATFM warms an empty one.
LOOKUP = re.compile(r'Prefetch request completed \(L1\+L2\): \d+/(\d+) retained keys \((\d+) L1, (\d+) L2\)'
                    r'.*?external_request_id=([^,]*),')


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


def engine_phases(case):
    """Mean per-request vLLM queue, prefill and decode time over the case, from before/after counters."""
    def delta(name):
        return counter(case / 'after-vllm.txt', name) - counter(case / 'before-vllm.txt', name)
    n = delta('vllm:request_queue_time_seconds_count')
    means = {f'{phase}_s_mean': delta(f'vllm:request_{phase}_time_seconds_sum') / n if n else None
             for phase in ('queue', 'prefill', 'decode')}
    return dict(**means, requests=int(n))


def tier_hits(path):
    """Where vLLM's lookups found their chunks: CPU tier (L1), filesystem tier (L2, a load on the critical path) or
    nowhere (recompute). ``l1_share_of_hits`` is the fast-tier share of reused chunks, the metric warming targets."""
    found = [m.groups() for m in map(LOOKUP.search, Path(path).read_text(errors='replace').splitlines()) if m]
    rows = [tuple(map(int, g[:3])) for g in found if g[3]]
    keys, l1, l2 = (sum(r[i] for r in rows) for i in range(3))
    return dict(lookups=len(rows), keys=keys, l1_keys=l1, l2_keys=l2, missed_keys=keys - l1 - l2,
                lookups_needing_l2=sum(1 for r in rows if r[2]), l1_share_of_hits=l1 / (l1 + l2) if l1 + l2 else None)


def ttft_stats(values):
    return dict(n=len(values), ttft_s_p50=statistics.median(values) if values else None,
                ttft_s_mean=statistics.mean(values) if values else None)


def post_gap_ttft(path, threshold_s=10.0):
    """TTFT of requests that follow an idle gap of at least ``threshold_s`` in their own session (where warming can
    help) versus all other requests, from AIPerf's per-request export."""
    by = {}
    for r in lines(path):
        meta = r['metadata']
        if meta.get('benchmark_phase', 'profiling') == 'profiling':
            by.setdefault(meta['x_correlation_id'], []).append(r)
    long_gap, other = [], []
    for rows in by.values():
        rows.sort(key=lambda r: r['metadata']['request_start_ns'])
        for prev, cur in zip([None] + rows, rows):
            gap = None if prev is None else (cur['metadata']['request_start_ns'] - prev['metadata']['request_end_ns']) / 1e9
            (long_gap if gap is not None and gap >= threshold_s else other).append(cur['metrics']['time_to_first_token']['value'] / 1e3)
    return dict(after_long_gap=ttft_stats(long_gap), other=ttft_stats(other))


def summarize(run):
    run = Path(run)
    warnings = sum('Failed to batched allocate' in line for line in (run / 'lmcache.log').read_text(errors='replace').splitlines())
    case = next(path.parent for path in sorted(run.glob('*/result.json')))
    out = dict(step=run.name, arm=arm(run.name), case=case.name, l1_warnings=warnings, **serving(case),
               post_gap=post_gap_ttft(case / 'aiperf/profile_export.jsonl'), engine=engine_phases(case),
               tiers=tier_hits(run / 'lmcache.log'))
    if (run / 'control.jsonl').exists():
        out['warms'] = dict(**warm_timing(directives(run), requests(run)), outcomes=outcomes(run))
    return out


if __name__ == '__main__':
    json.dump([summarize(path) for path in sys.argv[1:]], sys.stdout, indent=1)
    print()
