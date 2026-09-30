"""Measure where cached-context reuse beats recomputation on real vLLM and LMCache.

Each trial serves one prompt of a given length for one output token under one
condition: ``cold`` (never seen, full prefill), ``l2`` (resident only in the
LMCache filesystem tier after an L1 clear) or ``l1`` (warmed into CPU memory by
an ATFM prefetch directive after the same clear). Trial order is shuffled per
repeat block with a recorded seed. Each block runs on a fresh process pair with
its own L2 directory, removed afterwards to bound disk use. The OS page cache is
not dropped, so ``l2`` reads may be served from host memory.
"""
import argparse
import json
from pathlib import Path
import random
import shutil
import statistics
import time

import httpx

from .probe import idle_storage, prefetch, request, save
from .replay import MODEL, REVISION
from .stack import CACHE, INFERENCE, check_ports, commands, manifest, server

CONDITIONS = ('cold', 'l2', 'l1')
CHUNK = 16
STORAGE_WAIT_S = 600  # a 98k-token context is ~13.5 GiB of KV to write or load
TRACKED = {'requests': 'vllm:request_prefill_time_seconds_count',
           'prefill_s': 'vllm:request_prefill_time_seconds_sum',
           'queue_s': 'vllm:request_queue_time_seconds_sum',
           'ttft_s': 'vllm:time_to_first_token_seconds_sum',
           'computed_tokens': 'vllm:request_prefill_kv_computed_tokens_sum',
           'external_hit_tokens': 'vllm:external_prefix_cache_hits_total',
           'external_query_tokens': 'vllm:external_prefix_cache_queries_total'}
WORDS = ('the cache model token agent server memory request queue tool result stream '
         'prefix context block layer value data time plan state order file code test').split()
MEDIANS = ('client_s', 'prefill_s', 'queue_s', 'ttft_s', 'computed_tokens', 'external_hit_tokens',
           'prefetch_s', 'l1_bytes')


def plan(lengths, repeats, seed):
    rng, trials = random.Random(seed), []
    for block in range(repeats):
        cells = [(length, condition) for length in lengths for condition in CONDITIONS]
        rng.shuffle(cells)
        trials += [dict(trial=f'r{block}-{i:02d}-{condition}-{length}', repeat=block, length=length,
                        condition=condition) for i, (length, condition) in enumerate(cells)]
    return trials


def trial_tokens(header, filler, length, chunk=CHUNK):
    if length % chunk:
        raise ValueError(f'length {length} is not a multiple of chunk size {chunk}')
    tokens = list(header) + list(filler[:length - len(header)])
    if len(tokens) != length:
        raise ValueError(f'filler provides {len(tokens)} of {length} tokens')
    return tokens


def counters(text):
    values = dict.fromkeys(TRACKED.values(), 0.0)
    for line in text.splitlines():
        name = line.split('{', 1)[0].split(' ', 1)[0]
        if name in values:
            values[name] += float(line.rsplit(' ', 1)[1])
    return values


def delta(before, after):
    return {key: after[name] - before[name] for key, name in TRACKED.items()}


def median_of(trials, key):
    values = [trial[key] for trial in trials if trial.get(key) is not None]
    return statistics.median(values) if values else None


def row(length, condition, group):
    passed = [trial for trial in group if trial.get('passed')]
    medians = {f'{key}_p50': median_of(passed, key) for key in MEDIANS}
    return dict(length=length, condition=condition, n=len(passed), failed=len(group) - len(passed), **medians)


def summarize(trials):
    groups = {}
    for trial in trials:
        groups.setdefault((trial['length'], trial['condition']), []).append(trial)
    rows = [row(length, condition, group) for (length, condition), group in sorted(groups.items())]
    cold = {r['length']: r['client_s_p50'] for r in rows if r['condition'] == 'cold'}
    for r in rows:
        base = cold.get(r['length'])
        r['speedup_vs_cold'] = base / r['client_s_p50'] if base and r['client_s_p50'] else None
    return rows


def tokenize(client, text):
    return request(client, 'POST', INFERENCE + '/tokenize', json={'model': MODEL, 'prompt': text})['tokens']


def filler_tokens(client, length, seed=7):
    rng = random.Random(seed)
    tokens = tokenize(client, ' '.join(rng.choice(WORDS) for _ in range(length + 64)))
    if len(tokens) < length:
        raise ValueError(f'filler text produced {len(tokens)} < {length} tokens')
    return tokens


def read_counters(client):
    response = client.get(INFERENCE + '/metrics')
    response.raise_for_status()
    return counters(response.text)


def settled(client, before, timeout_s=10):
    deadline = time.monotonic() + timeout_s
    while True:
        after = read_counters(client)
        if after[TRACKED['requests']] > before[TRACKED['requests']] or time.monotonic() >= deadline:
            return after
        time.sleep(.05)


def serve(client, tokens):
    before = read_counters(client)
    start = time.monotonic()
    result = request(client, 'POST', INFERENCE + '/v1/completions', json={
        'model': MODEL, 'prompt': tokens, 'max_tokens': 1, 'temperature': 0, 'seed': 7})
    elapsed = time.monotonic() - start
    after = settled(client, before)
    return dict(client_s=elapsed, output_tokens=result['usage']['completion_tokens'], **delta(before, after))


def prepare(client, directory, condition, tokens):
    if condition == 'cold':
        return {}
    serve(client, tokens)
    idle_storage(client, directory, 'seeded', None, STORAGE_WAIT_S)
    save(directory, 'clear', request(client, 'POST', CACHE + '/cache/clear', json={'tier': 'l1', 'force': True}))
    idle_storage(client, directory, 'cleared', 0, STORAGE_WAIT_S)
    if condition == 'l2':
        return {}
    start = time.monotonic()
    prefetch(directory, tokens, MODEL, timeout_s=300)
    elapsed = time.monotonic() - start
    l1 = idle_storage(client, directory, 'warmed', len(tokens) // CHUNK, STORAGE_WAIT_S)
    return dict(prefetch_s=elapsed, l1_bytes=l1['memory_used_bytes'])


def verdict(trial, measured):
    reused = measured['external_hit_tokens'] >= trial['length'] - CHUNK
    expected = (measured['external_hit_tokens'] == 0) if trial['condition'] == 'cold' else reused
    return dict(passed=measured['output_tokens'] == 1 and measured['requests'] >= 1, reuse_as_expected=expected)


def run_trial(client, output, trial, filler):
    directory = output / trial['trial']
    directory.mkdir()
    try:
        tokens = trial_tokens(tokenize(client, f"Trial {trial['trial']}."), filler, trial['length'])
        extra = prepare(client, directory, trial['condition'], tokens)
        measured = serve(client, tokens)
        result = dict(**trial, **measured, **extra, **verdict(trial, measured))
    except (httpx.HTTPError, RuntimeError, TimeoutError, ValueError, KeyError) as exc:
        result = dict(**trial, passed=False, error=f'{type(exc).__name__}: {exc}')
    with (output / 'trials.jsonl').open('a') as log:
        log.write(json.dumps(result) + '\n')
    return result


def run_block(venv, output, block, trials, l1_gb):
    check_ports()
    l2 = output / f'l2-block{block}'
    cache, engine = commands(venv, output, MODEL, REVISION, 131072, l1_gb, CHUNK, l2=l2)
    try:
        with server(cache, output, f'lmcache-block{block}', CACHE + '/status', 900), \
                server(engine, output, f'vllm-block{block}', INFERENCE + '/health', 900), \
                httpx.Client(timeout=900) as client:
            filler = filler_tokens(client, max(t['length'] for t in trials))
            return [run_trial(client, output, trial, filler) for trial in trials]
    finally:
        shutil.rmtree(l2, ignore_errors=True)


def run(venv, output, lengths, repeats, seed, l1_gb):
    output.mkdir(parents=True, exist_ok=False)
    trials = plan(lengths, repeats, seed)
    cache, engine = commands(venv, output, MODEL, REVISION, 131072, l1_gb, CHUNK)
    save(output, 'manifest', dict(**manifest(venv, output, MODEL, REVISION), commands=[cache, engine], plan=trials,
                                 seed=seed, l1_gb=l1_gb, chunk_size=CHUNK, atfm_control_enabled=False))
    results = []
    for block in range(repeats):
        results += run_block(venv, output, block, [t for t in trials if t['repeat'] == block], l1_gb)
    save(output, 'summary', dict(passed=all(r['passed'] for r in results), rows=summarize(results),
                                scope='single-request retrieval paths; page cache not dropped; no policy claim'))


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--venv', type=Path, default=Path('tmp/venvs/gpu'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[2048, 8192, 32768, 98304])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--l1-gb', type=float, default=48, help='LMCache CPU (L1) tier size')
    return parser.parse_args(argv)


def main():
    args = parse()
    run(args.venv.resolve(), args.output.resolve(), args.lengths, args.repeats, args.seed, args.l1_gb)


if __name__ == '__main__':
    main()
