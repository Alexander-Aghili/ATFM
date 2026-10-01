"""Retain complete public Weka roots, with source and per-root provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from atfm_experiments.local_cluster.workload import requests
from .probe import save
from .tool_calls import prepare

SELECTED = {'short-branch': 'bbdcb12440a7ab3496b9fac8b5f9824b1672',
            'sequential': '5c5e408b76e5e22747853915f67be3c491a4',
            'multi-branch': '07dd40536557a1d6440a923557c3129dc929'}
# Four single-agent roots with many multi-second idle gaps (>=8 of 19-26 gaps of 10 s or more), replayed
# concurrently so their contexts (about 29 GiB at the median) contend for a 24 GiB CPU tier: stage E2.
FLEET = ('528995cdde1ebf15ccf0b79aec42dba31ce2', '979ae6377c88be0e93531cdb9632795b6a7a',
         '240cd2e1b10e5b738ac6fee166b2aca7eff2', '3735de09df4f9ac454aee913e2a00633ad0d')
CASES = (*SELECTED, 'fleet')


def case_sha256(source, name, trace_ids=None):
    """SHA-256 of a single-root case file, or of a fleet directory's traces in the selection's order."""
    if trace_ids:
        return hashlib.sha256(b''.join((source / name / f'{i}.json').read_bytes() for i in trace_ids)).hexdigest()
    return hashlib.sha256((source / f'{name}.jsonl').read_bytes()).hexdigest()


def verified(source):
    selection = json.loads((source / 'selection.json').read_text())
    for name, case in selection['cases'].items():
        if case_sha256(source, name, case.get('trace_ids')) != case['sha256']:
            raise ValueError(f'public workload hash mismatch: {name}')
    return selection


def write_fleet(output, lines, ids):
    """AIPerf's weka_trace loader reads one JSON document per file, so a fleet is a directory of roots."""
    (output / 'fleet').mkdir()
    for i in ids:
        (output / 'fleet' / f'{i}.json').write_bytes(lines[i])
    return hashlib.sha256(b''.join(lines[i] for i in ids)).hexdigest()


def describe(trace):
    calls = list(requests(trace))
    pending, groups, signals = list(trace['requests']), 0, Counter()
    while pending:
        item = pending.pop()
        if item['type'] == 'subagent':
            groups += 1
            pending.extend(item.get('requests', []))
        signals.update(item.get('input_types', []))
    return dict(trace_id=trace['id'], requests=len(calls), subagent_groups=groups,
                max_context=max(r['in'] + r['out'] for r in calls),
                input_tokens=sum(r['in'] for r in calls), output_tokens=sum(r['out'] for r in calls),
                input_type_signals=dict(signals))


def fleet_case(lines, ids):
    """Concatenate whole roots (raw JSONL lines) in order, with the combined workload description."""
    traces = [json.loads(lines[i]) for i in ids]
    parts = [describe(trace) for trace in traces]
    details = dict(trace_ids=list(ids), sessions=len(ids), requests=sum(p['requests'] for p in parts),
                   max_context=max(p['max_context'] for p in parts),
                   input_tokens=sum(p['input_tokens'] for p in parts), output_tokens=sum(p['output_tokens'] for p in parts))
    return b''.join(lines[i] for i in ids), details


def scan(source, wanted):
    """Return the source file's SHA-256 and the raw JSONL line of every wanted root."""
    digest, lines = hashlib.sha256(), {}
    with source.open('rb') as stream:
        for line in stream:
            digest.update(line)
            identifier = json.loads(line)['id']
            if identifier in wanted:
                lines[identifier] = line
    if lines.keys() != set(wanted):
        raise ValueError('selected public roots are missing')
    return digest.hexdigest(), lines


def select(source, output):
    output.mkdir(parents=True, exist_ok=False)
    source_sha, lines = scan(source, {*SELECTED.values(), *FLEET})
    selected = {}
    for name, identifier in SELECTED.items():
        details = describe(json.loads(lines[identifier]))
        if details['max_context'] > 131072:
            raise ValueError('whole trace exceeds the configured context')
        (output / f'{name}.jsonl').write_bytes(lines[identifier])
        selected[name] = dict(**details, sha256=hashlib.sha256(lines[identifier]).hexdigest())
    _, details = fleet_case(lines, FLEET)
    selected['fleet'] = dict(**details, sha256=write_fleet(output, lines, FLEET))
    return save(output, 'selection', dict(source_sha256=source_sha, cases=selected,
                source='https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126', modified=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/agentx/traces.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    select(args.source, args.output)
    prepare(args.output / 'bfcl')


if __name__ == '__main__':
    main()
