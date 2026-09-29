"""Retain complete public Weka roots, with source and per-root provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from atfm_experiments.local_cluster.workload import requests
from .probe import save

SELECTED = {'short-branch': 'bbdcb12440a7ab3496b9fac8b5f9824b1672',
            'sequential': '5c5e408b76e5e22747853915f67be3c491a4',
            'multi-branch': '2a2da059b7425d9dc1f999fca1177bc1cdb9'}


def verified(source):
    selection = json.loads((source / 'selection.json').read_text())
    for name in SELECTED:
        actual = hashlib.sha256((source / f'{name}.jsonl').read_bytes()).hexdigest()
        if actual != selection['cases'][name]['sha256']:
            raise ValueError(f'public workload hash mismatch: {name}')
    return selection


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


def select(source, output):
    output.mkdir(parents=True, exist_ok=False)
    digest, selected = hashlib.sha256(), {}
    names = {identifier: name for name, identifier in SELECTED.items()}
    with source.open('rb') as stream:
        for line in stream:
            digest.update(line)
            trace = json.loads(line)
            if trace['id'] in names:
                name, details = names[trace['id']], describe(trace)
                if details['max_context'] > 131072:
                    raise ValueError('whole trace exceeds the configured context')
                (output / f'{name}.jsonl').write_bytes(line)
                selected[name] = dict(**details, sha256=hashlib.sha256(line).hexdigest())
    if selected.keys() != SELECTED.keys():
        raise ValueError('selected public roots are missing')
    return save(output, 'selection', dict(source_sha256=digest.hexdigest(), cases=selected,
                source='https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126', modified=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/agentx/traces.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    select(args.source, args.output)


if __name__ == '__main__':
    main()
