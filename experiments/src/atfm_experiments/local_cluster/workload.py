"""Select one complete, small public AgentX root without materializing the corpus."""
import hashlib
import json
from pathlib import Path


def requests(trace):
    pending = list(trace['requests'])
    while pending:
        request = pending.pop()
        if request['type'] == 'subagent':
            pending.extend(request.get('requests', []))
        elif request['type'] in ('s', 'n'):
            yield request


def select_trace(source: Path, directory: Path):
    digest, best, key, count = hashlib.sha256(), None, None, 0
    with source.open('rb') as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            trace = json.loads(line)
            size = sum(r['in'] + r['out'] for r in requests(trace))
            count += 1
            if size > 0 and (key is None or (size, trace['id']) < key):
                best, key = trace, (size, trace['id'])
    if best is None:
        raise ValueError('corpus has no inference requests')
    path = directory / 'trace.json'
    path.write_text(json.dumps(best) + '\n')
    return dict(source_sha256=digest.hexdigest(), roots_scanned=count, trace_id=best['id'],
                selected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), requests=sum(1 for _ in requests(best)))
