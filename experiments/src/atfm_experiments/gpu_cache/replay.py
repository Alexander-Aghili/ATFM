"""Replay complete public tool/branch traces on real vLLM and LMCache."""
import argparse
import json
from typing import NamedTuple
from pathlib import Path
import subprocess
import time

import httpx

from atfm_experiments.local_cluster.process import run_client
from .probe import save
from .stack import CACHE, INFERENCE, check_ports, commands, manifest, server
from .trace_selection import SELECTED, verified
from .tool_calls import run as run_tools

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'


class Limits(NamedTuple):
    case_s: int = 1200
    total_s: int = 2700


def command(client, source, output):
    return [str(client), 'profile', '--url', INFERENCE, '--model', MODEL,
            '--endpoint-type', 'chat', '--streaming', '--input-file', str(source),
            '--custom-dataset-type', 'weka_trace', '--tokenizer', MODEL,
            '--tokenizer-revision', REVISION, '--no-fixed-schedule', '--concurrency', '1',
            '--random-seed', '7', '--extra-inputs', 'ignore_eos:true', '--num-sessions', '1',
            '--artifact-dir', str(output / 'aiperf'), '--ui', 'none', '--no-auto-plot']


def snapshot(output, label):
    with httpx.Client(timeout=30) as client:
        for name, url in (('vllm', INFERENCE + '/metrics'), ('lmcache', CACHE + '/status')):
            response = client.get(url)
            response.raise_for_status()
            (output / f'{label}-{name}.txt').write_text(response.text)


def inspect(output, expected):
    report = json.loads((output / 'aiperf/profile_export_aiperf.json').read_text())
    branches = report.get('branch_stats') or {}
    failed = ('children_errored', 'children_truncated', 'parents_failed_due_to_child_error')
    count = report['request_count']['avg']
    passed = (report['is_complete'] and not report['was_cancelled'] and not report['error_summary']
              and count == expected and not any(branches.get(key, 0) for key in failed))
    return dict(passed=passed, requests=count, expected_requests=expected, branches=branches,
                error_summary=report['error_summary'], is_complete=report['is_complete'])


def run_case(client, source, output, name, details, timeout):
    output.mkdir()
    args = command(client, source / f'{name}.jsonl', output)
    save(output, 'command', args)
    snapshot(output, 'before')
    started = time.monotonic()
    try:
        with (output / 'aiperf.log').open('w') as log:
            run_client(args, log, timeout=timeout)
        result = inspect(output, details['requests'])
    except (subprocess.SubprocessError, ValueError, KeyError, OSError) as exc:
        result = dict(passed=False, error=f'{type(exc).__name__}: {exc}')
    snapshot(output, 'after')
    return save(output, 'result', dict(**result, elapsed_s=time.monotonic() - started, workload=details))


def run_cases(client, source, output, selection, cases=SELECTED, limits=Limits()):
    deadline, results = time.monotonic() + limits.total_s, {}
    for name in cases:
        remaining = int(deadline - time.monotonic())
        if remaining < 60:
            results[name] = dict(passed=False, error='total replay deadline reached')
            break
        results[name] = run_case(client, source, output / name, name, selection['cases'][name],
                                 min(limits.case_s, remaining))
    return results


def run(venv, client, source, output, cases=SELECTED, l1_gb=24, chunk=16, limits=Limits()):
    check_ports()
    output.mkdir(parents=True, exist_ok=False)
    version = subprocess.check_output([str(client), '--version'], text=True).strip()
    if version != '0.13.0':
        raise ValueError('public replay requires AIPerf 0.13.0')
    selection = verified(source)
    cache, engine = commands(venv, output, MODEL, REVISION, 131072, l1_gb, chunk)
    engine += ['--enable-auto-tool-choice', '--tool-call-parser', 'hermes']
    save(output, 'manifest', dict(**manifest(venv, output, MODEL, REVISION), commands=[cache, engine],
                                 selection=selection, cases=list(cases), aiperf=version, atfm_control_enabled=False,
                                 l1_gb=l1_gb, chunk_size=chunk, limits=limits._asdict()))
    with server(cache, output, 'lmcache', CACHE + '/status', 900):
        with server(engine, output, 'vllm', INFERENCE + '/health', 900):
            tools = run_tools(source / 'bfcl', output / 'bfcl', MODEL)
            results = run_cases(client, source, output, selection, cases, limits)
    save(output, 'summary', dict(passed=tools['passed'] and all(r['passed'] for r in results.values()), cases=results,
                                servers_stopped=True, scope='selected public serving traces; no task-quality score'))


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=Path('tmp/venvs/gpu'))
    parser.add_argument('--client', type=Path, default=Path('tmp/venvs/aiperf/bin/aiperf'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=SELECTED, action='append', dest='cases')
    parser.add_argument('--l1-gb', type=float, default=24, help='LMCache CPU (L1) tier size')
    parser.add_argument('--chunk-size', type=int, default=16, help='LMCache chunk size in tokens')
    parser.add_argument('--case-timeout', type=int, default=1200, help='seconds per complete root')
    parser.add_argument('--total-timeout', type=int, default=2700, help='seconds for all roots')
    return parser.parse_args(argv)


def main():
    args = parse()
    run(args.venv.resolve(), args.client.resolve(), args.source.resolve(), args.output.resolve(),
        args.cases or SELECTED, args.l1_gb, args.chunk_size, Limits(args.case_timeout, args.total_timeout))

if __name__ == '__main__':
    main()
