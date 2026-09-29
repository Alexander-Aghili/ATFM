"""Replay complete public tool/branch traces on real vLLM and LMCache."""
import argparse
import json
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


def command(client, source, output, count):
    return [str(client), 'profile', '--url', INFERENCE, '--model', MODEL,
            '--endpoint-type', 'chat', '--streaming', '--input-file', str(source),
            '--custom-dataset-type', 'weka_trace', '--tokenizer', MODEL,
            '--tokenizer-revision', REVISION, '--no-fixed-schedule', '--concurrency', '1',
            '--extra-inputs', 'ignore_eos:true', '--request-count', str(count),
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
    args = command(client, source / f'{name}.jsonl', output, details['requests'])
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


def run_cases(client, source, output, selection):
    deadline, results = time.monotonic() + 2700, {}
    for name in SELECTED:
        remaining = int(deadline - time.monotonic())
        if remaining < 60:
            results[name] = dict(passed=False, error='total replay deadline reached')
            break
        results[name] = run_case(client, source, output / name, name, selection['cases'][name], min(1200, remaining))
    return results


def run(venv, client, source, output):
    check_ports()
    output.mkdir(parents=True, exist_ok=False)
    version = subprocess.check_output([str(client), '--version'], text=True).strip()
    if version != '0.13.0':
        raise ValueError('public replay requires AIPerf 0.13.0')
    selection = verified(source)
    cache, engine = commands(venv, output, MODEL, REVISION, 131072, 24)
    engine += ['--enable-auto-tool-choice', '--tool-call-parser', 'hermes']
    save(output, 'manifest', dict(**manifest(venv, output, MODEL, REVISION), commands=[cache, engine],
                                 selection=selection, aiperf=version, atfm_control_enabled=False))
    with server(cache, output, 'lmcache', CACHE + '/status', 900):
        with server(engine, output, 'vllm', INFERENCE + '/health', 900):
            run_tools(source / 'bfcl', output / 'bfcl', MODEL)
            results = run_cases(client, source, output, selection)
    save(output, 'summary', dict(passed=all(r['passed'] for r in results.values()), cases=results,
                                servers_stopped=True, scope='selected public serving traces; no task-quality score'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=Path('tmp/venvs/gpu'))
    parser.add_argument('--client', type=Path, default=Path('tmp/venvs/aiperf/bin/aiperf'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.venv.resolve(), args.client.resolve(), args.source.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
