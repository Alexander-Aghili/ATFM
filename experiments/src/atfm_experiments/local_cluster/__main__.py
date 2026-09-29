"""Run a bounded AIPerf AgentX smoke against a local Dynamo Mocker cluster."""
import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

from atfm.dynamo.local import LocalDynamo
from atfm_experiments.load.runtime import provenance, write_json
from .workload import select_trace
from .process import run_client


def command(client, directory, url):
    count = json.loads((directory / 'manifest.json').read_text())['workload']['requests']
    return [str(client), 'profile', '--url', url, '--model', 'Qwen/Qwen3-0.6B',
            '--endpoint-type', 'chat', '--streaming', '--input-file', str(directory / 'trace.json'),
            '--custom-dataset-type', 'weka_trace', '--tokenizer', 'Qwen/Qwen3-0.6B',
            '--no-fixed-schedule', '--ignore-trace-delays', '--concurrency', '1',
            '--synthesis-max-osl', '16', '--request-count', str(count),
            '--artifact-dir', str(directory / 'aiperf'), '--ui', 'none', '--no-auto-plot']


def run(source, directory, client, workers, port):
    if version('ai-dynamo') != '1.5.0':
        raise ValueError('this local recipe requires ai-dynamo==1.5.0')
    client_version = subprocess.check_output([str(client), '--version'], text=True).strip()
    if client_version != '0.13.0':
        raise ValueError('this local recipe requires aiperf==0.13.0')
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(kind='integration_smoke', hardware_evidence=False, lmcache_tested=False,
                    workers=workers, aiperf=client_version, dynamo=version('ai-dynamo'), environment=provenance())
    metadata['runner_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    metadata['workload'] = select_trace(source, directory)
    metadata['changes'] = ['smallest root by total tokens', 'ignore delays', 'cap output at 16', 'one root request count; 180s process timeout']
    write_json(directory / 'manifest.json', metadata)
    execute(directory, client, workers, port)


def execute(directory, client, workers, port):
    try:
        with LocalDynamo(workers=workers, port=port, log_dir=str(directory / 'dynamo')) as cluster:
            args = command(client, directory, cluster.base_url)
            write_json(directory / 'command.json', args)
            with (directory / 'aiperf.log').open('w') as log:
                run_client(args, log)
        write_json(directory / 'result.json', inspect_result(directory))
    except BaseException as exc:
        write_json(directory / 'result.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}'))
        raise


def inspect_result(directory):
    report = json.loads((directory / 'aiperf/profile_export_aiperf.json').read_text())
    branches = report.get('branch_stats') or {}
    failed = ('children_errored', 'children_truncated', 'parents_failed_due_to_child_error')
    if not report['is_complete'] or report['was_cancelled'] or report['error_summary']:
        raise ValueError('AIPerf reported an incomplete or failed replay')
    if report['request_count']['avg'] <= 0 or any(branches.get(k, 0) for k in failed):
        raise ValueError('AIPerf reported no requests or unsuccessful branches')
    return dict(status='passed', requests=report['request_count']['avg'], branches=branches,
                hardware_evidence=False, lmcache_tested=False, atfm_control_tested=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/agentx/traces.jsonl'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--client', type=Path, default=Path('tmp/venvs/aiperf/bin/aiperf'))
    parser.add_argument('--workers', type=int, choices=(2, 4, 8), default=2)
    parser.add_argument('--port', type=int, default=8790)
    args = parser.parse_args()
    run(args.source.resolve(), args.out.resolve(), args.client.resolve(), args.workers, args.port)


if __name__ == '__main__':
    main()
