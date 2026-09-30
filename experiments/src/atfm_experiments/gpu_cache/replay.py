"""Replay complete public tool/branch traces on real vLLM and LMCache."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from typing import NamedTuple
from pathlib import Path
import subprocess
import sys
import time

import httpx

from atfm_experiments.local_cluster.process import run_client
from . import arms
from .probe import save
from .stack import CACHE, INFERENCE, check_ports, commands, manifest, server
from .trace_selection import SELECTED, verified
from .tool_calls import run as run_tools

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'


class Limits(NamedTuple):
    case_s: int = 1200
    total_s: int = 2700


def command(client, source, output, url=INFERENCE, extra=()):
    return [str(client), 'profile', '--url', url, '--model', MODEL,
            '--endpoint-type', 'chat', '--streaming', '--input-file', str(source),
            '--custom-dataset-type', 'weka_trace', '--tokenizer', MODEL,
            '--tokenizer-revision', REVISION, '--no-fixed-schedule', '--concurrency', '1',
            '--random-seed', '7', '--extra-inputs', 'ignore_eos:true', '--num-sessions', '1',
            '--artifact-dir', str(output / 'aiperf'), '--ui', 'none', '--no-auto-plot', *extra]


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


def run_case(client, source, output, name, details, timeout, url=INFERENCE, extra=()):
    output.mkdir()
    args = command(client, source / f'{name}.jsonl', output, url, extra)
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


def run_cases(client, source, output, selection, cases=SELECTED, limits=Limits(), url=INFERENCE, extra=()):
    deadline, results = time.monotonic() + limits.total_s, {}
    for name in cases:
        remaining = int(deadline - time.monotonic())
        if remaining < 60:
            results[name] = dict(passed=False, error='total replay deadline reached')
            break
        results[name] = run_case(client, source, output / name, name, selection['cases'][name],
                                 min(limits.case_s, remaining), url=url, extra=extra)
    return results


class Setup(NamedTuple):
    arm: str = 'direct'
    l1_gb: float = 24
    chunk: int = 16
    limits: Limits = Limits()
    train: Path | None = None
    warm_gbps: float = 1.4
    trigger: str = 'q10'
    rewarm_after_s: float | None = None


def prepare(venv, client, source, output, cases, setup):
    check_ports()
    output.mkdir(parents=True, exist_ok=False)
    version = subprocess.check_output([str(client), '--version'], text=True).strip()
    if version != '0.13.0':
        raise ValueError('public replay requires AIPerf 0.13.0')
    selection = verified(source)
    cache, engine = commands(venv, output, MODEL, REVISION, 131072, setup.l1_gb, setup.chunk)
    engine += ['--enable-auto-tool-choice', '--tool-call-parser', 'hermes']
    policy = dict(trigger=setup.trigger, rewarm_after_s=setup.rewarm_after_s)
    extra = arms.processes(setup.arm, Path(sys.executable), output, setup.train, setup.l1_gb, setup.warm_gbps, MODEL, policy)
    save(output, 'manifest', dict(**manifest(venv, output, MODEL, REVISION), commands=[cache, engine, *[c for _, c, _ in extra]],
                                 selection=selection, cases=list(cases), aiperf=version, arm=setup.arm,
                                 atfm_control_enabled=setup.arm == 'atfm', l1_gb=setup.l1_gb, chunk_size=setup.chunk,
                                 limits=setup.limits._asdict(), train=train_record(setup.train), warm_gbps=setup.warm_gbps,
                                 prefetch_policy=policy))
    return selection, cache, engine, extra


def train_record(train):
    if train is None:
        return None
    return dict(path=str(train), sha256=hashlib.sha256(Path(train).read_bytes()).hexdigest())


def run(venv, client, source, output, cases=SELECTED, setup=Setup()):
    selection, cache, engine, extra = prepare(venv, client, source, output, cases, setup)
    with server(cache, output, 'lmcache', CACHE + '/status', 900):
        with server(engine, output, 'vllm', INFERENCE + '/health', 900), ExitStack() as owned:
            tools = run_tools(source / 'bfcl', output / 'bfcl', MODEL)
            for name, cmd, url in extra:
                owned.enter_context(server(cmd, output, name, url, 180))
            results = run_cases(client, source, output, selection, cases, setup.limits,
                                url=arms.client_url(setup.arm), extra=arms.client_flags(setup.arm))
    save(output, 'summary', dict(passed=tools['passed'] and all(r['passed'] for r in results.values()), cases=results,
                                arm=setup.arm, servers_stopped=True,
                                scope='selected public serving traces; no task-quality score'))


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
    parser.add_argument('--arm', choices=arms.ARMS, default='direct', help='stage E arm (default: direct vLLM)')
    parser.add_argument('--train', type=Path, help='held-out board training table (required for --arm atfm)')
    parser.add_argument('--warm-gbps', type=float, default=1.4, help='calibrated L2 -> CPU warm rate, GB/s')
    parser.add_argument('--prefetch-trigger', choices=('q10', 'q50'), default='q10', help='resumption quantile that starts a warm')
    parser.add_argument('--rewarm-after', type=float, default=None, help='seconds before a session may be warmed again in one turn')
    args = parser.parse_args(argv)
    if args.arm == 'atfm' and args.train is None:
        parser.error('--arm atfm requires --train')
    return args


def main():
    args = parse()
    setup = Setup(args.arm, args.l1_gb, args.chunk_size, Limits(args.case_timeout, args.total_timeout),
                  args.train.resolve() if args.train else None, args.warm_gbps, args.prefetch_trigger, args.rewarm_after)
    run(args.venv.resolve(), args.client.resolve(), args.source.resolve(), args.output.resolve(),
        args.cases or SELECTED, setup)

if __name__ == '__main__':
    main()
