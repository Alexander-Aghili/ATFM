"""Run paired synthetic workloads against one owned GPU serving stack."""
import argparse
from pathlib import Path

import httpx

from .probe import save
from .stack import CACHE, INFERENCE, check_ports, commands, manifest, server
from .workloads import run_cases


def run(venv, output, repeats=3, startup_timeout=900):
    check_ports()
    output.mkdir(parents=True, exist_ok=False)
    cache, engine = commands(venv, output)
    save(output, 'manifest', dict(**manifest(venv, output), commands=[cache, engine],
                                 repeats=repeats, startup_timeout=startup_timeout))
    with server(cache, output, 'lmcache', CACHE + '/status', startup_timeout):
        with server(engine, output, 'vllm', INFERENCE + '/health', startup_timeout):
            with httpx.Client(timeout=120) as client:
                results = run_cases(client, output, repeats)
    save(output, 'summary', dict(passed=True, servers_stopped=True, results=results,
                                scope='synthetic paired cache tests; not a representative policy benchmark'))
    print(output / 'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=Path('tmp/venvs/gpu'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--startup-timeout', type=int, default=900)
    args = parser.parse_args()
    if args.repeats < 1 or args.startup_timeout < 1:
        parser.error('repeats and startup timeout must be positive')
    run(args.venv.resolve(), args.output.resolve(), args.repeats, args.startup_timeout)


if __name__ == '__main__':
    main()
