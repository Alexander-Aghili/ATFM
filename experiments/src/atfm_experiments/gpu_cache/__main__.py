"""Run with .venv/bin/python -m atfm_experiments.gpu_cache --output runs/gpu-check."""
import argparse
from pathlib import Path

import httpx

from .probe import inference, prompt, save, validate, warm
from .stack import CACHE, INFERENCE, check_ports, commands, manifest, server


def run(venv, output, startup_timeout=360):
    check_ports()
    output.mkdir(parents=True, exist_ok=False)
    cache, engine = commands(venv, output)
    save(output, 'manifest', dict(**manifest(venv, output), commands=[cache, engine], startup_timeout=startup_timeout))
    with server(cache, output, 'lmcache', CACHE + '/status', startup_timeout):
        with server(engine, output, 'vllm', INFERENCE + '/health', startup_timeout), httpx.Client(timeout=120) as client:
            tokens = prompt(client, output)
            cold = inference(client, output, 'cold', tokens)
            prefetch = warm(client, output, tokens)
            result = inference(client, output, 'warm', tokens)
            summary = dict(**validate(cold, result, tokens), prefetch=prefetch)
    save(output, 'summary', dict(**summary, servers_stopped=True))
    print(output / 'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=Path('tmp/venvs/gpu'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--startup-timeout', type=int, default=360, help='Readiness seconds per service (default: 360)')
    args = parser.parse_args()
    if args.startup_timeout <= 0:
        parser.error('--startup-timeout must be positive')
    run(args.venv.resolve(), args.output.resolve(), args.startup_timeout)


if __name__ == '__main__':
    main()
