"""Run with .venv/bin/python -m atfm_experiments.gpu_cache --output runs/gpu-check."""
import argparse
from pathlib import Path

import httpx

from .probe import inference, prompt, save, validate, warm
from .stack import CACHE, INFERENCE, check_ports, commands, manifest, server


def run(venv, output):
    check_ports()
    output.mkdir(parents=True, exist_ok=False)
    cache, engine = commands(venv, output)
    save(output, 'manifest', dict(**manifest(venv, output), commands=[cache, engine]))
    with server(cache, output, 'lmcache', CACHE + '/status'):
        with server(engine, output, 'vllm', INFERENCE + '/health'), httpx.Client(timeout=120) as client:
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
    args = parser.parse_args()
    run(args.venv.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
