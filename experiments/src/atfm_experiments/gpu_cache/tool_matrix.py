"""Broader BFCL API coverage on an isolated serving stack, without executing tools."""
import argparse
from pathlib import Path

from .probe import save
from .replay import MODEL, REVISION
from .stack import CACHE, INFERENCE, check_ports, commands, manifest, server
from .tool_calls import CATEGORIES, prepare, run as run_tools


def run(venv, output):
    check_ports()
    output.mkdir(parents=True, exist_ok=False)
    prepare(output / 'source', limit=10, categories=(*CATEGORIES, 'parallel_multiple'))
    cache, engine = commands(venv, output, MODEL, REVISION, 131072, 24)
    engine += ['--enable-auto-tool-choice', '--tool-call-parser', 'hermes']
    save(output, 'manifest', dict(**manifest(venv, output, MODEL, REVISION), commands=[cache, engine],
                                 atfm_control_enabled=False))
    with server(cache, output, 'lmcache', CACHE + '/status', 900):
        with server(engine, output, 'vllm', INFERENCE + '/health', 900):
            result = run_tools(output / 'source', output / 'bfcl', MODEL)
    save(output, 'summary', dict(**result, servers_stopped=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=Path('tmp/venvs/gpu'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.venv.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
