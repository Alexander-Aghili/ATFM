"""Own disposable local servers and retain reproducibility information."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time

import httpx

from .config import LoadConfig


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def provenance() -> dict:
    root = Path.cwd()
    paths = sorted((root / 'src/atfm').rglob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))
    def git(*args):
        result = subprocess.run(['git', *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    cpuinfo = Path('/proc/cpuinfo')
    cpu = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                if line.startswith('model name')), None) if cpuinfo.exists() else None
    return {'cpu': cpu, 'python': sys.version, 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
            'git_commit': git('rev-parse', 'HEAD'),
            'rlimit_nofile': dict(zip(('soft', 'hard'), resource.getrlimit(resource.RLIMIT_NOFILE))),
            'packages': {name: package_version(name) for name in ('atfm', 'atfm-experiments', 'numpy', 'httpx', 'httpcore',
                                                               'anyio', 'sniffio', 'uvicorn', 'yappi')},
            'source_sha256': {str(p.relative_to(root) if p.is_relative_to(root) else p):
                              hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


@contextmanager
def local_stack(cfg: LoadConfig, directory: Path):
    """Use inherited sockets to avoid free-port races; own every child and log."""
    if os.name != 'posix':
        raise RuntimeError('the local process harness requires POSIX inherited sockets')
    processes = {}
    payload = {'config': cfg.model_dump(), 'directory': str(directory.resolve())}
    env = dict(os.environ)
    env['NO_PROXY'] = env.get('NO_PROXY', '') + ',127.0.0.1,localhost'
    env['no_proxy'] = env['NO_PROXY']
    with ExitStack() as stack:
        try:
            for role in ('worker', 'board', 'proxy'):
                process, url = _launch(role, directory, payload, env, stack)
                processes[role] = process
                payload[f'{role}_url'] = url
                _wait_ready(role, directory, process, url)
            write_json(directory / 'stack.json', payload | {'pids': {k: p.pid for k, p in processes.items()}})
            yield {role: payload[f'{role}_url'] for role in processes}
        finally:
            _shutdown(directory, processes)


def _shutdown(directory, processes):
    for process in reversed(list(processes.values())):
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    write_json(directory / 'shutdown.json', {role: {'pid': p.pid, 'exit_code': p.returncode}
                                             for role, p in processes.items()})


def _wait_ready(role, directory, process, url):
    deadline = time.monotonic() + 30
    with httpx.Client(timeout=.5, trust_env=False) as client:
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'{role} exited during startup; see {directory / (role + ".log")}')
            try:
                if client.get(url + '/healthz').status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{role} startup timed out; see its log')
            time.sleep(.03)


def _launch(role, directory, payload, env, stack):
    path = directory / f'{role}-payload.json'
    write_json(path, payload)
    log = stack.enter_context((directory / f'{role}.log').open('w'))
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(2048)
        url = f'http://127.0.0.1:{listener.getsockname()[1]}'
        process = subprocess.Popen(
            [sys.executable, '-m', 'atfm_experiments.load.server', '--role', role,
             '--payload', str(path), '--fd', str(listener.fileno())],
            pass_fds=(listener.fileno(),), stdout=log, stderr=subprocess.STDOUT, env=env)
    return process, url
