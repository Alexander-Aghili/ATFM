"""Isolated, owned serving processes; no existing server is cleared or stopped."""
from contextlib import contextmanager, ExitStack
import json
import hashlib
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import httpx

MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
PORTS = (18180, 18181, 15555)
INFERENCE, CACHE = 'http://127.0.0.1:18180', 'http://127.0.0.1:18181'


def check_ports():
    with ExitStack() as stack:
        for port in PORTS:
            sock = stack.enter_context(socket.socket())
            sock.bind(('127.0.0.1', port))


def commands(venv, output):
    cache = [str(venv / 'bin/lmcache'), 'server', '--host', '127.0.0.1', '--port', '15555',
             '--http-host', '127.0.0.1', '--http-port', '18181', '--l1-size-gb', '0.5',
             '--chunk-size', '16', '--eviction-policy', 'LRU', '--l2-adapter',
             json.dumps(dict(type='fs', base_path=str(output / 'l2')))]
    connector = dict(kv_connector='LMCacheMPConnector', kv_role='kv_both',
                     kv_connector_module_path='lmcache.integration.vllm.lmcache_mp_connector',
                     kv_connector_extra_config={'lmcache.mp.host': '127.0.0.1', 'lmcache.mp.port': 15555})
    inference = [str(venv / 'bin/vllm'), 'serve', MODEL, '--revision', REVISION,
                 '--tokenizer-revision', REVISION, '--host', '127.0.0.1', '--port', '18180',
                 '--max-model-len', '2048', '--max-num-seqs', '2', '--gpu-memory-utilization', '0.55',
                 '--enforce-eager', '--no-enable-prefix-caching', '--kv-transfer-config', json.dumps(connector)]
    return cache, inference


def stop(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    process.wait(timeout=10)


def ready(process, url, timeout=360):
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=2) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'server exited with {process.returncode}; inspect server logs')
            try:
                if client.get(url).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.5)
    raise TimeoutError(f'server readiness timed out: {url}')


@contextmanager
def server(command, output, name, url, startup_timeout=360):
    env = dict(os.environ, PYTHONHASHSEED='0', DO_NOT_TRACK='1', VLLM_NO_USAGE_STATS='1')
    env['PATH'] = str(Path(command[0]).parent) + os.pathsep + env.get('PATH', '')
    with (output / f'{name}.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            ready(process, url, timeout=startup_timeout)
            yield process
        finally:
            stop(process)


def manifest(venv, output):
    code = "import importlib.metadata as m; print('\\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in m.distributions())))"
    versions = subprocess.check_output([str(venv / 'bin/python'), '-c', code], text=True)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    (output / 'installed.txt').write_text(versions)
    environment = {key: os.environ[key] for key in ('CUDA_HOME', 'HF_HOME') if key in os.environ}
    return dict(model=MODEL, model_revision=REVISION, gpu=gpu, git_revision=revision,
                source_sha256=fingerprints(), environment=environment)


def fingerprints():
    roots = (Path(__file__).parent, Path('src/atfm/control'))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for root in roots for path in sorted(root.glob('*.py'))}
