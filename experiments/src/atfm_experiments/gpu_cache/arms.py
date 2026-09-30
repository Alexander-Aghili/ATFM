"""Stage E arms: which ATFM processes sit in front of vLLM and where AIPerf sends requests.

``direct`` is the ordinary vLLM/LMCache baseline. ``proxy`` adds only the ATFM proxy, isolating the hop's
cost. ``atfm`` adds the board (forecasts from a held-out training table, proxy-only gap phases) and the
control loop, which turns forecast-driven prefetch directives into asynchronous LMCache L2 -> CPU warms.
"""
from pathlib import Path

import yaml

from .stack import CACHE, INFERENCE

ARMS = ('direct', 'proxy', 'atfm')
PROXY_PORT, BOARD_PORT = 18190, 18191
PROXY, BOARD = f'http://127.0.0.1:{PROXY_PORT}', f'http://127.0.0.1:{BOARD_PORT}'
ROOT = Path(__file__).resolve().parents[4]
BYTES_PER_TOKEN = 147456   # Qwen3-4B KV layout measured in stage C
INTERVAL_S = 2.0


def control_config(l1_gb, warm_gbps, interval_s=INTERVAL_S, trigger='q10', rewarm_after_s=None):
    return dict(prefetch=dict(bytes_per_token=BYTES_PER_TOKEN, warm_bytes_per_s=warm_gbps * 1e9, overhead_s=.2,
                              interval_s=interval_s, margin_s=1.0, budget_bytes=l1_gb * 2**30 / 2, max_per_plan=8,
                              trigger=trigger, rewarm_after_s=rewarm_after_s))


def client_url(arm):
    return INFERENCE if arm == 'direct' else PROXY


def client_flags(arm):
    if arm == 'direct':
        return []
    return ['--session-header', 'x-atfm-session', '--server-metrics', INFERENCE + '/metrics']


def _proxy(python, output, board):
    command = [str(python), str(ROOT / 'scripts/run_proxy.py'), '--upstream', INFERENCE, '--port', str(PROXY_PORT),
               '--window', '64', '--events', str(output / 'events.jsonl'), '--trace', str(output / 'calls.jsonl')]
    return ('proxy', command + (['--board', BOARD] if board else []), PROXY + '/healthz')


def _board(python, output, train, config):
    return ('board', [str(python), str(ROOT / 'scripts/run_board.py'), '--events', str(output / 'events.jsonl'),
                      '--snapshots', str(output / 'snapshots.jsonl'), '--train', str(train), '--tick', str(INTERVAL_S),
                      '--serve', str(BOARD_PORT), '--control', str(config), '--gap-after-done'], BOARD + '/healthz')


def _control(python, output, model):
    return ('control', [str(python), str(ROOT / 'scripts/run_control.py'), '--board', BOARD, '--proxy', PROXY,
                        '--interval', str(INTERVAL_S), '--log', str(output / 'control.jsonl'), '--lmcache', CACHE,
                        '--lmcache-model', model, '--lmcache-chunk-size', '16', '--inference', INFERENCE,
                        '--lmcache-async'], None)


def processes(arm, python, output, train, l1_gb, warm_gbps, model='Qwen/Qwen3-4B-Instruct-2507', policy=None):
    """(name, command, readiness URL or None) in start order."""
    if arm not in ARMS:
        raise ValueError(f'unknown arm {arm!r}; expected one of {ARMS}')
    if arm == 'direct':
        return []
    if arm == 'proxy':
        return [_proxy(python, output, board=False)]
    config = output / 'control.yaml'
    config.write_text(yaml.safe_dump(control_config(l1_gb, warm_gbps, **(policy or {}))))
    return [_board(python, output, train, config), _proxy(python, output, board=True), _control(python, output, model)]
