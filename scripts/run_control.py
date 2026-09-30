"""Run board/proxy control, optionally warming LMCache MP CPU storage."""
import argparse
from contextlib import ExitStack

import httpx

from atfm.control.loop import ControlLoop
from atfm.control.lmcache import LMCacheActuator, LMCacheConfig, PromptTokens


def _actuator(a, stack):
    if not a.lmcache:
        return None
    http = stack.enter_context(httpx.Client(timeout=10))
    def prompt_source(sid):
        response = http.get(f"{a.proxy.rstrip('/')}/session/{sid}/prompt")
        response.raise_for_status()
        return response.json().get('messages')
    def tokenize(messages):
        response = http.post(a.inference.rstrip('/') + '/tokenize', json={
            'model': a.lmcache_model, 'messages': messages, 'add_generation_prompt': True})
        response.raise_for_status()
        return response.json()['tokens']
    cfg = LMCacheConfig(a.lmcache, a.lmcache_model, chunk_size=a.lmcache_chunk_size,
                        completion_timeout_s=a.lmcache_timeout, wait_for_completion=not a.lmcache_async)
    actuator = LMCacheActuator(cfg, tokens=PromptTokens(tokenize, prompt_source))
    stack.callback(actuator.close)
    return actuator


def main():
    a = _arguments()
    with ExitStack() as stack:
        lmcache = _actuator(a, stack)
        http = stack.enter_context(httpx.Client(timeout=2))
        ControlLoop(a.board, a.proxy, client=http, interval_s=a.interval,
                    log_path=a.log, lmcache=lmcache).run(steps=a.steps)


def _arguments():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--board', default='http://127.0.0.1:8081')
    ap.add_argument('--proxy', default='http://127.0.0.1:8799')
    ap.add_argument('--interval', type=float, default=5.0)
    ap.add_argument('--log', default='runs/control/control.jsonl')
    ap.add_argument('--steps', type=int)
    ap.add_argument('--lmcache', help='LMCache MP 0.5.5 HTTP URL; CPU prefetch only')
    ap.add_argument('--lmcache-model', help='Exact model name used by the inference server')
    ap.add_argument('--lmcache-chunk-size', type=int, default=256)
    ap.add_argument('--inference', help='vLLM URL providing the /tokenize endpoint')
    ap.add_argument('--lmcache-timeout', type=float, default=2.0, help='seconds to wait for a synchronous warm')
    ap.add_argument('--lmcache-async', action='store_true', help='submit warms and reconcile completion on later steps')
    a = ap.parse_args()
    if a.lmcache and not (a.lmcache_model and a.inference):
        ap.error('--lmcache requires --lmcache-model and --inference')
    return a


if __name__ == '__main__':
    main()
