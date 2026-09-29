"""Prove disk-to-CPU warming and external-cache reuse with real inference."""
import json
import time

from atfm.control import TierDirective
from atfm.control.lmcache import LMCacheActuator, LMCacheConfig, PromptTokens
from .stack import CACHE, INFERENCE, MODEL


def save(output, name, value):
    (output / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    return value


def request(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


def metric(text, name):
    values = [float(line.rsplit(' ', 1)[1]) for line in text.splitlines()
              if line.startswith(name + '{') or line.startswith(name + ' ')]
    if not values:
        raise ValueError(f'missing metric: {name}')
    return sum(values)


def metrics(client, output, name):
    response = client.get(INFERENCE + '/metrics')
    response.raise_for_status()
    (output / f'{name}-metrics.txt').write_text(response.text)
    return metric(response.text, 'vllm:external_prefix_cache_hits_total')


def prompt(client, output):
    result = request(client, 'POST', INFERENCE + '/tokenize', json={
        'model': MODEL, 'prompt': 'Explain how a cache reuses stored information. ' * 40})
    tokens = result['tokens']
    tokens = tokens[:len(tokens) // 16 * 16]
    if not tokens or len(tokens) + 8 > 2048:
        raise ValueError('probe prompt outside the configured context window')
    save(output, 'request', dict(model=MODEL, prompt=tokens, max_tokens=8, temperature=0, seed=7))
    return tokens


def inference(client, output, name, tokens):
    start = time.monotonic()
    result = request(client, 'POST', INFERENCE + '/v1/completions', json={
        'model': MODEL, 'prompt': tokens, 'max_tokens': 8, 'temperature': 0, 'seed': 7})
    save(output, name, result)
    return dict(text=result['choices'][0]['text'], elapsed_s=time.monotonic() - start,
                external_hits=metrics(client, output, name))


def idle_storage(client, output, name, objects):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = request(client, 'GET', CACHE + '/status')
        storage = status['storage_manager']
        l1, store = storage['l1_manager'], storage['store_controller']
        busy = sum(l1[k] for k in ('write_locked_count', 'read_locked_count', 'temporary_count'))
        busy += store['pending_keys_count'] + store['in_flight_task_count']
        if not busy and l1['total_object_count'] == objects:
            save(output, name, status)
            return l1
        time.sleep(.1)
    raise TimeoutError(f'cache did not become idle with {objects} objects')


def warm(client, output, tokens):
    idle_storage(client, output, 'before-clear', len(tokens) // 16)
    save(output, 'clear', request(client, 'POST', CACHE + '/cache/clear', json={'tier': 'l1', 'force': True}))
    idle_storage(client, output, 'after-clear', 0)
    source = PromptTokens(lambda _: tokens, lambda _: [{'content': 'synthetic-probe'}])
    actuator = LMCacheActuator(LMCacheConfig(CACHE, MODEL, chunk_size=16, completion_timeout_s=10), tokens=source)
    directive = TierDirective(session_id='probe', action='prefetch', tier='cpu', eta_q10=0, eta_q90=1, expires_at=time.time() + 30)
    try:
        result = save(output, 'prefetch', actuator.apply_tier(directive, time.time()))
    finally:
        actuator.close()
    if result.get('ok') is not True:
        raise RuntimeError(f'prefetch did not complete: {result}')
    idle_storage(client, output, 'after-prefetch', len(tokens) // 16)
    return result


def validate(cold, warm_result, tokens):
    reused = warm_result['external_hits'] - cold['external_hits']
    if cold['external_hits'] != 0 or reused < len(tokens) - 1:
        raise RuntimeError(f'external reuse not proven: cold={cold}, warm={warm_result}')
    if cold['text'] != warm_result['text']:
        raise RuntimeError('deterministic output changed across cache reuse')
    return dict(passed=True, prompt_tokens=len(tokens), external_reused_tokens=reused,
                cold=cold, warm=warm_result, scope='single-request compatibility; not a policy benchmark')
