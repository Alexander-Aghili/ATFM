"""Paired synthetic cache experiments; no representative-workload claims."""
from dataclasses import asdict, dataclass
import hashlib
import time

from .probe import idle_storage, inference, metrics, prefetch, request, save, validate
from .stack import CACHE, INFERENCE, MODEL


@dataclass(frozen=True)
class Case:
    name: str
    tokens: int
    output_tokens: int = 8
    distractors: int = 0


CASES = [Case(f'prompt-{size}', size) for size in (64, 352, 1024, 1920)]
CASES += [Case('decode-128', 1024, 128), Case('cache-pressure', 1920, distractors=3)]


def tokens_for(client, identity, size):
    prefix = hashlib.sha256(identity.encode()).hexdigest()[:24]
    text = prefix + ' Explain how a cache reuses stored information. ' * 1024
    tokens = request(client, 'POST', INFERENCE + '/tokenize', json={'model': MODEL, 'prompt': text})['tokens']
    if size <= 0 or size % 16 or size + 128 > 2048 or len(tokens) < size:
        raise ValueError('workload must fit the context and align with cache chunks')
    return tokens[:size]


def clear_l1(client, output, label='reset'):
    idle_storage(client, output, label + '-before', None)
    save(output, label, request(client, 'POST', CACHE + '/cache/clear', json={'tier': 'l1', 'force': True}))
    idle_storage(client, output, label + '-after', 0)


def measured_inference(client, output, name, tokens, maximum):
    before = metrics(client, output, name + '-before')
    result = inference(client, output, name, tokens, maximum)
    result['external_hits'] -= before
    return result


def pressure(client, output, identity, case):
    for index in range(case.distractors):
        tokens = tokens_for(client, f'{identity}-distractor-{index}', case.tokens)
        measured_inference(client, output, f'distractor-{index}', tokens, 8)
        idle_storage(client, output, f'after-distractor-{index}', None)


def prepare_revisit(client, output, identity, case):
    if case.distractors:
        pressure(client, output, identity, case)
    else:
        clear_l1(client, output)
    return idle_storage(client, output, 'before-revisit', None)


def revisit(client, output, tokens, case, mode):
    started = time.monotonic()
    transfer = prefetch(output, tokens) if mode == 'prefetch' else None
    lead = time.monotonic() - started if transfer else 0.
    residency = idle_storage(client, output, 'ready-for-revisit', None)
    result = measured_inference(client, output, 'revisit', tokens, case.output_tokens)
    return result, dict(prefetch=transfer, prefetch_s=lead, residency=residency,
                        lead_plus_request_s=lead + result['elapsed_s'])


def run_case(client, root, case, repeat, mode):
    identity = f'{case.name}-{repeat}-{mode}'
    output = root / identity
    output.mkdir()
    clear_l1(client, output, 'initial-reset')
    tokens = tokens_for(client, identity, case.tokens)
    save(output, 'request', dict(case=asdict(case), repeat=repeat, mode=mode, tokens=tokens))
    cold = measured_inference(client, output, 'cold', tokens, case.output_tokens)
    idle_storage(client, output, 'after-cold', None)
    before = prepare_revisit(client, output, identity, case)
    result, transfer = revisit(client, output, tokens, case, mode)
    checked = validate(cold, result, tokens)
    after = idle_storage(client, output, 'after-revisit', None)
    if after['memory_used_bytes'] > after['memory_configured_bytes']:
        raise RuntimeError('CPU cache exceeded configured capacity')
    return save(output, 'result', dict(**checked, **transfer, before=before, after=after,
                                     case=asdict(case), repeat=repeat, mode=mode))


def run_cases(client, output, repeats):
    results = []
    for case in CASES:
        for repeat in range(repeats):
            modes = ('disk', 'prefetch') if repeat % 2 == 0 else ('prefetch', 'disk')
            for mode in modes:
                results.append(run_case(client, output, case, repeat, mode))
    return results
