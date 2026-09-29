"""Small BFCL-derived API checks, not an official BFCL accuracy evaluation."""
import copy
import json
import time

import httpx

from .probe import request, save
from .stack import INFERENCE


def tool_schema(function):
    function = copy.deepcopy(function)
    function['name'] = function['name'].replace('.', '_')
    pending = [function['parameters']]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if isinstance(item.get('type'), str):
                item['type'] = {'dict': 'object', 'float': 'number', 'list': 'array'}.get(item['type'], item['type'])
            pending.extend(value for value in item.values() if isinstance(value, (dict, list)))
        else:
            pending.extend(value for value in item if isinstance(value, (dict, list)))
    return dict(type='function', function=function)


def payload(item, model):
    tools = [tool_schema(function) for function in item['function']]
    names = [tool['function']['name'] for tool in tools]
    if len(names) != len(set(names)) or len(item['question']) != 1:
        raise ValueError('ambiguous tool names or unsupported multi-turn BFCL item')
    return dict(model=model, messages=item['question'][0], tools=tools, tool_choice='auto',
                max_tokens=512, temperature=0, seed=7)


def inspect(result, body):
    choice = result['choices'][0]
    calls = choice['message'].get('tool_calls') or []
    names = {tool['function']['name'] for tool in body['tools']}
    valid = all(call['function']['name'] in names and
                isinstance(json.loads(call['function']['arguments']), dict) for call in calls)
    return dict(passed=bool(calls) and valid and choice['finish_reason'] != 'length',
                tool_calls=len(calls), finish_reason=choice['finish_reason'], usage=result.get('usage'),
                semantic_accuracy_scored=False, tools_executed=False)


def one(client, item, output, model):
    body = payload(item, model)
    save(output, item['id'] + '-request', body)
    started = time.monotonic()
    try:
        result = request(client, 'POST', INFERENCE + '/v1/chat/completions', json=body)
        save(output, item['id'] + '-response', result)
        checked = inspect(result, body)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        checked = dict(passed=False, error=f'{type(exc).__name__}: {exc}')
    return dict(id=item['id'], elapsed_s=time.monotonic() - started, **checked)


def run(source, output, model):
    output.mkdir()
    results = {}
    with httpx.Client(timeout=120) as client:
        for category in ('simple_python', 'multiple', 'parallel'):
            items = json.loads((source / f'{category}.json').read_text())
            results[category] = [one(client, item, output, model) for item in items]
    return save(output, 'summary', dict(results=results,
                provenance=json.loads((source / 'manifest.json').read_text()),
                scope='BFCL-derived tool-call API sample; no tools executed or official benchmark score'))
