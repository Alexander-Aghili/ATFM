"""Public workloads retain their shape; benchmark labels remain explicit."""
from atfm_experiments.gpu_cache import replay, tool_calls
from atfm_experiments.gpu_cache.trace_selection import describe


def test_public_replay_preserves_sizes_and_schedule(tmp_path):
    command = replay.command(tmp_path / 'aiperf', tmp_path / 'trace', tmp_path)
    assert '--no-fixed-schedule' in command and '--tokenizer-revision' in command
    assert '--ignore-trace-delays' not in command
    assert '--synthesis-max-osl' not in command and '--synthesis-max-isl' not in command
    assert command[command.index('--num-sessions') + 1] == '1'
    assert '--request-count' not in command
    assert command[command.index('--random-seed') + 1] == '7'


def test_trace_statistics_include_nested_requests():
    trace = {'id': 'root', 'requests': [{'type': 's', 'in': 100, 'out': 4},
             {'type': 'subagent', 'requests': [{'type': 'n', 'in': 1000, 'out': 50}]}]}
    stats = describe(trace)
    assert stats['requests'] == 2 and stats['max_context'] == 1050
    assert stats['subagent_groups'] == 1


def test_bfcl_schema_conversion_does_not_mutate_source():
    function = {'name': 'math.add', 'parameters': {'type': 'dict', 'properties': {'x': {'type': 'float'}}}}
    converted = tool_calls.tool_schema(function)['function']
    assert converted['name'] == 'math_add' and converted['parameters']['type'] == 'object'
    assert converted['parameters']['properties']['x']['type'] == 'number'
    assert function['name'] == 'math.add' and function['parameters']['type'] == 'dict'


def test_bfcl_check_rejects_unknown_tool_and_truncated_output():
    body = {'tools': [{'function': {'name': 'allowed'}}]}
    result = {'choices': [{'finish_reason': 'tool_calls', 'message': {'tool_calls': [
              {'function': {'name': 'unknown', 'arguments': '{}'}}]}}]}
    assert not tool_calls.inspect(result, body)['passed']
    result['choices'][0]['message']['tool_calls'][0]['function']['name'] = 'allowed'
    assert tool_calls.inspect(result, body)['passed']
    assert not tool_calls.inspect(result, body, minimum_calls=2)['passed']
    result['choices'][0]['finish_reason'] = 'length'
    assert not tool_calls.inspect(result, body)['passed']


def test_tool_summary_propagates_case_failure(tmp_path, monkeypatch):
    import hashlib
    import json

    source = tmp_path / 'source'
    source.mkdir()
    data = b'[{"id": "sample"}]'
    (source / 'simple_python.json').write_bytes(data)
    (source / 'manifest.json').write_text(json.dumps({
        'simple_python': {'selected_sha256': hashlib.sha256(data).hexdigest()}}))
    monkeypatch.setattr(tool_calls, 'CATEGORIES', ('simple_python',))
    monkeypatch.setattr(tool_calls, 'one', lambda *args: {'passed': False})
    summary = tool_calls.run(source, tmp_path / 'result', 'test-model')
    assert summary['passed'] is False


def test_public_case_validates_export_and_records_verdict(tmp_path, monkeypatch):
    import json

    output = tmp_path / 'case'

    def client(*args, **kwargs):
        (output / 'aiperf').mkdir()
        report = dict(is_complete=True, was_cancelled=False, error_summary=[],
                      request_count={'avg': 21}, branch_stats={'children_completed': 1})
        (output / 'aiperf/profile_export_aiperf.json').write_text(json.dumps(report))

    monkeypatch.setattr(replay, 'run_client', client)
    monkeypatch.setattr(replay, 'snapshot', lambda *args: None)
    result = replay.run_case(tmp_path / 'client', tmp_path, output, 'trace', {'requests': 21}, 60)
    assert result['passed'] and result['requests'] == result['expected_requests'] == 21
    assert json.loads((output / 'result.json').read_text()) == result


def test_public_retry_runs_only_selected_complete_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, 'run_case', lambda *args: {'name': args[3]})
    selection = {'cases': {'short-branch': {}, 'sequential': {}, 'multi-branch': {}}}
    result = replay.run_cases(tmp_path, tmp_path, tmp_path, selection, cases=('multi-branch',))
    assert result == {'multi-branch': {'name': 'multi-branch'}}


def test_stack_commands_take_cache_capacity_and_chunk_size(tmp_path):
    from atfm_experiments.gpu_cache.stack import commands

    cache, _ = commands(tmp_path, tmp_path, context=131072, cpu_gb=48, chunk=256)
    assert cache[cache.index('--l1-size-gb') + 1] == '48'
    assert cache[cache.index('--chunk-size') + 1] == '256'
    assert commands(tmp_path, tmp_path)[0][cache.index('--chunk-size') + 1] == '16'


def test_public_replay_bounds_each_case_by_configured_timeouts(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(replay, 'run_case', lambda *args: seen.append(args[-1]) or {})
    selection = {'cases': {'multi-branch': {}}}
    replay.run_cases(tmp_path, tmp_path, tmp_path, selection, ('multi-branch',), replay.Limits(2400, 3000))
    assert 2390 <= seen[0] <= 2400
    replay.run_cases(tmp_path, tmp_path, tmp_path, selection, ('multi-branch',), replay.Limits(2400, 600))
    assert 590 <= seen[1] <= 600


def test_public_replay_cli_defaults_match_first_h100_run():
    args = replay.parse(['--source', 's', '--output', 'o'])
    assert (args.l1_gb, args.chunk_size, args.case_timeout, args.total_timeout) == (24, 16, 1200, 2700)
    args = replay.parse(['--source', 's', '--output', 'o', '--l1-gb', '48', '--chunk-size', '256',
                         '--case-timeout', '2400', '--total-timeout', '3000'])
    assert (args.l1_gb, args.chunk_size, args.case_timeout, args.total_timeout) == (48, 256, 2400, 3000)
