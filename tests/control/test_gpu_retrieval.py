import pytest

from atfm_experiments.gpu_cache import retrieval


def test_plan_covers_every_length_and_condition_once_per_block():
    trials = retrieval.plan([2048, 8192], repeats=3, seed=11)
    assert len(trials) == 3 * 2 * len(retrieval.CONDITIONS)
    for block in range(3):
        cells = {(t['length'], t['condition']) for t in trials if t['repeat'] == block}
        assert cells == {(n, c) for n in (2048, 8192) for c in retrieval.CONDITIONS}
    assert len({t['trial'] for t in trials}) == len(trials)


def test_plan_order_is_seeded():
    assert retrieval.plan([2048, 8192], 2, seed=5) == retrieval.plan([2048, 8192], 2, seed=5)
    orders = {tuple((t['length'], t['condition']) for t in retrieval.plan([2048, 8192], 2, seed=s)) for s in range(6)}
    assert len(orders) > 1


def test_trial_tokens_prefix_unique_header_and_align_to_chunks():
    tokens = retrieval.trial_tokens([7, 8, 9], list(range(100, 200)), length=48, chunk=16)
    assert tokens[:3] == [7, 8, 9] and len(tokens) == 48 and tokens[3] == 100


def test_trial_tokens_reject_short_filler_and_unaligned_length():
    with pytest.raises(ValueError):
        retrieval.trial_tokens([1], [2] * 10, length=32, chunk=16)
    with pytest.raises(ValueError):
        retrieval.trial_tokens([1], [2] * 100, length=40, chunk=16)


METRICS = '''# HELP x
vllm:request_prefill_time_seconds_sum{engine="0"} 2.5
vllm:request_prefill_time_seconds_count{engine="0"} 4.0
vllm:request_queue_time_seconds_sum{engine="0"} 0.5
vllm:request_prefill_kv_computed_tokens_sum{engine="0"} 100.0
vllm:external_prefix_cache_hits_total{engine="0"} 64.0
'''


def test_counters_read_tracked_sums_and_missing_as_zero():
    values = retrieval.counters(METRICS)
    assert values['vllm:request_prefill_time_seconds_sum'] == 2.5
    assert values['vllm:external_prefix_cache_hits_total'] == 64.0
    assert values['vllm:time_to_first_token_seconds_sum'] == 0.0


def test_counter_deltas_describe_one_request():
    after = METRICS.replace('2.5', '3.25').replace('100.0', '148.0').replace('64.0', '96.0')
    delta = retrieval.delta(retrieval.counters(METRICS), retrieval.counters(after))
    assert delta['prefill_s'] == pytest.approx(.75)
    assert delta['computed_tokens'] == 48 and delta['external_hit_tokens'] == 32


def test_summary_reports_medians_and_speedup_against_cold():
    trials = [dict(length=2048, condition=c, passed=True, client_s=s, prefill_s=s / 2,
                   computed_tokens=k, external_hit_tokens=2048 - k, prefetch_s=p, l1_bytes=b)
              for c, s, k, p, b in [('cold', 1.0, 2048, None, None), ('cold', 3.0, 2048, None, None),
                                    ('l2', 1.0, 1, None, None), ('l1', .5, 1, .2, 10), ('l1', .5, 1, .4, 10)]]
    rows = {r['condition']: r for r in retrieval.summarize(trials)}
    assert rows['cold']['client_s_p50'] == 2.0 and rows['cold']['n'] == 2
    assert rows['l1']['speedup_vs_cold'] == 4.0 and rows['l1']['prefetch_s_p50'] == pytest.approx(.3)
    assert rows['l2']['speedup_vs_cold'] == 2.0


def test_summary_excludes_failed_trials():
    trials = [dict(length=2048, condition='cold', passed=False, client_s=9.0),
              dict(length=2048, condition='cold', passed=True, client_s=1.0, prefill_s=.5,
                   computed_tokens=2048, external_hit_tokens=0)]
    (row,) = retrieval.summarize(trials)
    assert row['n'] == 1 and row['failed'] == 1 and row['client_s_p50'] == 1.0


def test_stack_commands_take_an_l2_directory(tmp_path):
    from atfm_experiments.gpu_cache.stack import commands

    cache, _ = commands(tmp_path, tmp_path, l2=tmp_path / 'block-1')
    assert str(tmp_path / 'block-1') in cache[cache.index('--l2-adapter') + 1]
    assert str(tmp_path / 'l2') in commands(tmp_path, tmp_path)[0][cache.index('--l2-adapter') + 1]


def fake_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(retrieval, 'serve', lambda client, tokens: calls.append('serve') or {})
    monkeypatch.setattr(retrieval, 'idle_storage', lambda c, d, name, n, timeout_s=30: calls.append(name) or {'memory_used_bytes': 7})
    monkeypatch.setattr(retrieval, 'request', lambda *a, **k: calls.append('clear') or {'status': 'ok'})
    monkeypatch.setattr(retrieval, 'prefetch', lambda *a, **k: calls.append('prefetch') or {'ok': True})
    return calls


def test_warm_condition_seeds_clears_then_prefetches(tmp_path, monkeypatch):
    calls = fake_calls(monkeypatch)
    extra = retrieval.prepare(None, tmp_path, 'l1', [1] * 32)
    assert calls == ['serve', 'seeded', 'clear', 'cleared', 'prefetch', 'warmed']
    assert extra['l1_bytes'] == 7 and extra['prefetch_s'] >= 0


def test_l2_condition_stops_after_clear_and_cold_does_nothing(tmp_path, monkeypatch):
    calls = fake_calls(monkeypatch)
    assert retrieval.prepare(None, tmp_path, 'l2', [1] * 32) == {}
    assert calls == ['serve', 'seeded', 'clear', 'cleared']
    assert retrieval.prepare(None, tmp_path, 'cold', [1] * 32) == {} and len(calls) == 4


def test_verdict_requires_reuse_only_for_cached_conditions():
    measured = dict(output_tokens=1, requests=1, external_hit_tokens=2032)
    assert retrieval.verdict(dict(length=2048, condition='l1'), measured)['reuse_as_expected']
    assert not retrieval.verdict(dict(length=2048, condition='cold'), measured)['reuse_as_expected']
    assert not retrieval.verdict(dict(length=4096, condition='l2'), measured)['reuse_as_expected']


def test_failed_trial_is_recorded_and_does_not_raise(tmp_path, monkeypatch):
    import json

    def broken(client, text):
        raise TimeoutError('cache did not become idle')

    monkeypatch.setattr(retrieval, 'tokenize', broken)
    trial = dict(trial='r0-00-l1-2048', repeat=0, length=2048, condition='l1')
    result = retrieval.run_trial(None, tmp_path, trial, [])
    assert result['passed'] is False and 'TimeoutError' in result['error']
    assert json.loads((tmp_path / 'trials.jsonl').read_text())['trial'] == 'r0-00-l1-2048'


def test_cli_defaults():
    args = retrieval.parse(['--output', 'o'])
    assert (args.lengths, args.repeats, args.l1_gb) == ([2048, 8192, 32768, 98304], 3, 48)
