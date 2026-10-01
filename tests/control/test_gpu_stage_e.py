import json

import pytest

from atfm_experiments.gpu_cache import stage_e


def test_warm_timing_classifies_each_directive_against_the_next_request():
    requests = {'a': [10.0, 30.0], 'b': [5.0]}
    warms = [(12.0, 'a'), (29.5, 'a'), (31.0, 'a'), (6.0, 'b')]
    timing = stage_e.warm_timing(warms, requests)
    assert timing['issued'] == 4 and timing['before_next_request'] == 2 and timing['no_next_request'] == 2
    assert timing['lead_s_p50'] == pytest.approx((18.0 + .5) / 2)


def test_warm_timing_with_no_warms():
    assert stage_e.warm_timing([], {'a': [1.0]}) == dict(issued=0, before_next_request=0, no_next_request=0,
                                                         lead_s_p50=None)


def test_directives_and_requests_are_read_from_run_logs(tmp_path):
    (tmp_path / 'control.jsonl').write_text('\n'.join(json.dumps(r) for r in [
        {'t': 1.0, 'tier': [{'session_id': 'a', 'action': 'prefetch'}], 'cache_outcomes': {}},
        {'t': 3.0, 'tier': [], 'cache_outcomes': {'completed': 1}}]) + '\n')
    (tmp_path / 'events.jsonl').write_text('\n'.join(json.dumps(r) for r in [
        {'kind': 'llm.request', 't': 0.5, 'session_id': 'a'}, {'kind': 'llm.done', 't': .9, 'session_id': 'a'},
        {'kind': 'llm.request', 't': 4.0, 'session_id': 'a'}]) + '\n')
    assert stage_e.directives(tmp_path) == [(1.0, 'a')]
    assert stage_e.requests(tmp_path) == {'a': [0.5, 4.0]}
    assert stage_e.outcomes(tmp_path) == {'completed': 1}


def test_arm_names_come_from_step_names():
    assert stage_e.arm('atfm-q10-2') == 'atfm-q10' and stage_e.arm('direct-1') == 'direct'


def record(session, start_s, end_s, ttft_ms):
    return {'metadata': {'x_correlation_id': session, 'request_start_ns': int(start_s * 1e9),
                         'request_end_ns': int(end_s * 1e9), 'benchmark_phase': 'profiling'},
            'metrics': {'time_to_first_token': {'value': ttft_ms}}}


def test_post_gap_ttft_separates_returns_after_long_idle_gaps(tmp_path):
    rows = [record('a', 0, 2, 100), record('a', 20, 22, 900), record('a', 23, 24, 200),
            record('b', 0, 1, 150), record('b', 30, 31, 700)]
    path = tmp_path / 'profile_export.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    split = stage_e.post_gap_ttft(path, threshold_s=10)
    assert split['after_long_gap'] == dict(n=2, ttft_s_p50=0.8, ttft_s_mean=0.8)
    assert split['other'] == dict(n=3, ttft_s_p50=0.15, ttft_s_mean=pytest.approx(0.15))


def test_engine_phase_means_come_from_counter_deltas(tmp_path):
    before = ('vllm:request_queue_time_seconds_sum{engine="0"} 10.0\nvllm:request_queue_time_seconds_count{engine="0"} 5.0\n'
              'vllm:request_prefill_time_seconds_sum{engine="0"} 2.0\nvllm:request_prefill_time_seconds_count{engine="0"} 5.0\n'
              'vllm:request_decode_time_seconds_sum{engine="0"} 1.0\nvllm:request_decode_time_seconds_count{engine="0"} 5.0\n')
    after = before.replace('10.0', '40.0').replace('2.0', '7.0').replace('1.0', '21.0').replace('5.0', '15.0')
    (tmp_path / 'before-vllm.txt').write_text(before)
    (tmp_path / 'after-vllm.txt').write_text(after)
    phases = stage_e.engine_phases(tmp_path)
    assert phases == dict(queue_s_mean=3.0, prefill_s_mean=0.5, decode_s_mean=2.0, requests=10)


def lookup(total, l1, l2, request='chatcmpl-1'):
    return (f'\x1b[32m[2026-10-01 02:08:27,159] LMCache INFO:\x1b[0m Prefetch request completed (L1+L2): {l1 + l2}/{total} '
            f'retained keys ({l1} L1, {l2} L2) in 4.7 ms (external_request_id={request}, prefetch_request_id=1) (x.py:1)\n')


def test_tier_hits_count_request_lookups_by_tier_and_skip_atfm_warms(tmp_path):
    (tmp_path / 'lmcache.log').write_text(lookup(10, 3, 5) + lookup(10, 10, 0, 'chatcmpl-2')
                                          + lookup(8, 0, 8, request='') + 'Stored 48 tokens\n')
    hits = stage_e.tier_hits(tmp_path / 'lmcache.log')
    assert hits == dict(lookups=2, keys=20, l1_keys=13, l2_keys=5, missed_keys=2, lookups_needing_l2=1,
                        l1_share_of_hits=pytest.approx(13 / 18))


def test_tier_hits_without_lookups(tmp_path):
    (tmp_path / 'lmcache.log').write_text('')
    assert stage_e.tier_hits(tmp_path / 'lmcache.log')['l1_share_of_hits'] is None
