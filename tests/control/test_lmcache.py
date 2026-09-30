"""LMCache MP contract: acceptance, completion and unsupported placement differ."""
import httpx
import pytest

from atfm.control import TierDirective, TouchDirective
from atfm.control.lmcache import LMCacheActuator, LMCacheConfig, PromptTokens


def actuator(polls=None, **config):
    calls, pending = [], list(polls or [dict(status='completed', found_keys=2, total_keys=2)])
    def handler(request):
        calls.append(request)
        path = request.url.path
        if path == '/version': return httpx.Response(200, json='0.5.5-g05a013b2')
        if path == '/status': return httpx.Response(200, json=dict(chunk_size=2, is_healthy=True))
        if request.method == 'POST': return httpx.Response(202, json=dict(status='submitted', chunks=2, request_id='abc'))
        body = pending.pop(0) if len(pending) > 1 else pending[0]
        return httpx.Response(200, json=dict(request_id='abc', **body))
    tokens = PromptTokens(lambda messages: [1, 2, 3, 4], lambda sid: [{'content': sid}])
    cfg = LMCacheConfig('http://lmcache', 'model', chunk_size=2, **config)
    return LMCacheActuator(cfg, httpx.Client(transport=httpx.MockTransport(handler)), tokens), calls


def test_waits_for_complete_transfer_with_exact_wire_body():
    import json
    act, calls = actuator([dict(status='pending'), dict(status='completed', found_keys=2, total_keys=2)])
    result = act.prefetch('s')
    assert result['ok'] and result['target'] == 'cpu_l1' and not act.pending
    post = next(c for c in calls if c.method == 'POST')
    assert json.loads(post.content) == dict(model_name='model', world_size=1, token_ids=[1, 2, 3, 4],
                                          cache_salt='', source_tier='l2', target_tier='l1')
    assert len([c for c in calls if c.url.path.endswith('/abc')]) == 2


@pytest.mark.parametrize('body', [dict(status='completed', found_keys=1, total_keys=2),
                                dict(status='completed', found_keys=2, total_keys=3),
                                dict(status='failed'), dict(status='completed', found_keys=True, total_keys=2)])
def test_partial_failed_and_malformed_completions_never_succeed(body):
    act, _ = actuator([body])
    assert not act.prefetch('s')['ok']


def test_timeout_keeps_snapshot_and_does_not_resubmit():
    act, calls = actuator([dict(status='pending')], completion_timeout_s=.001)
    assert act.prefetch('s')['status'] == 'pending'
    assert act.prefetch('s')['status'] == 'pending'
    act.tokens = PromptTokens(lambda _: [5, 6, 7, 8], lambda _: [{'content': 'new'}])
    assert act.prefetch('s')['status'] == 'previous_prompt_pending'
    assert act.pending['s'][0] == (1, 2, 3, 4)
    assert len([c for c in calls if c.method == 'POST']) == 1


def test_no_pin_or_gpu_placement_is_claimed():
    act, calls = actuator()
    assert act.apply_touch(TouchDirective(session_id='s', expires_at=2), 1)['status'] == 'unsupported'
    for action, tier in [('pin', 'cpu'), ('prefetch', 'gpu'), ('promote', 'cpu'), ('demote', 'disk')]:
        directive = TierDirective(session_id='s', action=action, tier=tier, expires_at=2, eta_q10=0, eta_q90=1)
        assert act.apply_tier(directive, 1)['status'] == 'unsupported'
    assert act.release_expired(3) == [] and not calls


def test_cpu_prefetch_directive_and_expiry():
    act, _ = actuator()
    directive = TierDirective(session_id='s', action='prefetch', tier='cpu', expires_at=2, eta_q10=0, eta_q90=1)
    assert act.apply_tier(directive, 2)['status'] == 'expired'
    assert act.apply_tier(directive, 1)['ok']


def test_transport_error_remains_fail_open():
    act, _ = actuator()
    act.client = httpx.Client(transport=httpx.MockTransport(lambda _: (_ for _ in ()).throw(ConnectionError('down'))))
    assert act.prefetch('s')['status'] == 'error' and act.errors == 1


def test_prompt_tokens_refresh_without_mutating_an_existing_snapshot():
    source = {'s': [{'content': 'old'}]}
    calls = []
    tokens = PromptTokens(lambda m: calls.append(m) or [len(m[0]['content'])], source.get)
    before = tuple(tokens.get('s'))
    assert tokens.get('s') == [3] and len(calls) == 1
    source['s'] = [{'content': 'longer'}]
    assert tokens.get('s') == [6] and before == (3,)


@pytest.mark.parametrize('version', ['0.5.4', '0.5.6', {'version': '0.5.5'}])
def test_incompatible_backend_is_rejected_before_submission(version):
    act, calls = actuator()
    act.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=version)))
    assert act.prefetch('s')['status'] == 'error'
    assert not act.pending and not calls


def test_consumed_or_missing_completion_is_unknown():
    act, _ = actuator([dict(status='pending')], completion_timeout_s=.001)
    assert act.prefetch('s')['status'] == 'pending'
    act.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    assert act.prefetch('s')['status'] == 'unknown'
    assert not act.pending


def test_pending_limit_does_not_submit_more_work():
    act, calls = actuator([dict(status='pending')], completion_timeout_s=.001, max_pending=1)
    assert act.prefetch('a')['status'] == 'pending'
    assert act.prefetch('b')['status'] == 'error'
    assert len([c for c in calls if c.method == 'POST']) == 1


def test_failed_poll_preserves_request_for_reconciliation():
    act, calls = actuator([dict(status='pending')], completion_timeout_s=.001)
    act.prefetch('s')
    original = act.client
    act.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    assert act.prefetch('s')['status'] == 'error' and 's' in act.pending
    act.client = original
    assert act.prefetch('s')['status'] == 'pending'
    assert len([c for c in calls if c.method == 'POST']) == 1


def test_acknowledgement_requires_integer_chunk_count():
    from atfm.control.lmcache_protocol import submitted
    with pytest.raises(ValueError):
        submitted(dict(status='submitted', chunks=True, request_id='abc'), 1)


def test_asynchronous_mode_submits_without_waiting_and_reconciles_later():
    act, calls = actuator([dict(status='pending'), dict(status='completed', found_keys=2, total_keys=2)],
                          wait_for_completion=False)
    first = act.prefetch('s')
    assert first == dict(ok=None, status='submitted', request_id='abc')
    assert not [c for c in calls if c.url.path.endswith('/abc')]
    assert act.release_expired(1.0) == [] and 's' in act.pending            # still pending on the first poll
    act.release_expired(2.0)
    assert not act.pending and act.outcomes == {'completed': 1}


def test_asynchronous_resubmission_of_the_same_prompt_is_not_duplicated():
    act, calls = actuator([dict(status='pending')], wait_for_completion=False)
    assert act.prefetch('s')['status'] == 'submitted'
    assert act.prefetch('s')['status'] == 'submitted'
    assert len([c for c in calls if c.method == 'POST']) == 1


def test_outcomes_count_partial_and_unknown_completions():
    act, _ = actuator([dict(status='completed', found_keys=1, total_keys=2)])
    act.prefetch('s')
    assert act.outcomes == {'partial': 1}
