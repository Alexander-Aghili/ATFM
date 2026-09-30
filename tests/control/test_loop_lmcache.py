"""Only verified supported cache operations count as applied control actions."""
from atfm.control.loop import ControlLoop
from tests.control.test_lmcache import actuator
from tests.control.test_loop import DIRECTIVES, FakeClient, FakeResponse


def test_loop_counts_completed_prefetch_but_does_not_claim_pins(tmp_path):
    directive = dict(kind='tier', session_id='s', action='prefetch', tier='cpu',
                     eta_q10=0, eta_q90=1, expires_at=200)
    directives = DIRECTIVES | {'tier': [directive]}
    http = FakeClient({('POST', 'http://board/tick'): FakeResponse(200, {'sessions': 2}),
                       ('POST', 'http://board/directives'): FakeResponse(200, directives),
                       ('POST', 'http://proxy/directives/batch'): FakeResponse(200, {'ok': True, 'applied': 1, 'expired': 0})})
    act, calls = actuator()
    loop = ControlLoop('http://board', 'http://proxy', client=http, lmcache=act)
    result = loop.step(now=100)
    assert result['holds'] == 1 and result['pins'] == 0 and result['tier_applied'] == 1
    assert not any(c.url.path in ('/pin', '/unpin', '/move') for c in calls)


def test_asynchronous_prefetch_is_counted_as_submitted_and_outcomes_are_logged(tmp_path):
    import json
    directive = dict(kind='tier', session_id='s', action='prefetch', tier='cpu',
                     eta_q10=0, eta_q90=1, expires_at=200)
    http = FakeClient({('POST', 'http://board/tick'): FakeResponse(200, {'sessions': 1}),
                       ('POST', 'http://board/directives'): FakeResponse(200, DIRECTIVES | {'tier': [directive], 'holds': []})})
    act, _ = actuator([dict(status='completed', found_keys=2, total_keys=2)], wait_for_completion=False)
    log = tmp_path / 'control.jsonl'
    result = ControlLoop('http://board', 'http://proxy', client=http, lmcache=act, log_path=log).step(now=100)
    assert result['tier_applied'] == 0 and result['cache_submitted'] == 1
    assert json.loads(log.read_text())['cache_outcomes'] == {'completed': 1}   # reconciled in the same step
