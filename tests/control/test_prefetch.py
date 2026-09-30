import pytest

from atfm.board.state import SessionState
from atfm.control.prefetch import PrefetchPlanner


def state(sid, phase='tool_running', ctx=32768, turn=1):
    return SessionState(session_id=sid, cls='background', tenant='t0', parent_session_id=None,
                        phase=phase, turn_index=turn, t_phase_start=0.0, ctx_tokens=ctx)


def planner(**kw):
    cfg = dict(bytes_per_token=147456, warm_bytes_per_s=1.4e9, overhead_s=0.2, interval_s=5.0, margin_s=1.0,
               budget_bytes=40 * 2**30, max_per_plan=8)
    cfg.update(kw)
    return PrefetchPlanner(**cfg)


def test_lead_time_grows_with_context_bytes():
    p = planner()
    assert p.lead_s(0) == pytest.approx(.2)
    assert p.lead_s(98304) == pytest.approx(.2 + 98304 * 147456 / 1.4e9)


def test_warms_when_earliest_return_is_inside_lead_plus_interval():
    p = planner()
    lead = p.lead_s(32768)
    soon = {'a': (lead + 5.0, lead + 20, lead + 60)}
    later = {'a': (lead + 7.0, lead + 20, lead + 60)}
    (d,) = p.plan(100.0, soon, {'a': state('a')})
    assert (d.session_id, d.action, d.tier) == ('a', 'prefetch', 'cpu')
    assert d.expires_at > 100.0 and d.eta_q10 == pytest.approx(lead + 5.0)
    assert planner().plan(100.0, later, {'a': state('a')}) == []


def test_skips_running_sessions_tiny_contexts_and_hopeless_warms():
    p = planner()
    q = {'a': (0.0, 1.0, 2.0)}
    assert p.plan(0.0, q, {'a': state('a', phase='llm_running')}) == []
    assert p.plan(0.0, q, {'a': state('a', ctx=8)}) == []
    lead = p.lead_s(98304)
    assert p.plan(0.0, {'a': (0.1, 0.2, lead - 0.5)}, {'a': state('a', ctx=98304)}) == []


def test_one_warm_per_session_turn():
    p = planner()
    q = {'a': (1.0, 5.0, 30.0)}
    assert len(p.plan(0.0, q, {'a': state('a', turn=3)})) == 1
    assert p.plan(5.0, q, {'a': state('a', turn=3)}) == []
    assert len(p.plan(10.0, q, {'a': state('a', turn=4)})) == 1


def test_byte_budget_and_per_plan_cap_prefer_soonest_returns():
    p = planner(budget_bytes=2 * 32768 * 147456, max_per_plan=8)
    q = {s: (1.0 + i, 10.0, 60.0) for i, s in enumerate('cba')}
    states = {s: state(s) for s in 'abc'}
    assert [d.session_id for d in p.plan(0.0, q, states)] == ['c', 'b']
    assert [d.session_id for d in planner(max_per_plan=1).plan(0.0, q, states)] == ['c']


def test_budget_is_released_when_warms_settle():
    p = planner(budget_bytes=32768 * 147456)
    q = {'a': (1.0, 5.0, 30.0), 'b': (1.0, 5.0, 30.0)}
    states = {'a': state('a'), 'b': state('b')}
    assert len(p.plan(0.0, q, states)) == 1
    p.settle('a')
    assert [d.session_id for d in p.plan(1.0, q, states)] == ['b']


def test_in_flight_warms_expire_after_their_deadline():
    p = planner(budget_bytes=32768 * 147456)
    states = {'a': state('a'), 'b': state('b')}
    (d,) = p.plan(0.0, {'a': (1.0, 5.0, 30.0)}, states)
    assert p.plan(d.expires_at - 1, {'b': (1.0, 5.0, 30.0)}, states) == []
    assert len(p.plan(d.expires_at + 1, {'b': (1.0, 5.0, 30.0)}, states)) == 1


def test_rejects_invalid_calibration():
    with pytest.raises(ValueError):
        planner(warm_bytes_per_s=0)
    with pytest.raises(ValueError):
        planner(budget_bytes=-1)


def test_budget_returns_when_session_starts_its_next_call():
    p = planner(budget_bytes=32768 * 147456)
    q = {'a': (1.0, 5.0, 30.0), 'b': (1.0, 5.0, 30.0)}
    assert [d.session_id for d in p.plan(0.0, q, {'a': state('a'), 'b': state('b')})] == ['a']
    returned = {'a': state('a', phase='llm_running'), 'b': state('b')}
    assert [d.session_id for d in p.plan(1.0, q, returned)] == ['b']


def test_median_trigger_waits_until_the_likely_return_is_near():
    lead = planner().lead_s(32768)
    q = {'a': (0.5, lead + 20.0, lead + 90.0)}                       # could return soon, probably much later
    assert len(planner(trigger='q10').plan(0.0, q, {'a': state('a')})) == 1
    assert planner(trigger='q50').plan(0.0, q, {'a': state('a')}) == []
    near = {'a': (0.5, lead + 4.0, lead + 90.0)}
    assert len(planner(trigger='q50').plan(0.0, near, {'a': state('a')})) == 1


def test_rewarm_after_interval_within_the_same_turn():
    p = planner(rewarm_after_s=20.0)
    q = {'a': (0.5, 5.0, 60.0)}
    (d,) = p.plan(0.0, q, {'a': state('a')})
    p.settle('a')
    assert p.plan(10.0, q, {'a': state('a')}) == []                   # too soon: likely still resident
    assert len(p.plan(25.0, q, {'a': state('a')})) == 1


def test_unknown_trigger_is_rejected():
    with pytest.raises(ValueError):
        planner(trigger='q99')
