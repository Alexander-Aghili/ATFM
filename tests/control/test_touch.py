"""Placement touch controller (deployable form of KV placement, L2 design note) and the tier logger (spec 6.3)."""
from atfm.control import TierDirective, TouchDirective
from atfm.control.touch import Residency, TierLogger, TouchController


def _res(worker="w0", blocks=100, last_used=0.0):
    return Residency(worker_id=worker, blocks=blocks, last_used=last_used)


def test_touches_imminent_at_risk_sessions_most_imminent_first_within_budget():
    tc = TouchController(horizon_s=30.0, age_s=10.0, budget_per_s=1.0)
    now = 100.0
    resumptions = {"a": (5.0, 10.0, 20.0), "b": (2.0, 4.0, 8.0), "c": (50.0, 90.0, 200.0), "d": (1.0, 3.0, 6.0)}
    residency = {"a": _res(last_used=40.0), "b": _res(last_used=45.0), "c": _res(last_used=30.0), "d": _res(last_used=95.0)}
    # eviction frontier: blocks last used before now - 55 are next to go; a (60 old) and b (55 old) are at risk within the margin,
    # d was used 5 s ago and is safe, c is at risk but not imminent
    out = tc.plan(now, resumptions, residency, evict_frontier_age={"w0": 55.0})
    assert [d.session_id for d in out] == ["b", "a"]
    assert all(isinstance(d, TouchDirective) and d.worker_id == "w0" and d.expires_at == now + 30.0 for d in out)
    assert out[0].eta_q50 == 4.0


def test_budget_accumulates_across_ticks_and_zero_budget_emits_nothing():
    tc = TouchController(horizon_s=30.0, age_s=10.0, budget_per_s=0.0)
    assert tc.plan(0.0, {"a": (1.0, 2.0, 3.0)}, {"a": _res(last_used=-100.0)}, {"w0": 10.0}) == []
    tc = TouchController(horizon_s=30.0, age_s=10.0, budget_per_s=0.5, tick_s=1.0)
    res = {f"s{i}": _res(last_used=-100.0) for i in range(4)}
    rs = {f"s{i}": (1.0, 2.0, 3.0) for i in range(4)}
    n = [len(tc.plan(float(t), rs, res, {"w0": 10.0})) for t in range(4)]
    assert n == [0, 1, 0, 1]                                    # half a touch per tick accumulates to one every other tick


def test_unknown_worker_frontier_means_no_risk_and_missing_data_never_raises():
    tc = TouchController()
    assert tc.plan(0.0, {"a": (1.0, 2.0, 3.0)}, {}, {}) == []
    assert tc.plan(0.0, {}, {"a": _res()}, {"w0": 1.0}) == []


def test_tier_logger_picks_the_deepest_tier_that_can_be_back_in_time():
    tl = TierLogger(tier_lead_s={"gpu": 0.0, "cpu": 5.0, "disk": 60.0})
    now = 0.0
    out = tl.plan(now, {"soon": (2.0, 4.0, 8.0), "later": (20.0, 40.0, 80.0), "far": (300.0, 600.0, 900.0)})
    d = {x.session_id: x for x in out}
    assert d["soon"].action == "pin" and d["soon"].tier == "gpu"
    assert d["later"].action == "demote" and d["later"].tier == "cpu"     # 5 s lead < q10 20 s; disk's 60 s is not
    assert d["far"].action == "demote" and d["far"].tier == "disk"
    assert all(isinstance(x, TierDirective) and x.eta_q10 >= 0 and x.expires_at > now for x in out)
    assert tl.log[-1][0] == now and len(tl.log[-1][1]) == 3              # logged only, never enforced
