"""GDP planner (spec 6.2): 30 s slots over 15 min, greedy ration-by-schedule on the forecast samples,
per-slot per-resource chance constraint, hard cap, per-tenant fairness. Directives expire."""
import numpy as np
import pytest

from atfm.control import Deferrable, GdpPlanner, HoldDirective
from atfm.schema.forecast import ForecastSnapshot


def _snap(t, horizons, kv_by_h, pf_by_h, n=64, jitter=0.0, rng=None):
    """Cumulative interactive demand samples per horizon (H, n): kv blocks and prefill tokens."""
    rng = rng or np.random.default_rng(0)
    H = len(horizons)
    kv = np.array(kv_by_h, float)[:, None] + jitter * rng.standard_normal((H, n))
    pf = np.array(pf_by_h, float)[:, None] + jitter * 10 * rng.standard_normal((H, n))
    z = np.zeros((H, n))
    return ForecastSnapshot(t=t, horizons=list(horizons), model_id="test",
                            samples={"kv_blocks": {"interactive": kv, "background": z},
                                     "prefill_tokens": {"interactive": pf, "background": z}})


HORIZONS = [30.0, 60.0, 90.0, 120.0]


def test_greedy_ration_by_schedule_releases_when_the_slot_has_room():
    # capacity 1000 blocks; interactive demand per slot: 900, 100, 100, 100 (cumulative 900, 1000, 1100, 1200)
    snap = _snap(0.0, HORIZONS, [900, 1000, 1100, 1200], [0, 0, 0, 0])
    planner = GdpPlanner(slot_s=30.0, horizon_s=120.0, eps=0.1, max_hold_s=600.0)
    cap = {"kv_blocks": 1000.0, "prefill_tokens": 1e9}
    a = Deferrable(session_id="a", tenant="t1", eta_s=0.0, kv_blocks=200, prefill_tokens=100, cost_per_s=1.0)
    b = Deferrable(session_id="b", tenant="t1", eta_s=0.0, kv_blocks=50, prefill_tokens=100, cost_per_s=1.0)
    out = planner.plan(now=0.0, snap=snap, capacity=cap, deferrable=[b, a])
    d = {x.session_id: x for x in out}
    assert d["b"].release_not_before == 0.0                     # 900 + 50 <= 1000 in slot 0
    assert d["a"].release_not_before == 30.0                    # slot 0 would be 1150 > 1000; slot 1 has 100 + 200
    assert all(isinstance(x, HoldDirective) and x.expires_at == 30.0 for x in out)
    assert d["a"].reason == "gdp" and d["b"].reason == "gdp"


def test_infeasible_everywhere_hits_the_cap_not_a_loop():
    snap = _snap(0.0, HORIZONS, [2000, 4000, 6000, 8000], [0, 0, 0, 0])   # 2000 blocks of demand in every slot
    planner = GdpPlanner(slot_s=30.0, horizon_s=120.0, eps=0.1, max_hold_s=60.0)
    a = Deferrable(session_id="a", tenant="t1", eta_s=0.0, kv_blocks=10, prefill_tokens=10, cost_per_s=1.0)
    out = planner.plan(now=100.0, snap=snap, capacity={"kv_blocks": 1000.0, "prefill_tokens": 1e9}, deferrable=[a])
    assert out[0].release_not_before == 160.0 and out[0].reason == "capped"


def test_prefill_resource_binds_too():
    snap = _snap(0.0, HORIZONS, [0, 0, 0, 0], [15000, 15000, 30000, 30000])   # 15k tokens in slot 0, 15k in slot 2
    planner = GdpPlanner(slot_s=30.0, horizon_s=120.0, eps=0.1, max_hold_s=600.0)
    a = Deferrable(session_id="a", tenant="t1", eta_s=0.0, kv_blocks=1, prefill_tokens=6000, cost_per_s=1.0)
    out = planner.plan(0.0, snap, {"kv_blocks": 1e9, "prefill_tokens": 20000.0 * 30.0 / 30.0}, [a])
    assert out[0].release_not_before == 30.0                    # slot 0: 15000 + 6000 > 20000; slot 1: 0 + 6000


def test_property_never_violates_cap_or_its_own_chance_constraint():
    rng = np.random.default_rng(1)
    planner = GdpPlanner(slot_s=30.0, horizon_s=300.0, eps=0.1, max_hold_s=120.0)
    hz = [30.0 * k for k in range(1, 11)]
    for _ in range(50):
        cum = np.cumsum(rng.integers(0, 600, size=10))
        snap = _snap(0.0, hz, cum, cum * 10, jitter=60.0, rng=rng)
        cap = {"kv_blocks": float(rng.integers(500, 2500)), "prefill_tokens": float(rng.integers(5000, 25000))}
        defs = [Deferrable(session_id=f"s{i}", tenant=f"t{i % 3}", eta_s=float(rng.integers(0, 120)),
                           kv_blocks=int(rng.integers(10, 300)), prefill_tokens=int(rng.integers(100, 3000)), cost_per_s=1.0)
                for i in range(8)]
        out = planner.plan(0.0, snap, cap, defs)
        assert len(out) == len(defs)
        for d in out:
            eta = next(x for x in defs if x.session_id == d.session_id).eta_s
            assert 0.0 <= d.release_not_before <= eta + 120.0 + 1e-9        # never more than the cap after resumption
        _assert_assignments_feasible(planner, snap, defs, cap)


def _assert_assignments_feasible(planner, snap, defs, cap):
    # released (uncapped) sessions must satisfy the constraint on the planner's own samples
    assigned = planner.last_assignment                       # slot -> list of session ids
    for slot, sids in assigned.items():
        for res in ("kv_blocks", "prefill_tokens"):
            inter = planner.slot_demand(snap, res)[slot]     # (n,) samples of interactive demand in that slot
            load = inter + sum(next(x for x in defs if x.session_id == s).__getattribute__(res) for s in sids)
            assert (load <= cap[res]).mean() >= 1 - planner.eps - 1e-9


def test_tenant_fairness_caps_a_tenants_imposed_delay():
    snap = _snap(0.0, HORIZONS, [2000, 4000, 6000, 6000], [0, 0, 0, 0])   # slots 0-2 full (2000 each), slot 3 free
    planner = GdpPlanner(slot_s=30.0, horizon_s=120.0, eps=0.1, max_hold_s=600.0)
    a = Deferrable(session_id="a", tenant="gold", eta_s=0.0, kv_blocks=10, prefill_tokens=10, cost_per_s=1.0)
    b = Deferrable(session_id="b", tenant="t1", eta_s=0.0, kv_blocks=10, prefill_tokens=10, cost_per_s=1.0)
    out = planner.plan(0.0, snap, {"kv_blocks": 1000.0, "prefill_tokens": 1e9}, [a, b], tenant_max_delay={"gold": 30.0})
    d = {x.session_id: x for x in out}
    assert d["b"].release_not_before == 90.0                    # first feasible slot
    assert d["a"].release_not_before == 30.0 and d["a"].reason == "tenant_cap"   # fairness bound wins over the constraint
    assert planner.max_imposed_delay == {"gold": 30.0, "t1": 90.0}
