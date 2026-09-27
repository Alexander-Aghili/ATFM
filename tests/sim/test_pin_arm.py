"""Hard-pin placement (what LMCache pin executes): pinned sessions are excluded from eviction until their
directive expires, within a block budget; and a touch that yields to real requests."""
import numpy as np

from atfm.proxy.config import ProxyConfig
from atfm.sim.core import Simulator
from atfm.sim.engine import EngineConfig, Request, Worker
from tests.sim.test_touch_arm import _prog


def _req(sid, isl, rid=None):
    return Request(request_id=rid or f"{sid}:0", session_id=sid, cls="background", isl_total=isl, isl_new=isl, osl=16, tier=0, index=0.0, t_queued=0.0)


def test_worker_pins_are_excluded_from_eviction_until_unpinned():
    w = Worker("w0", EngineConfig(kv_blocks=160, max_batch=4, prefill_tps=1e4, decode_tps=100.0))
    for s, t in (("a", 0.0), ("b", 1.0), ("c", 2.0)):
        w.submit(_req(s, 800), t); w.schedule(t); w.complete(f"{s}:0", t + 0.5)
    w.pin("a", until=100.0)
    w.submit(_req("d", 800), 3.0); w.schedule(3.0)
    assert "a" in w.resident and "b" not in w.resident                 # LRU head 'a' was pinned, so 'b' went
    w.expire_pins(now=101.0)
    w.submit(_req("e", 800), 102.0); w.schedule(102.0)
    assert "a" not in w.resident                                       # unpinned: back in LRU order (oldest)
    w2 = Worker("w1", EngineConfig(kv_blocks=90, max_batch=4, prefill_tps=1e4, decode_tps=100.0))   # 51 pinned + 51 needed > 90
    w2.submit(_req("p", 800), 0.0); w2.schedule(0.0); w2.complete("p:0", 1.0)
    w2.pin("p", until=100.0)
    w2.submit(_req("q", 800), 2.0)
    assert w2.schedule(2.0) == [] and w2.pinned_blocks() == 51         # a pin can block admission: that is the cost


def test_forecast_pin_arm_pins_imminent_sessions_within_a_block_budget():
    from atfm.sim.kv_placement import OraclePinPolicy
    engines = [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]
    progs = [_prog("soon", "interactive", 0.0, 800, "bash", 20.0, turns=4), _prog("late", "background", 0.0, 800, "build", 400.0)]
    pol = OraclePinPolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=0.5), horizon_s=30.0, pin_budget_blocks=200)
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    log = sim.run()
    assert sim.pins > 0 and "pins" in log.columns and pol.name == "oracle_pin"
    assert all(sid == "soon" or t >= 400.0 - 30.0 - 5.0 for t, sid in pol.pinned_log)
    assert max(pol.max_pinned_blocks_seen, 0) <= 200
    from atfm_experiments.h2sim import ARMS
    assert {"forecast_M1_pin", "forecast_M2_pin", "oracle_pin", "pin_random"} <= set(ARMS)


def test_yielding_touch_only_fires_when_a_slot_is_free_now():
    from atfm.sim.kv_placement import OracleTouchPolicy
    engines = [EngineConfig(kv_blocks=3000, max_batch=1, prefill_tps=20000.0, decode_tps=40.0, touch_step_s=0.05)]
    progs = [_prog("a", "interactive", 0.0, 800, "bash", 20.0, turns=6), _prog("b", "interactive", 0.5, 800, "bash", 20.0, turns=6)]
    pol = OracleTouchPolicy(window=1, cfg=ProxyConfig(upstream_url="x", beta=0.5), horizon_s=30.0, age_s=1000.0, budget_per_s=2.0, yield_to_requests=True)
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    sim.run()
    assert sim.touch_fails == 0 and sim.touch_retries == 0 and pol.skipped_busy > 0    # never dropped, never queued: skipped
