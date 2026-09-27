"""Step 3: honest touch costs in the simulator. A touch occupies a batch slot for one step; the touch
controller may target sessions whose KV is already gone (a miss = speculative prefill)."""
import numpy as np

from atfm.proxy.config import ProxyConfig
from atfm.sim.core import Simulator
from atfm.sim.engine import EngineConfig, Request, Worker


def _req(sid, isl, rid=None, osl=16):
    return Request(request_id=rid or f"{sid}:0", session_id=sid, cls="background", isl_total=isl, isl_new=isl, osl=osl, tier=0, index=0.0, t_queued=0.0)


def test_touch_occupies_a_batch_slot_for_one_step_and_blocks_admission_meanwhile():
    w = Worker("w0", EngineConfig(kv_blocks=1000, max_batch=1, prefill_tps=1e4, decode_tps=100.0, touch_step_s=0.5))
    w.submit(_req("a", 160), 0.0); w.schedule(0.0); w.complete("a:0", 1.0)
    kind, _ = w.touch("a", blocks=11, now=2.0)
    assert kind == "hit" and w.busy_until("touch") == 2.5 and len(w.running) == 1          # the slot is taken by the touch
    w.submit(_req("b", 160), 2.1)
    assert w.schedule(2.1) == []                                                              # batch full while the touch runs
    w.expire_touches(2.6)
    assert len(w.running) == 0 and len(w.schedule(2.6)) == 1
    assert w.touch_slot_s == 0.5


def test_simulator_expires_touch_slots_and_charges_slot_seconds():
    from atfm.sim.kv_placement import OracleTouchPolicy
    from tests.sim.test_touch_arm import _prog
    engines = [EngineConfig(kv_blocks=3000, max_batch=2, prefill_tps=20000.0, decode_tps=40.0, touch_step_s=0.2)]
    progs = [_prog("soon", "interactive", 0.0, 800, "bash", 20.0, turns=4), _prog("late", "background", 0.0, 800, "build", 400.0)]
    pol = OracleTouchPolicy(window=2, cfg=ProxyConfig(upstream_url="x", beta=0.5), horizon_s=30.0, age_s=1000.0, budget_per_s=1.0)
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    log = sim.run()
    assert sim.touches > 0 and abs(sim.touch_slot_s - sim.touches * 0.2) < 1e-9 and "touch_slot_s" in log.columns
    assert all(len(w.running) == 0 for w in sim.workers)                                      # no touch left occupying a slot


def test_touch_controller_targets_evicted_but_imminent_sessions_as_misses():
    from atfm.sim.kv_placement import OracleTouchPolicy
    from tests.sim.test_touch_arm import _prog
    engines = [EngineConfig(kv_blocks=120, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]   # room for ~2 contexts
    progs = [_prog(f"s{i}", "background", 5.0 * i, 800, "bash", 40.0) for i in range(4)]         # 4 sessions x 51 blocks
    pol = OracleTouchPolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=0.5), horizon_s=15.0, age_s=1000.0,
                            budget_per_s=1.0, prefetch=True)
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    sim.run()
    assert sim.touch_misses > 0 and sim.touch_prefill_tokens > 0                              # evicted contexts were prefetched
    pol2 = OracleTouchPolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=0.5), horizon_s=15.0, age_s=1000.0, budget_per_s=1.0)
    sim2 = Simulator([_prog(f"s{i}", "background", 5.0 * i, 800, "bash", 40.0) for i in range(4)], engines, pol2, rng=np.random.default_rng(0))
    sim2.run()
    assert sim2.touch_misses == 0                                                             # default: resident sessions only
