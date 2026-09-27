"""Step 5 prerequisites: the touch_random ablation (same touch rate, random targets) and touches that wait
for a slot instead of being dropped."""
import numpy as np

from atfm.proxy.config import ProxyConfig
from atfm.sim.core import Simulator
from atfm.sim.engine import EngineConfig
from tests.sim.test_touch_arm import _prog


def test_random_touch_arm_touches_at_the_same_rate_without_a_forecast():
    from atfm.sim.kv_placement import RandomTouchPolicy
    engines = [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]
    progs = [_prog(f"s{i}", "background", 2.0 * i, 800, "bash", 30.0, turns=4) for i in range(6)]
    pol = RandomTouchPolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=0.5), budget_per_s=1.0, tick_s=5.0)
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    sim.run()
    assert pol.name == "touch_random" and sim.touches > 0
    assert sim.touches <= (sim.now / 5.0 + 1) * 5                      # never more than the budget
    assert len({sid for _, sid in pol.touched}) >= 3                     # spread over sessions, not a ranking
    from atfm_experiments.h2sim import ARMS
    assert "touch_random" in ARMS


def test_failed_touches_are_retried_when_a_slot_frees_within_the_tick():
    from atfm.sim.kv_placement import OracleTouchPolicy
    engines = [EngineConfig(kv_blocks=3000, max_batch=1, prefill_tps=20000.0, decode_tps=40.0, touch_step_s=0.05)]
    progs = [_prog("a", "interactive", 0.0, 800, "bash", 20.0, turns=6), _prog("b", "interactive", 0.5, 800, "bash", 20.0, turns=6)]
    pol = OracleTouchPolicy(window=1, cfg=ProxyConfig(upstream_url="x", beta=0.5), horizon_s=30.0, age_s=1000.0,
                            budget_per_s=2.0, retry=True)
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    sim.run()
    assert sim.touches > 0 and sim.touch_retries > 0                     # some touches waited for the batch and then ran
    assert sim.touch_fails < sim.touches                                 # a wait is not a failure
