import numpy as np
from atfm.proxy.config import ProxyConfig
from atfm.sim.programs import Program, Turn
from atfm.sim.engine import EngineConfig
from atfm.sim.core import Simulator
from atfm.sim.policies import ProxyRulesPolicy, OraclePolicy

def _eng(batch=1):
    return EngineConfig(kv_blocks=100000, max_batch=batch, prefill_tps=1000.0, decode_tps=1.0)

def _one_turn(sid, t0, cls, osl=10, deadline=None):
    return Program(session_id=sid, cls=cls, tenant="t", t_arrival=t0, turns=[Turn(16, osl, None, None)], deadline_s=deadline)

def test_proxy_rules_window_and_tier_order():
    progs = [_one_turn("bg1", 0.0, "background"), _one_turn("bg2", 0.01, "background"), _one_turn("bg3", 0.02, "background"),
             _one_turn("it", 0.03, "interactive", deadline=5.0)]
    pol = ProxyRulesPolicy(window=1, cfg=ProxyConfig(upstream_url="x", slack_threshold_s=5.0))
    log = Simulator(progs, [_eng(batch=4)], pol, rng=np.random.default_rng(0)).run().sort_values("t_release")
    assert list(log.session_id)[:2] == ["bg1", "it"]
    assert (log.queue_proxy_s.iloc[1:] > 0).all() and (log.queue_worker_s == 0).all()

def test_oracle_uses_true_next_tool_duration():
    long_tool = Program(session_id="L", cls="background", tenant="t", t_arrival=0.0,
                        turns=[Turn(16, 10, "pytest", 300.0, "ci"), Turn(16, 1, None, None)])
    short_tool = Program(session_id="S", cls="background", tenant="t", t_arrival=0.0,
                         turns=[Turn(16, 10, "bash", 1.0, "local"), Turn(16, 1, None, None)])
    filler = _one_turn("F", 0.0, "background", osl=5)
    pol = OraclePolicy(window=1, cfg=ProxyConfig(upstream_url="x", beta=1.0), hold=False)
    log = Simulator([filler, short_tool, long_tool], [_eng(batch=4)], pol, rng=np.random.default_rng(0)).run()
    first = log.sort_values("t_release").session_id.tolist()
    assert first.index("L") < first.index("S")
