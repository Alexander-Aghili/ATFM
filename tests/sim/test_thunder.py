import numpy as np
from atfm.sim.programs import Program, Turn
from atfm.sim.engine import EngineConfig
from atfm.sim.core import Simulator
from atfm.sim.policies import WorkingSetPolicy

def _prog(sid, t0, isl, cls="background"):
    return Program(session_id=sid, cls=cls, tenant="t", t_arrival=t0, turns=[Turn(isl, 10, "bash", 5.0, "local"), Turn(16, 1, None, None)])

def test_working_set_budget_pauses_and_resumes_smallest_first():
    # calls take about a second, so arrivals overlap; the 101-block "big" session alone exceeds the 100-block budget
    eng = EngineConfig(kv_blocks=100000, max_batch=8, prefill_tps=1000.0, decode_tps=1e6)
    progs = [_prog("big", 0.0, 1600), _prog("mid", 0.001, 800), _prog("small", 0.002, 400), _prog("it", 0.003, 800, cls="interactive")]
    pol = WorkingSetPolicy(budget_blocks=100, low_watermark=0.5)
    log = Simulator(progs, [eng], pol, harness_overhead_s=0.0, rng=np.random.default_rng(0)).run()
    first_turn = log[log.turn_index == 0].sort_values("t_release")
    order = first_turn.session_id.tolist()
    assert order[0] == "big" and order[1] == "it"                       # interactive is never paused
    assert order.index("small") < order.index("mid")                   # smallest context resumes first
    assert (first_turn[first_turn.session_id.isin(["small", "mid"])].queue_proxy_s > 0).all()
