import numpy as np
from atfm.sim.programs import Program, Turn
from atfm.sim.engine import EngineConfig
from atfm.sim.core import Simulator
from atfm.sim.policies import NativePolicy

def _prog(sid, t0, cls="background", turns=None, deadline=None):
    turns = turns or [Turn(160, 16, "pytest", 30.0, "ci", [(10.0, 33.0, 100.0), (20.0, 66.0, 100.0)]), Turn(64, 16, None, None)]
    return Program(session_id=sid, cls=cls, tenant="t", t_arrival=t0, turns=turns, deadline_s=deadline)

def test_closed_loop_timing_and_log():
    sim = Simulator([_prog("a", 0.0)], [EngineConfig(kv_blocks=1000, max_batch=4, prefill_tps=1000.0, decode_tps=16.0)],
                    NativePolicy(), harness_overhead_s=0.5, rng=np.random.default_rng(0))
    log = sim.run()
    assert len(log) == 2
    r0, r1 = log.sort_values("turn_index").itertuples()
    assert r0.t_arrival == 0.0 and abs(r0.t_first_token - 0.16) < 1e-9 and abs(r0.t_end - 1.16) < 1e-9
    assert r0.tool_name == "pytest" and r0.tool_duration == 30.0
    assert abs(r1.t_arrival - (1.16 + 30.0 + 0.5)) < 1e-9
    assert r1.prefix_hit_tokens == 176 and r1.recomputed_tokens == 64
    kinds = [e.kind for e in sim.events.drain()]
    assert kinds.count("tool.progress") == 2 and "session.start" in kinds and kinds.count("llm.done") == 2
    assert sim.session_log[0]["turns"] == 2 and sim.session_log[0]["missed"] is False

def test_deadline_missed_but_session_completes():
    p = _prog("b", 0.0, cls="interactive", turns=[Turn(160, 16, "pytest", 100.0, "ci"), Turn(64, 16, None, None)], deadline=10.0)
    sim = Simulator([p], [EngineConfig(kv_blocks=1000, max_batch=4, prefill_tps=1000.0, decode_tps=16.0)], NativePolicy(), rng=np.random.default_rng(0))
    log = sim.run()
    assert len(log) == 2 and sim.session_log[0]["missed"] is True and bool(log["deadline_missed"].iloc[-1])

def test_spawned_child_arrives_at_parent_tool_end():
    child = Program(session_id="c", cls="background", tenant="t", t_arrival=0.0, turns=[Turn(32, 8, None, None)], parent="p")
    parent = Program(session_id="p", cls="background", tenant="t", t_arrival=0.0,
                     turns=[Turn(160, 16, "pytest", 30.0, "ci"), Turn(64, 16, None, None)], spawn_at_turn=[(0, child)])
    sim = Simulator([parent], [EngineConfig(kv_blocks=1000, max_batch=4, prefill_tps=1000.0, decode_tps=16.0)], NativePolicy(), rng=np.random.default_rng(0))
    log = sim.run()
    c = log[log.session_id == "c"]
    assert len(c) == 1 and abs(c.t_arrival.iloc[0] - (1.16 + 30.0)) < 1e-9

def test_native_priority_serves_interactive_first_when_batch_full():
    eng = EngineConfig(kv_blocks=10000, max_batch=1, prefill_tps=1000.0, decode_tps=1.0, priority=True)
    progs = [_prog("bg1", 0.0, turns=[Turn(16, 10, None, None)]), _prog("bg2", 0.01, turns=[Turn(16, 10, None, None)]),
             _prog("it", 0.02, cls="interactive", turns=[Turn(16, 10, None, None)])]
    sim = Simulator(progs, [eng], NativePolicy(priority_by_class=True), rng=np.random.default_rng(0))
    log = sim.run().sort_values("t_start")
    assert list(log.session_id)[:2] == ["bg1", "it"]
    assert (log.queue_worker_s >= 0).all() and (log.queue_proxy_s == 0).all()
