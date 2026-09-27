"""Keep-alive touch in the simulator: the deployable placement mechanism (L2 design note)."""
import numpy as np

from atfm.proxy.config import ProxyConfig
from atfm.sim.core import Simulator
from atfm.sim.engine import EngineConfig, Request, Worker
from atfm.sim.programs import Program, Turn


def _req(sid, isl, rid=None, osl=16):
    return Request(request_id=rid or f"{sid}:0", session_id=sid, cls="background", isl_total=isl, isl_new=isl, osl=osl,
                   tier=0, index=0.0, t_queued=0.0)


def test_worker_touch_refreshes_recency_and_tracks_last_used():
    w = Worker("w0", EngineConfig(kv_blocks=160, max_batch=4, prefill_tps=1e4, decode_tps=100.0))
    for s, t in (("a", 0.0), ("b", 1.0), ("c", 2.0)):
        w.submit(_req(s, 800), t); w.schedule(t); w.complete(f"{s}:0", t + 0.5)
    assert w.last_used["a"] == 0.5 and w.frontier_age(now=10.0) == 9.5           # oldest idle resident is 'a'
    kind, evicted = w.touch("a", blocks=51, now=10.0)
    assert kind == "hit" and evicted == [] and list(w.resident)[-1] == "a" and w.last_used["a"] == 10.0
    w.submit(_req("d", 800), 11.0); w.schedule(11.0)                                 # needs room: LRU 'b' goes, not 'a'
    assert "b" not in w.resident and "a" in w.resident


def test_worker_touch_miss_prefetches_and_reports_recompute():
    w = Worker("w0", EngineConfig(kv_blocks=160, max_batch=4, prefill_tps=1e4, decode_tps=100.0))
    for s, t in (("a", 0.0), ("b", 1.0), ("c", 2.0)):
        w.submit(_req(s, 800), t); w.schedule(t); w.complete(f"{s}:0", t + 0.5)
    kind, evicted = w.touch("z", blocks=51, now=5.0)                                 # not resident: speculative prefill
    assert kind == "miss" and evicted == ["a"] and w.resident["z"] == 51
    w2 = Worker("w1", EngineConfig(kv_blocks=100, max_batch=4, prefill_tps=1e4, decode_tps=100.0))
    w2.submit(_req("r", 800, osl=100), 0.0); w2.schedule(0.0)                        # running request holds 57 of 100 blocks
    assert w2.touch("z", blocks=90, now=1.0)[0] == "fail"


def _prog(sid, cls, t_arrival, isl, tool, dur, turns=2):
    ts = [Turn(isl_new=isl, osl=10, tool_name=tool, tool_duration=dur, backend_id="ci", progress=[], think=False)] * (turns - 1)
    ts = ts + [Turn(isl_new=100, osl=10, tool_name=None, tool_duration=None, backend_id="ci", progress=[], think=False)]
    return Program(session_id=sid, cls=cls, tenant="t", t_arrival=t_arrival, turns=ts, deadline_s=None, parent=None, spawn_at_turn={})


def test_oracle_touch_arm_touches_imminent_at_risk_sessions_and_logs_costs():
    from atfm.sim.kv_placement import OracleTouchPolicy
    engines = [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]
    progs = [_prog("soon", "interactive", 0.0, 800, "bash", 20.0), _prog("late", "background", 0.0, 800, "build", 400.0)]
    pol = OracleTouchPolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=0.5), horizon_s=30.0, age_s=1000.0, budget_per_s=1.0)
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    log = sim.run()
    assert sim.touches >= 1 and sim.touch_hits + sim.touch_misses + sim.touch_fails == sim.touches
    sids = {sid for _, sid in pol.touched}
    assert "soon" in sids
    # 'late' may be touched only once its 400 s build is within the horizon of ending (plus one tick of slack)
    assert all(t >= 400.0 - 30.0 - 5.0 for t, sid in pol.touched if sid == "late")
    assert "touches" in log.columns and log["touches"].iloc[-1] <= sim.touches
    assert pol.name == "oracle_touch"


def test_forecast_touch_arm_registered_and_uses_predictor_quantiles():
    from atfm_experiments.h2sim import ARMS, H2SimConfig, _arm
    from atfm.sim.kv_placement import ForecastTouchPolicy
    assert {"forecast_M1_touch", "forecast_M2_touch", "oracle_touch"} <= set(ARMS)
    cfg = H2SimConfig(name="t", regime="short_tool", touch_budget_per_s=2.0, touch_horizon_s=45.0)
    engines = [EngineConfig(**e) for e in cfg.engines]
    from atfm.sim.programs import programs_from_spec
    from atfm_experiments.h2sim import regime_spec
    train = programs_from_spec(regime_spec("short_tool", 300.0, 1), np.random.default_rng(1))
    pol = _arm("forecast_M2_touch", cfg, engines, train, np.random.default_rng(0))
    assert isinstance(pol, ForecastTouchPolicy) and pol.name == "forecast_M2_touch"
    assert pol.touch.budget_per_s == 2.0 and pol.touch.horizon_s == 45.0 and pol.e_tool_next(None, None) == 0.0
