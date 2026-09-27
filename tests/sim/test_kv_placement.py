"""Forecast-driven KV placement: the engine evicts the idle session forecast to return *last*, instead of
the least recently used one. This is the lever the second batch of H2 points at; each test names the
production change that makes it pass."""
import numpy as np

from atfm.proxy.config import ProxyConfig
from atfm.sim.engine import EngineConfig, Request, Worker
from atfm.sim.programs import Program, Turn


def _req(sid, isl, rid=None):
    return Request(request_id=rid or f"{sid}:0", session_id=sid, cls="background", isl_total=isl, isl_new=isl, osl=16,
                   tier=0, index=0.0, t_queued=0.0)


def _fill(w, sids, isl=800):
    for s in sids:
        w.submit(_req(s, isl), 0.0)
    w.schedule(0.0)
    for s in sids:
        w.complete(f"{s}:0", 1.0)


def test_worker_victim_policy_overrides_lru_order():
    # 3 idle sessions of 51 blocks each fill a 160-block cache; admitting a 4th needs one victim
    w = Worker("w0", EngineConfig(kv_blocks=160, max_batch=4, prefill_tps=1e4, decode_tps=100.0))
    _fill(w, ["a", "b", "c"])
    w.victim_policy = lambda candidates: sorted(candidates, reverse=True)        # evict "c" first, not LRU "a"
    w.submit(_req("d", 800), 2.0)
    w.schedule(2.0)
    assert "c" not in w.resident and "a" in w.resident and "d" in w.resident
    assert w.last_evictions == [("d:0", "c")]
    w2 = Worker("w1", EngineConfig(kv_blocks=160, max_batch=4, prefill_tps=1e4, decode_tps=100.0))
    _fill(w2, ["a", "b", "c"])
    w2.submit(_req("d", 800), 2.0)
    w2.schedule(2.0)
    assert "a" not in w2.resident                                               # default stays LRU


def _prog(sid, cls, t_arrival, isl, tool, dur):
    turns = [Turn(isl_new=isl, osl=10, tool_name=tool, tool_duration=dur, backend_id="ci", progress=[], think=False),
             Turn(isl_new=100, osl=10, tool_name=None, tool_duration=None, backend_id="ci", progress=[], think=False)]
    return Program(session_id=sid, cls=cls, tenant="t", t_arrival=t_arrival, turns=turns, deadline_s=None, parent=None, spawn_at_turn={})


def test_oracle_kv_policy_evicts_the_session_that_returns_last():
    from atfm.sim.core import Simulator
    from atfm.sim.kv_placement import OracleKvPolicy
    engines = [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]
    progs = [_prog("soon", "background", 0.0, 800, "pytest", 20.0), _prog("late", "background", 0.0, 800, "build", 400.0)]
    pol = OracleKvPolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=0.5))
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    sim.prime()
    sim.run(until=10.0)                         # both sessions have made their first call and are in their tools
    pol.on_tick(sim, 10.0)
    assert pol.kv_victims(sim, sim.workers[0], ["soon", "late"]) == ["late", "soon"]
    assert pol.name == "oracle_kv"
    assert pol.e_tool_next(sim, None) == 0.0          # placement only: the index is proxy_rules' index


def test_forecast_kv_policy_orders_by_predicted_return_and_prefers_unknown_first():
    from atfm.sim.core import Simulator
    from atfm.sim.forecast_arm import fit_predictor_on_programs
    from atfm.sim.kv_placement import ForecastKvPolicy
    engines = [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]
    train = [_prog(f"t{k}", "background", 100.0 * k, 800, "pytest" if k % 2 else "build", 20.0 if k % 2 else 400.0) for k in range(20)]
    pred, table = fit_predictor_on_programs("M1", train, engines, np.random.default_rng(1))
    progs = [_prog("soon", "background", 0.0, 800, "pytest", 20.0), _prog("late", "background", 0.0, 800, "build", 400.0)]
    pol = ForecastKvPolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=0.5), predictor=pred, train_table=table, horizons=[30.0, 120.0], n=64)
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    sim.prime()
    sim.run(until=10.0)
    pol.on_tick(sim, 10.0)
    order = pol.kv_victims(sim, sim.workers[0], ["soon", "late", "never-seen"])
    assert order[0] == "never-seen" and order[1] == "late" and order[2] == "soon"
    assert pol.name == "forecast_M1_kv"
    assert pol.e_tool_next(sim, None) == 0.0


def test_kv_arms_registered_and_wired_into_workers():
    from atfm.experiments.h2sim import ARMS, H2SimConfig, _arm, build_simulator
    from atfm.sim.kv_placement import OracleKvPolicy
    assert {"forecast_M1_kv", "forecast_M2_kv", "oracle_kv"} <= set(ARMS)
    cfg = H2SimConfig(name="t", regime="short_tool")
    engines = [EngineConfig(**e) for e in cfg.engines]
    pol = _arm("oracle_kv", cfg, engines, None, np.random.default_rng(0))
    assert isinstance(pol, OracleKvPolicy)
    sim = build_simulator(cfg, [_prog("a", "background", 1.0, 100, "bash", 1.0)], engines, pol, seed=0)
    assert all(w.victim_policy is not None for w in sim.workers)


def test_oracle_rule_noidx_arm_drops_the_next_tool_index_term():
    """Diagnostic for the index: same true-demand hold rule, but E[next tool] = 0 like proxy_rules."""
    from atfm.experiments.h2sim import ARMS, H2SimConfig, _arm
    from atfm.sim.forecast_arm import OracleRuleNoIdxPolicy
    assert "oracle_rule_noidx" in ARMS
    cfg = H2SimConfig(name="t", regime="short_tool")
    engines = [EngineConfig(**e) for e in cfg.engines]
    pol = _arm("oracle_rule_noidx", cfg, engines, None, np.random.default_rng(0))
    assert isinstance(pol, OracleRuleNoIdxPolicy) and pol.name == "oracle_rule_noidx"
    assert pol.e_tool_next(None, None) == 0.0


def test_oracle_kv_treats_sessions_waiting_at_the_proxy_as_imminent():
    """A session whose call is in the proxy queue has no heap event; it must be the last candidate evicted,
    not the first."""
    from atfm.sim.core import PendingCall, SessionRun, Simulator
    from atfm.sim.kv_placement import OracleKvPolicy
    engines = [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]
    progs = [_prog("late", "background", 0.0, 800, "build", 400.0), _prog("queued", "background", 0.0, 800, "bash", 1.0)]
    pol = OracleKvPolicy(window=1, cfg=ProxyConfig(upstream_url="x", beta=0.5))
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    sim.prime()
    sim.run(until=10.0)
    run = sim.sessions["queued"]
    call = PendingCall(session=run, turn_index=1, t_arrival=10.0, isl_total=900, isl_new=100, osl=10)
    sim.proxy_queue.append(call)                     # waiting for the window, no heap event of its own
    pol.on_tick(sim, 10.0)
    assert pol.kv_victims(sim, sim.workers[0], ["queued", "late"]) == ["late", "queued"]


def test_oracle_kv_treats_sessions_waiting_in_a_worker_queue_as_imminent():
    """A released request that could not be scheduled yet (batch full or no KV room) sits in the worker
    queue with no heap event; its session is about to run and must be the last evicted."""
    from atfm.sim.core import Simulator
    from atfm.sim.kv_placement import OracleKvPolicy
    engines = [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]
    progs = [_prog("late", "background", 0.0, 800, "build", 400.0), _prog("wq", "background", 0.0, 800, "bash", 1.0)]
    pol = OracleKvPolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=0.5))
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    sim.prime()
    sim.run(until=10.0)
    w = sim.workers[0]
    w.submit(Request(request_id="wq:1", session_id="wq", cls="background", isl_total=900, isl_new=100, osl=10, tier=0, index=0.0, t_queued=10.0), 10.0)
    pol.on_tick(sim, 10.0)
    assert pol.kv_victims(sim, w, ["wq", "late"]) == ["late", "wq"]


def test_oracle_kv_gives_running_sessions_a_return_time():
    """A session that is running at tick time has no start/arrive/tool_end event in the heap; its next
    call is at the end of this one plus its tool. Without that, it becomes 'unknown' after the call and
    is evicted first (the defect that made oracle_kv worse than LRU)."""
    from atfm.sim.core import Simulator
    from atfm.sim.kv_placement import OracleKvPolicy
    engines = [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]
    progs = [_prog("late", "background", 0.0, 800, "build", 400.0), _prog("run", "background", 0.0, 800, "bash", 5.0)]
    pol = OracleKvPolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=0.5))
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    sim.prime()
    sim.run(until=0.5)                       # both first calls admitted and running (prefill 0.04 s, decode 0.25 s)
    w = sim.workers[0]
    w.submit(Request(request_id="run:9", session_id="run", cls="background", isl_total=900, isl_new=100, osl=400, tier=0, index=0.0, t_queued=0.5), 0.5)
    w.schedule(0.5)                          # 'run' is now running with t_end = 0.5 + 0.045 + 10 s
    assert any(r.session_id == "run" for r, _, _ in w.running.values())
    pol.on_tick(sim, 0.6)
    assert np.isfinite(pol._eta["run"]) and pol._eta["run"] < pol._eta["late"]
    assert pol.kv_victims(sim, w, ["run", "late"]) == ["late", "run"]
