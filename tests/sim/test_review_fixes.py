"""Tests pinning the fixes from the l0-sim review (capacity measure, hold cap and accounting, working set,
native priority, re-consulted holds, registry pruning, makespan metrics, conditional index term, head-of-line,
child cloning, context reset, hysteresis)."""
import numpy as np, pandas as pd
from atfm.proxy.config import ProxyConfig
from atfm.schema.forecast import ForecastSnapshot
from atfm.schema.trace import TraceRow, TraceTable
from atfm.sim.programs import Program, Turn, programs_from_spec, programs_from_table
from atfm.sim.engine import EngineConfig, Worker, Request
from atfm.sim.core import Simulator
from atfm.sim.policies import NativePolicy, ProxyRulesPolicy, WorkingSetPolicy
from atfm.sim.forecast_arm import GdpLite, ForecastPolicy, fit_predictor_on_programs
from atfm.eval.serving import serving_metrics
from atfm.traces.synthetic import WorkloadSpec, ClassSpec, ToolSpec

def _req(rid, sid, isl, osl=8, tier=0, cls="background", isl_new=None):
    return Request(request_id=rid, session_id=sid, cls=cls, isl_total=isl, isl_new=isl if isl_new is None else isl_new, osl=osl, tier=tier)

# C1: capacity = blocks not held by running requests (idle resident blocks are evictable capacity)
def test_capacity_free_blocks_counts_idle_resident_blocks():
    w = Worker("w0", EngineConfig(kv_blocks=100, max_batch=4, prefill_tps=1000.0, decode_tps=10.0))
    w.submit(_req("r1", "a", 160), 0.0); w.schedule(0.0); w.complete("r1", 1.0)     # a: 11 idle blocks
    w.submit(_req("r2", "b", 160), 1.0); w.schedule(1.0)                             # b: 11 running blocks
    assert w.free_blocks() == 78 and w.capacity_free_blocks() == 89

def test_gdp_uses_occupancy_not_call_count():
    snap = ForecastSnapshot(t=0.0, horizons=[30.0], model_id="m",
                            samples={"kv_blocks": {"interactive": np.full((1, 8), 100.0), "background": np.zeros((1, 8))},
                                     "prefill_tokens": {"interactive": np.full((1, 8), 9000.0), "background": np.zeros((1, 8))}})
    g = GdpLite(slot_s=30.0, eps=0.1)
    # 3 calls of ~3000 tokens in 30 s, each taking 2 s of service: occupancy 0.2 slots -> no hold with 1 free slot
    assert g.hold_until(0.0, snap, free_blocks=1000, free_slots=1, mean_isl=3000.0, e_service_s=2.0) is None
    # each taking 20 s: occupancy 2 slots > 1 free -> hold
    assert g.hold_until(0.0, snap, free_blocks=1000, free_slots=1, mean_isl=3000.0, e_service_s=20.0) == 30.0

# C2 + I5: cap enforced in the core for any policy; holds re-consulted; caps counted
class _AlwaysHold(ProxyRulesPolicy):
    name = "always_hold"
    def on_arrival(self, sim, call):
        if call.session.program.cls == "background":
            self.last_hold_reason = "forever"
            return sim.now + 30.0
        return None

def _bg(sid, t0, turns=2):
    return Program(session_id=sid, cls="background", tenant="t", t_arrival=t0,
                   turns=[Turn(160, 8, "bash", 1.0, "local")] * (turns - 1) + [Turn(16, 1, None, None)])

def test_hold_cap_enforced_in_core_and_reconsulted():
    eng = EngineConfig(kv_blocks=10000, max_batch=8, prefill_tps=1e5, decode_tps=1e5)
    sim = Simulator([_bg("a", 0.0)], [eng], _AlwaysHold(window=8, cfg=ProxyConfig(upstream_url="x")), max_hold_s=100.0, rng=np.random.default_rng(0))
    log = sim.run()
    assert sim.caps >= 1 and (log.hold_s <= 100.0 + 1e-6).all() and log.hold_s.max() > 99.0
    assert (log.hold_s <= log.queue_proxy_s + 1e-9).all()

# C3: hold cost only for held calls; window waits are not holds
def test_no_hold_arms_report_zero_hold_cost():
    eng = EngineConfig(kv_blocks=2000, max_batch=1, prefill_tps=1000.0, decode_tps=10.0)
    progs = [_bg(f"b{i}", 0.0) for i in range(6)]
    sim = Simulator(progs, [eng], ProxyRulesPolicy(window=1, cfg=ProxyConfig(upstream_url="x")), rng=np.random.default_rng(0))
    log = sim.run()
    assert (log.queue_proxy_s > 0).any()
    assert (log.hold_s == 0).all() and (log.hold_kv_block_s == 0).all() and (log.evictions_caused == 0).all()

def test_held_session_kv_cost_and_displacement_are_charged():
    # a: held while resident; b arrives needing room -> a's retained blocks force evictions charged to a
    eng = EngineConfig(kv_blocks=30, max_batch=8, prefill_tps=1e5, decode_tps=1e5)
    class HoldA(ProxyRulesPolicy):
        name = "hold_a"
        def on_arrival(self, sim, call):
            if call.session.program.session_id == "a" and call.turn_index == 1:
                self.last_hold_reason = "test"
                return sim.now + 50.0
            return None
    progs = [_bg("a", 0.0), _bg("b", 5.0), _bg("c", 6.0)]
    log = Simulator(progs, [eng], HoldA(window=8, cfg=ProxyConfig(upstream_url="x")), rng=np.random.default_rng(0)).run()
    a1 = log[(log.session_id == "a") & (log.turn_index == 1)].iloc[0]
    assert a1.hold_s > 40 and a1.hold_kv_block_s > 0

# I4: engine priority on by default in the experiment config
def test_h2sim_engine_priority_default():
    from atfm_experiments.h2sim import H2SimConfig
    cfg = H2SimConfig(name="x", regime="short_tool")
    assert all(e.get("priority", False) for e in cfg.engines)

# I6: ended sessions are pruned from the forecast arm's registry
def test_forecast_arm_prunes_ended_sessions():
    tools = [ToolSpec(name="bash", weight=1.0, log_mu=np.log(2.0), log_sigma=0.3, signal="none")]
    spec = WorkloadSpec(duration_s=120.0, seed=0, classes=[ClassSpec(cls="background", rate_per_hour=600.0, turns_mean=2, isl0=500, isl_growth=100, osl_mean=20, tools=tools)])
    eng = [EngineConfig(kv_blocks=100000, max_batch=16, prefill_tps=1e5, decode_tps=1e4)]
    pred, table = fit_predictor_on_programs("M1", programs_from_spec(spec, np.random.default_rng(1)), eng, np.random.default_rng(1))
    pol = ForecastPolicy(window=16, cfg=ProxyConfig(upstream_url="x"), predictor=pred, train_table=table, horizons=[30.0], n=16, hold=False)
    sim = Simulator(programs_from_spec(spec, np.random.default_rng(0)), eng, pol, rng=np.random.default_rng(0))
    sim.run()
    pol.on_tick(sim, sim.now + 1.0)
    assert len(pol.registry.states(sim.now + 1.0)) == 0

# I7: throughput and GPU hours use the realized makespan
def test_serving_metrics_use_makespan():
    log = pd.DataFrame([{"session_id": "s", "class": "background", "tenant": "t", "turn_index": 0, "t_arrival": 0.0, "t_release": 0.0,
                         "t_first_token": 1.0, "t_end": 2.0, "isl": 10, "prefix_hit_tokens": 0, "recomputed_tokens": 10, "held_s": 0.0,
                         "hold_s": 0.0, "queue_proxy_s": 0.0, "queue_worker_s": 0.0, "hold_kv_block_s": 0.0, "evictions_caused": 0,
                         "evictions_to_admit": 0, "deadline_missed": False}])
    sessions = [{"session_id": "s", "class": "background", "tenant": "t", "t_start": 0.0, "t_end": 1800.0, "deadline": None, "missed": False, "turns": 1}]
    m = serving_metrics(log, sessions, slo_ttft_s=2.0, sim_duration_s=600.0, gpu_count=2, makespan_s=1800.0)
    assert m["tasks_per_hour"] == 2.0 and m["gpu_hours"] == 1.0

# I8: the forecast arm's index term conditions on the session's tool history
def test_forecast_arm_index_term_conditions_on_tool():
    rows = _duration_training_rows()
    from atfm.board.predictors import SurvivalPredictor
    pred = SurvivalPredictor().fit(TraceTable.from_rows(rows))
    pol = ForecastPolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=1.0), predictor=pred, train_table=None, horizons=[30.0], n=8, hold=False)
    from atfm.schema.events import parse_event
    pol.registry.apply(parse_event({"kind": "session.start", "t": 0.0, "session_id": "x", "tenant": "t", "class": "background"}))
    pol.registry.apply(parse_event({"kind": "tool.start", "t": 1.0, "session_id": "x", "turn_index": 0, "call_id": "c", "tool_name": "pytest", "backend_id": "ci"}))
    pol.registry.apply(parse_event({"kind": "tool.end", "t": 61.0, "session_id": "x", "call_id": "c", "exit_status": 0}))
    class _S: pass
    class _P: session_id = "x"; cls = "background"; tenant = "t"; parent = None
    s = _S(); s.program = _P(); s.deadline = None
    call = type("C", (), {"session": s, "turn_index": 1, "isl_total": 120, "osl": 10})()
    sim = type("Sim", (), {"now": 62.0})()
    assert abs(pol.e_tool_next(sim, call) - 60.0) < 1e-6        # conditioned on the pytest history, not the pooled mean (~30)


def _duration_training_rows():
    rows = []
    for k in range(10):
        t = k * 100.0
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=0, t_request=t, t_first_token=t + 1, t_last_token=t + 2,
                             isl=100, osl=10, tool_name="pytest", backend_id="ci", t_tool_start=t + 2, t_tool_end=t + 62, source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=1, t_request=t + 63, t_first_token=t + 64, t_last_token=t + 65,
                             isl=120, osl=10, tool_name="bash", backend_id="local", t_tool_start=t + 65, t_tool_end=t + 66, source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=2, t_request=t + 67, t_first_token=t + 68, t_last_token=t + 69,
                             isl=140, osl=10, tool_name=None, source="test"))
    return rows

# I9: no head-of-line skipping under priority
def test_priority_engine_does_not_skip_head_of_line():
    w = Worker("w0", EngineConfig(kv_blocks=20, max_batch=8, prefill_tps=1000.0, decode_tps=10.0, priority=True))
    w.submit(_req("hog", "h", 160), 0.0); w.schedule(0.0)                          # 11 running blocks
    w.submit(_req("big_it", "i", 160, tier=2, cls="interactive"), 1.0)               # needs 11, only 9 free
    w.submit(_req("small_bg", "s", 32, tier=0), 1.0)                                 # would fit
    out = w.schedule(1.0)
    assert out == [] and [r.request_id for r in w.queue] == ["big_it", "small_bg"]

# I10: overlay copies clone children with the copy suffix
def test_programs_from_table_overlay_clones_children():
    rows = []
    for sid, parent, t0 in (("r", None, 0.0), ("r/kid", "r", 3.0)):
        rows.append(TraceRow(session_id=sid, cls="background", tenant="t", turn_index=0, t_request=t0, t_first_token=t0 + 1, t_last_token=t0 + 2,
                             isl=100, osl=10, tool_name="bash", backend_id="local", t_tool_start=t0 + 2, t_tool_end=t0 + 10, parent_session_id=parent, source="test"))
        rows.append(TraceRow(session_id=sid, cls="background", tenant="t", turn_index=1, t_request=t0 + 10, t_first_token=t0 + 11, t_last_token=t0 + 12,
                             isl=120, osl=10, tool_name=None, parent_session_id=parent, source="test"))
    progs = programs_from_table(TraceTable.from_rows(rows), rate_per_hour=3600.0, duration_s=10.0, rng=np.random.default_rng(0))
    kids = [c.session_id for p in progs for _, c in p.spawn_at_turn]
    assert len(progs) >= 2 and len(kids) == len(set(kids)) and all("#" in k for k in kids)

# Minor 11: a shrinking context is a reset (full prefill), not a cache hit
def test_context_shrink_is_a_reset():
    rows = [TraceRow(session_id="a", cls="background", tenant="t", turn_index=0, t_request=0.0, t_first_token=1.0, t_last_token=2.0,
                     isl=1000, osl=50, tool_name="bash", backend_id="local", t_tool_start=2.0, t_tool_end=3.0, source="test"),
            TraceRow(session_id="a", cls="background", tenant="t", turn_index=1, t_request=4.0, t_first_token=5.0, t_last_token=6.0,
                     isl=400, osl=10, tool_name=None, source="test")]
    prog = programs_from_table(TraceTable.from_rows(rows), None, 100.0, np.random.default_rng(0))[0]
    assert prog.turns[1].reset and prog.turns[1].isl_new == 400
    eng = EngineConfig(kv_blocks=1000, max_batch=4, prefill_tps=1000.0, decode_tps=1e5)
    log = Simulator([prog], [eng], NativePolicy(), rng=np.random.default_rng(0)).run().sort_values("turn_index")
    assert log.iloc[1].isl == 400 and log.iloc[1].prefix_hit_tokens == 0 and log.iloc[1].recomputed_tokens == 400

# Minor 14: working-set hysteresis releases when sessions end and the watermark clears
def test_working_set_hysteresis_release():
    eng = EngineConfig(kv_blocks=100000, max_batch=8, prefill_tps=1000.0, decode_tps=1e6)
    def one(sid, t0, isl):
        return Program(session_id=sid, cls="background", tenant="t", t_arrival=t0, turns=[Turn(isl, 8, "bash", 3.0, "local"), Turn(16, 1, None, None)])
    progs = [one("big", 0.0, 1600), one("late", 0.5, 400)]
    pol = WorkingSetPolicy(budget_blocks=100, low_watermark=0.5)
    log = Simulator(progs, [eng], pol, harness_overhead_s=0.0, rng=np.random.default_rng(0)).run()
    late0 = log[(log.session_id == "late") & (log.turn_index == 0)].iloc[0]
    assert late0.hold_s > 0 and late0.hold_reason == "working_set"
