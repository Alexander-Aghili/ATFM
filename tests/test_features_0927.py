"""Features added 2026-09-27: command signatures for progress curves, interactive-long-tool regime,
same-rule oracle, configurable hold cap. Each test names the production change that makes it pass."""
import math

import numpy as np

from atfm.board.live import SessionRegistry
from atfm.board.predictors.progress import ProgressCurve, ProgressPredictor
from atfm.board.state import SessionState
from atfm.proxy.config import ProxyConfig
from atfm.schema.events import SessionStart, ToolStart
from atfm.schema.trace import TraceRow, TraceTable
from atfm.sidecar.core import command_signature
from atfm.sim.core import Simulator
from atfm.sim.engine import EngineConfig
from atfm.sim.programs import Program, Turn
from atfm.traces.sidecar import events_to_trace_table


# ---------------------------------------------------------------- F4: progress curves keyed by command signature
def test_command_signature_normalizes_sizes_but_keeps_attached_flags():
    a = command_signature("python /w/pipeline.py --rows 600 --rate 8 --signal strong")
    b = command_signature("python /w/pipeline.py --rows 900 --rate 5 --signal strong")
    assert a == b                                            # sizes are not part of the signature
    j1 = command_signature("cd /w/fmt && cmake -S . -B build && cmake --build build -j1")
    j2 = command_signature("cd /w/fmt && cmake -S . -B build && cmake --build build -j2")
    assert j1 != j2                                          # a digit attached to a flag is a mode, not a size
    assert command_signature("pytest -v 2>&1") == command_signature("pytest   -v 2>&1 ")
    assert len(a) <= 64


def _curve_rows(tool, sig, shape, n=3, t0=0.0):
    """`shape(progress_fraction) -> time_fraction` phases of 100 s with 9 reports."""
    rows = []
    for k in range(n):
        t = t0 + 1000.0 * k
        ev = [{"t": t + 2 + 100.0 * shape(p), "completed": int(100 * p), "total": 100, "phase": "run"} for p in np.linspace(0.1, 0.9, 9)]
        rows.append(TraceRow(session_id=f"{tool}-{sig}-{k}", cls="background", tenant="t", turn_index=0, t_request=t,
                             t_first_token=t + 1, t_last_token=t + 2, isl=100, osl=10, tool_name=tool, tool_args_hash=sig,
                             t_tool_start=t + 2, t_tool_end=t + 102, backend_id="ci", progress_events=ev, source="test"))
        rows.append(TraceRow(session_id=f"{tool}-{sig}-{k}", cls="background", tenant="t", turn_index=1, t_request=t + 103,
                             t_first_token=t + 104, t_last_token=t + 105, isl=150, osl=10, tool_name=None, source="test"))
    return rows


def test_progress_curve_prefers_signature_key_and_falls_back_to_tool_name():
    fast_first = lambda p: p ** 2          # 50% reported at 25% of the time
    slow_first = lambda p: math.sqrt(p)    # 50% reported at 71% of the time
    table = TraceTable.from_rows(_curve_rows("build", "sigA", fast_first) + _curve_rows("build", "sigB", slow_first, t0=10000.0))
    pc = ProgressCurve().fit(table)
    assert abs(pc.time_fraction("build", 0.5, key="sigA") - 0.25) < 0.08
    assert abs(pc.time_fraction("build", 0.5, key="sigB") - 0.71) < 0.08
    pooled = pc.time_fraction("build", 0.5, key="unknown-sig")        # falls back to the name-level curve
    assert 0.3 < pooled < 0.65 and pc.has("build", "unknown-sig") and not pc.has("nope", None)


def test_m2_uses_the_session_signature():
    table = TraceTable.from_rows(_curve_rows("build", "sigA", lambda p: p ** 2) + _curve_rows("build", "sigB", lambda p: math.sqrt(p), t0=10000.0))
    m2 = ProgressPredictor().fit(table)
    base = dict(session_id="x", cls="background", tenant="t", parent_session_id=None, phase="tool_running", turn_index=0,
                tool_name="build", backend_id="ci", t_tool_start=0.0, ctx_tokens=110, t_phase_start=0.0,
                progress=[{"t": 25.0, "completed": 50, "total": 100, "phase": "run"}])
    ra = m2.resumption(SessionState(**base, tool_args_hash="sigA"), 25.0, 2000, np.random.default_rng(0))
    rb = m2.resumption(SessionState(**base, tool_args_hash="sigB"), 25.0, 2000, np.random.default_rng(0))
    # sigA: 50% at 25 s means a 100 s tool -> ~75 s left; sigB: 50% at 25 s means a 35 s tool -> ~10 s left
    assert np.median(ra) > 2.5 * np.median(rb)


def test_registry_and_trace_adapter_carry_args_hash():
    evs = [SessionStart(t=0.0, session_id="s", tenant="t", cls="background"),
           ToolStart(t=1.0, session_id="s", turn_index=0, call_id="c", tool_name="pytest", backend_id="ci", args_hash="sigZ")]
    reg = SessionRegistry()
    for e in evs:
        reg.apply(e)
    assert reg._s["s"].tool_args_hash == "sigZ"
    table = events_to_trace_table(evs)
    assert list(table.df["tool_args_hash"]) == ["sigZ"]


# ---------------------------------------------------------------- F1: interactive-long-tool regime
def test_interactive_long_tool_regime_gives_interactive_sessions_long_signalled_tools():
    from atfm_experiments.h2sim import regime_spec
    spec = regime_spec("interactive_long_tool")
    it = [c for c in spec.classes if c.cls == "interactive"][0]
    strong = [t for t in it.tools if t.signal == "strong"]
    assert strong and strong[0].log_mu >= math.log(60.0) and sum(t.weight for t in strong) >= 0.3


# ---------------------------------------------------------------- F2: same-rule oracle
def _prog(sid, cls, t_arrival, isl, tool=None, dur=None):
    turns = [Turn(isl_new=isl, osl=10, tool_name=tool, tool_duration=dur, backend_id="local", progress=[], think=False)]
    return Program(session_id=sid, cls=cls, tenant="t", t_arrival=t_arrival, turns=turns, deadline_s=None, parent=None, spawn_at_turn={})


def test_oracle_rule_snapshot_is_true_first_call_demand_per_horizon():
    from atfm.sim.forecast_arm import OracleRulePolicy
    progs = [_prog("a", "interactive", 10.0, 1600), _prog("b", "interactive", 100.0, 3200), _prog("c", "background", 5.0, 800)]
    engines = [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]
    pol = OracleRulePolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=0.5), horizons=[30.0, 120.0], n=8)
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    sim.prime()
    pol.on_tick(sim, 0.0)
    kv = pol.snapshot.samples["kv_blocks"]["interactive"]
    assert kv.shape == (2, 8)
    assert np.allclose(kv[0], math.ceil(1600 / 16)) and np.allclose(kv[1], math.ceil(1600 / 16) + math.ceil(3200 / 16))
    assert np.allclose(pol.snapshot.samples["prefill_tokens"]["interactive"][1], 4800)
    assert np.allclose(pol.snapshot.samples["kv_blocks"]["background"][0], math.ceil(800 / 16))
    assert pol.name == "oracle_rule"


def test_oracle_rule_arm_is_registered_and_holds_only_background():
    from atfm_experiments.h2sim import ARMS, _arm, H2SimConfig
    assert "oracle_rule" in ARMS
    cfg = H2SimConfig(name="t", regime="short_tool", max_hold_s=60.0)
    engines = [EngineConfig(**e) for e in cfg.engines]
    pol = _arm("oracle_rule", cfg, engines, None, np.random.default_rng(0))
    assert pol.name == "oracle_rule" and pol.gdp.max_hold_s == 60.0


# ---------------------------------------------------------------- F3: configurable hold cap
def test_h2sim_config_hold_cap_reaches_the_simulator():
    from atfm_experiments.h2sim import H2SimConfig, build_simulator
    cfg = H2SimConfig(name="t", regime="short_tool", max_hold_s=60.0)
    engines = [EngineConfig(**e) for e in cfg.engines]
    sim = build_simulator(cfg, [_prog("a", "interactive", 1.0, 100)], engines, _arm_stub(), seed=0)
    assert sim.max_hold_s == 60.0 and sim.slo_ttft_s == cfg.slo_ttft_s


def _arm_stub():
    from atfm.sim.policies import NativePolicy
    return NativePolicy()


# ---------------------------------------------------------------- GDP-lite v2: capacity integrated over the slot
def test_gdp_lite_counts_capacity_that_frees_within_the_slot():
    """Under load free slots are ~0 at every instant, so an instantaneous test holds everything. The slot test
    must count requests that will finish inside the slot as capacity for the calls forecast to arrive in it."""
    from atfm.schema.forecast import ForecastSnapshot
    from atfm.sim.forecast_arm import GdpLite
    snap = ForecastSnapshot(t=0.0, horizons=[30.0], model_id="m",
                            samples={"kv_blocks": {"interactive": np.full((1, 8), 100.0), "background": np.zeros((1, 8))},
                                     "prefill_tokens": {"interactive": np.full((1, 8), 9000.0), "background": np.zeros((1, 8))}})
    g = GdpLite(slot_s=30.0, eps=0.1)
    # 3 calls x 10 s service = 1 busy slot over 30 s; nothing free now, but 2 requests finish inside the slot
    assert g.hold_until(100.0, snap, free_blocks=0, free_slots=0, mean_isl=3000.0, e_service_s=10.0) == 130.0
    assert g.hold_until(100.0, snap, free_blocks=0, free_slots=0, mean_isl=3000.0, e_service_s=10.0,
                        freeing_slots=2, freeing_blocks=200) is None
    # the freeing capacity is not enough for 3 busy slots worth of demand -> still hold
    assert g.hold_until(100.0, snap, free_blocks=0, free_slots=0, mean_isl=3000.0, e_service_s=30.0,
                        freeing_slots=2, freeing_blocks=200) == 130.0


def test_forecast_and_oracle_rule_pass_freeing_capacity():
    from atfm.sim.forecast_arm import ForecastPolicy, OracleRulePolicy, freeing_capacity
    engines = [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]
    progs = [_prog("a", "interactive", 1.0, 1600, tool="bash", dur=5.0)]
    sim = Simulator(progs, engines, _arm_stub(), rng=np.random.default_rng(0))
    sim.prime()
    w = sim.workers[0]
    from atfm.sim.engine import Request
    w.submit(Request(request_id="r", session_id="a", cls="interactive", isl_total=1600, isl_new=1600, osl=10, tier=1, index=0.0, t_queued=0.0), 0.0)
    w.schedule(0.0)
    slots, blocks = freeing_capacity(sim, now=0.0, slot_s=30.0, e_service_s=None)     # true t_end from the engine
    assert slots == 1 and blocks == w.resident_blocks("a") >= math.ceil(1600 / 16)
    slots2, _ = freeing_capacity(sim, now=0.0, slot_s=0.01, e_service_s=None)          # nothing frees in 10 ms
    assert slots2 == 0
    slots3, _ = freeing_capacity(sim, now=0.0, slot_s=30.0, e_service_s=5.0)           # estimated: t_queued + E[S]
    assert slots3 == 1
