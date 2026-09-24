import numpy as np
from atfm.proxy.config import ProxyConfig
from atfm.traces.synthetic import WorkloadSpec, ClassSpec, ToolSpec
from atfm.sim.programs import programs_from_spec
from atfm.sim.engine import EngineConfig
from atfm.sim.core import Simulator
from atfm.sim.forecast_arm import ForecastPolicy, GdpLite, fit_predictor_on_programs
from atfm.schema.forecast import ForecastSnapshot

def _spec(seed=0):
    tools = [ToolSpec(name="bash", weight=0.6, log_mu=np.log(3.0), log_sigma=0.4, signal="none"),
             ToolSpec(name="pytest", weight=0.4, log_mu=np.log(60.0), log_sigma=0.4, signal="strong", backend_id="ci")]
    return WorkloadSpec(duration_s=900.0, seed=seed, classes=[
        ClassSpec(cls="background", rate_per_hour=240.0, turns_mean=5, isl0=2000, isl_growth=300, osl_mean=60, tools=tools),
        ClassSpec(cls="interactive", rate_per_hour=120.0, turns_mean=4, isl0=3000, isl_growth=400, osl_mean=40, tools=tools[:1],
                  think_log_mu=np.log(10.0), think_log_sigma=0.4, deadline_s=300.0)])

def test_gdp_lite_holds_when_forecast_exceeds_capacity():
    snap = ForecastSnapshot(t=0.0, horizons=[30.0], model_id="m",
                            samples={"kv_blocks": {"interactive": np.full((1, 8), 900.0), "background": np.zeros((1, 8))},
                                     "prefill_tokens": {"interactive": np.full((1, 8), 9000.0), "background": np.zeros((1, 8))}})
    g = GdpLite(slot_s=30.0, eps=0.1)
    assert g.hold_until(100.0, snap, free_blocks=500, free_slots=10, mean_isl=3000.0) == 130.0
    assert g.hold_until(100.0, snap, free_blocks=2000, free_slots=10, mean_isl=3000.0) is None
    assert g.hold_until(100.0, snap, free_blocks=2000, free_slots=1, mean_isl=3000.0) == 130.0

def test_forecast_arm_runs_and_holds_only_background():
    engines = [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=20000.0, decode_tps=40.0)]
    train = programs_from_spec(_spec(1), np.random.default_rng(1))
    pred, table = fit_predictor_on_programs("M2", train, engines, np.random.default_rng(1))
    assert len(table) > 50 and pred.name == "M2_progress"
    progs = programs_from_spec(_spec(0), np.random.default_rng(0))
    pol = ForecastPolicy(window=4, cfg=ProxyConfig(upstream_url="x", beta=0.5), predictor=pred, train_table=table,
                         horizons=[30.0, 120.0], n=64, hold=True)
    log = Simulator(progs, engines, pol, max_hold_s=60.0, rng=np.random.default_rng(0)).run()
    assert pol.name == "forecast_M2" and len(log) > 100
    held = log[log.hold_reason != ""]                                   # holds, not ordinary window queueing
    assert len(held) > 0 and (held["class"] == "background").all()
    assert (log.held_s <= 60.0 + 1e-6).all()
    assert pol.snapshots > 10
