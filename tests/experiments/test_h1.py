import numpy as np
from atfm.experiments.h1 import H1Config, run_h1
from atfm.traces.synthetic import WorkloadSpec, ClassSpec, ToolSpec

def test_run_h1_synthetic_smoke(tmp_path):
    tools = [ToolSpec(name="bash", weight=0.8, log_mu=np.log(3.0), log_sigma=0.5, signal="none"),
             ToolSpec(name="pytest", weight=0.2, log_mu=np.log(120.0), log_sigma=0.5, signal="strong", backend_id="ci")]
    spec = WorkloadSpec(duration_s=1200.0, seed=1, classes=[
        ClassSpec(cls="background", rate_per_hour=120.0, turns_mean=6, isl0=2000, isl_growth=400, osl_mean=100, tools=tools)])
    cfg = H1Config(name="smoke", source="synthetic", synthetic=spec, overlay_duration_s=1200.0, tick_s=60.0,
                   horizons=[30.0, 120.0], n_samples=32, models=["B0", "M1", "M2"], out_dir=str(tmp_path))
    df = run_h1(cfg)
    assert set(df["model"]) == {"B0_constant", "M1_survival", "M2_progress"}
    assert (tmp_path / "smoke" / "metrics.csv").exists() and (tmp_path / "smoke" / "surge.json").exists()
    assert df["pinball90"].notna().all() and (df["n"] > 5).all()

def test_ticks_bounded_by_fleet_window():
    from atfm.experiments.h1 import _ticks
    ticks = _ticks(t_min=0.0, t_max=1_000_000.0, window_s=1200.0, horizons=[30.0, 120.0], tick_s=60.0)
    assert ticks[0] == 0.0 and ticks[-1] <= 1200.0 - 120.0 and len(ticks) == 19
