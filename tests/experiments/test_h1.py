import numpy as np
from atfm_experiments.h1 import H1Config, run_h1
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
    from atfm_experiments.h1 import _ticks
    ticks = _ticks(t_min=0.0, t_max=1_000_000.0, window_s=1200.0, horizons=[30.0, 120.0], tick_s=60.0)
    assert ticks[0] == 0.0 and ticks[-1] <= 1200.0 - 120.0 and len(ticks) == 19

def test_tracelab_tables_fit_on_unoverlaid_train(tmp_path):
    from atfm_experiments.h1 import _tables
    from atfm.schema.trace import TraceRow, TraceTable
    rows = []
    for k in range(12):
        t0 = k * 3 * 86400.0
        rows.append(TraceRow(session_id=f"s{k}", cls="interactive", tenant="t", turn_index=0, t_request=t0,
                             t_first_token=t0 + 1, t_last_token=t0 + 2, isl=100, osl=10, tool_name="Bash",
                             t_tool_start=t0 + 2, t_tool_end=t0 + 12, source="tracelab"))
        rows.append(TraceRow(session_id=f"s{k}", cls="interactive", tenant="t", turn_index=1, t_request=t0 + 12,
                             t_first_token=t0 + 13, t_last_token=t0 + 14, isl=120, osl=10, tool_name=None, source="tracelab"))
    p = tmp_path / "tl.parquet"
    TraceTable.from_rows(rows).to_parquet(p)
    cfg = H1Config(name="x", source="tracelab", tracelab_parquet=str(p), overlay_rate_per_hour=50.0,
                   overlay_duration_s=3600.0, out_dir=str(tmp_path))
    train, test = _tables(cfg)
    assert not train.df["session_id"].str.contains("#").any()      # fitted on real sessions, not resampled copies
    assert test.df["session_id"].str.contains("#").all()            # replayed as a Poisson fleet

def test_exogenous_starts_exclude_children():
    from atfm_experiments.h1 import _session_starts
    import pandas as pd
    df = pd.DataFrame({"turn_index": [0, 0, 1], "t_request": [1.0, 2.0, 3.0], "class": ["background"] * 3,
                       "parent_session_id": [None, "p", None]})
    st = _session_starts(df)
    assert st["t_request"].tolist() == [1.0]

def test_h1_sidecar_source_runs(tmp_path):
    from atfm.bus import JsonlBus
    from atfm.schema.events import parse_event
    bus = JsonlBus(tmp_path / "ev.jsonl")
    t = 0.0
    for k in range(12):
        sid = f"s{k}"
        bus.publish(parse_event({"kind": "session.start", "t": t, "session_id": sid, "tenant": "t", "class": "background"}))
        bus.publish(parse_event({"kind": "tool.start", "t": t + 1, "session_id": sid, "turn_index": 0, "call_id": f"c{k}", "tool_name": "pytest", "backend_id": "ci"}))
        for j in range(1, 10):
            bus.publish(parse_event({"kind": "tool.progress", "t": t + 1 + 10 * j, "session_id": sid, "call_id": f"c{k}", "completed": 10 * j, "total": 100, "phase": "run"}))
        bus.publish(parse_event({"kind": "tool.end", "t": t + 101, "session_id": sid, "call_id": f"c{k}", "exit_status": 0}))
        t += 150.0
    cfg = H1Config(name="sc", source="sidecar", sidecar_events=str(tmp_path / "ev.jsonl"), block_seconds=600.0, test_fraction=0.5,
                   tick_s=30.0, horizons=[30.0, 120.0], n_samples=32, models=["M1", "M2"], out_dir=str(tmp_path))
    df = run_h1(cfg)
    assert set(df["model"]) == {"M1_survival", "M2_progress"} and (df["n"] > 3).all()

def test_h1_clear_errors_on_short_span_and_empty_train(tmp_path):
    import pytest
    from atfm.bus import JsonlBus
    from atfm.schema.events import parse_event
    bus = JsonlBus(tmp_path / "ev.jsonl")
    for k in range(4):   # four 20 s tools inside one 100 s window
        sid = f"s{k}"; t = 25.0 * k
        bus.publish(parse_event({"kind": "session.start", "t": t, "session_id": sid, "tenant": "t", "class": "background"}))
        bus.publish(parse_event({"kind": "tool.start", "t": t + 1, "session_id": sid, "turn_index": 0, "call_id": f"c{k}", "tool_name": "bash", "backend_id": "local"}))
        bus.publish(parse_event({"kind": "tool.end", "t": t + 21, "session_id": sid, "call_id": f"c{k}", "exit_status": 0}))
    base = dict(name="e", source="sidecar", sidecar_events=str(tmp_path / "ev.jsonl"), tick_s=10.0, n_samples=8, models=["M1"], out_dir=str(tmp_path))
    with pytest.raises(ValueError, match="train split is empty"):
        run_h1(H1Config(**base, block_seconds=600.0, test_fraction=0.5, horizons=[10.0]))
    with pytest.raises(ValueError, match="shorter than the longest horizon"):
        run_h1(H1Config(**base, block_seconds=30.0, test_fraction=0.5, horizons=[900.0]))


def test_h1_calibrate_wraps_session_models(tmp_path):
    tools = [ToolSpec(name="bash", weight=0.8, log_mu=np.log(3.0), log_sigma=0.5, signal="none"),
             ToolSpec(name="pytest", weight=0.2, log_mu=np.log(120.0), log_sigma=0.5, signal="strong", backend_id="ci")]
    spec = WorkloadSpec(duration_s=1200.0, seed=1, classes=[
        ClassSpec(cls="background", rate_per_hour=120.0, turns_mean=6, isl0=2000, isl_growth=400, osl_mean=100, tools=tools)])
    cfg = H1Config(name="cal", source="synthetic", synthetic=spec, tick_s=60.0, horizons=[30.0, 120.0], n_samples=32,
                   models=["B0", "M1"], calibrate=True, out_dir=str(tmp_path))
    df = run_h1(cfg)
    assert set(df["model"]) == {"B0_constant", "M1_survival+cal"}
    assert (tmp_path / "cal" / "calibration.json").exists()
    # calibration must not leak train-fleet state into the scored run: with k == 1 the calibrated model
    # scores exactly like the uncalibrated one on the same seed
    plain = run_h1(cfg.model_copy(update={"name": "plain", "calibrate": False}))
    import json
    k = json.load(open(tmp_path / "cal" / "calibration.json"))["inflation"]["M1"]
    if all(abs(x - 1.0) < 1e-9 for cls_k in k.values() for x in cls_k):
        a = df[df.model == "M1_survival+cal"].sort_values(["class", "h"])["pinball90"].to_numpy()
        b = plain[plain.model == "M1_survival"].sort_values(["class", "h"])["pinball90"].to_numpy()
        assert np.allclose(a, b)

def test_calibration_is_fitted_out_of_sample(tmp_path, monkeypatch):
    """The predictor used for calibration must be fitted on a strict subset of train (holdout > 0),
    and on all of train only in the in-sample diagnostic mode (holdout == 0)."""
    import atfm_experiments.h1 as h1mod
    from atfm.traces.synthetic import generate
    tools = [ToolSpec(name="pytest", weight=1.0, log_mu=np.log(60.0), log_sigma=0.3, signal="none", backend_id="ci")]
    spec = WorkloadSpec(duration_s=2400.0, seed=3, classes=[
        ClassSpec(cls="background", rate_per_hour=180.0, turns_mean=5, isl0=2000, isl_growth=200, osl_mean=50, tools=tools)])
    cfg = H1Config(name="oos", source="synthetic", synthetic=spec, tick_s=30.0, horizons=[30.0, 120.0], n_samples=32,
                   models=["M1"], calibrate=True, calibration_holdout=0.5, out_dir=str(tmp_path))
    train = generate(spec)
    seen = []
    real_build = h1mod._build
    monkeypatch.setattr(h1mod, "_build", lambda c, t: (seen.append(len(t)), real_build(c, t))[1])
    h1mod._calibrate(cfg, real_build(cfg, train), train)
    assert seen and 0 < seen[-1] < len(train)
    seen.clear()
    h1mod._calibrate(cfg.model_copy(update={"calibration_holdout": 0.0}), real_build(cfg, train), train)
    assert seen[-1] == len(train)

def test_calibration_overlays_the_holdout_like_the_test(tmp_path, monkeypatch):
    """With an overlay rate configured, the calibration part of train is replayed as a fleet at that rate."""
    import atfm_experiments.h1 as h1mod
    from atfm.schema.trace import TraceRow, TraceTable
    rows = []
    for k in range(16):
        t0 = k * 3 * 86400.0
        rows.append(TraceRow(session_id=f"s{k}", cls="interactive", tenant="t", turn_index=0, t_request=t0, t_first_token=t0 + 1,
                             t_last_token=t0 + 2, isl=100, osl=10, tool_name="Bash", t_tool_start=t0 + 2, t_tool_end=t0 + 40, source="tracelab"))
        rows.append(TraceRow(session_id=f"s{k}", cls="interactive", tenant="t", turn_index=1, t_request=t0 + 40, t_first_token=t0 + 41,
                             t_last_token=t0 + 42, isl=120, osl=10, tool_name=None, source="tracelab"))
    train = TraceTable.from_rows(rows)
    cfg = H1Config(name="ov", source="tracelab", tracelab_parquet="unused", overlay_rate_per_hour=120.0, overlay_duration_s=1800.0,
                   tick_s=30.0, horizons=[30.0], n_samples=16, models=["M1"], calibrate=True, block_seconds=7 * 86400.0, out_dir=str(tmp_path))
    calls = []
    real = h1mod.overlay_sessions
    monkeypatch.setattr(h1mod, "overlay_sessions", lambda t, r, d, s, **kw: (calls.append((len(t), r, d)), real(t, r, d, s, **kw))[1])
    h1mod._calibrate(cfg, h1mod._build(cfg, train), train)
    assert len(calls) == 1 and 0 < calls[0][0] < len(train) and calls[0][1] == 120.0 and calls[0][2] == 1800.0

def test_calibration_ticks_stay_inside_the_overlay_window(tmp_path, monkeypatch):
    import atfm_experiments.h1 as h1mod
    from atfm.schema.trace import TraceRow, TraceTable
    rows = []
    for k in range(16):                                      # sessions that run 10 hours after they start
        t0 = k * 3 * 86400.0
        rows.append(TraceRow(session_id=f"s{k}", cls="interactive", tenant="t", turn_index=0, t_request=t0, t_first_token=t0 + 1,
                             t_last_token=t0 + 2, isl=100, osl=10, tool_name="Bash", t_tool_start=t0 + 2, t_tool_end=t0 + 36000, source="tracelab"))
        rows.append(TraceRow(session_id=f"s{k}", cls="interactive", tenant="t", turn_index=1, t_request=t0 + 36000, t_first_token=t0 + 36001,
                             t_last_token=t0 + 36002, isl=120, osl=10, tool_name=None, source="tracelab"))
    train = TraceTable.from_rows(rows)
    cfg = H1Config(name="tk", source="tracelab", tracelab_parquet="unused", overlay_rate_per_hour=120.0, overlay_duration_s=1800.0,
                   tick_s=30.0, horizons=[30.0], n_samples=16, models=["M1"], calibrate=True, block_seconds=7 * 86400.0, out_dir=str(tmp_path))
    seen = {}
    real = h1mod.fit_inflation
    def spy(fc, ticks, **kw):
        seen["ticks"] = list(ticks); return real(fc, ticks, **kw)
    monkeypatch.setattr(h1mod, "fit_inflation", spy)
    h1mod._calibrate(cfg, h1mod._build(cfg, train), train)
    assert seen["ticks"] and max(seen["ticks"]) - min(seen["ticks"]) <= 1800.0


def test_calibration_updates_exogenous_rate_incrementally(tmp_path, monkeypatch):
    """During calibration the exogenous arrival rate must track the fleet (about the configured rate),
    not a burst of every early start reported at the first tick."""
    import atfm_experiments.h1 as h1mod
    from atfm.schema.trace import TraceRow, TraceTable
    train = _calibration_training()
    cfg = H1Config(name="rate", source="tracelab", tracelab_parquet="unused", overlay_rate_per_hour=360.0, overlay_duration_s=3600.0,
                   tick_s=30.0, horizons=[30.0], n_samples=16, models=["M1"], calibrate=True, block_seconds=7 * 86400.0, out_dir=str(tmp_path))
    captured = {}
    real = h1mod.fit_inflation
    def spy(fc, ticks, **kw):
        out = real(fc, ticks, **kw); captured["fc"] = fc; return out
    monkeypatch.setattr(h1mod, "fit_inflation", spy)
    h1mod._calibrate(cfg, h1mod._build(cfg, train), train)
    rate = captured["fc"].exo.rate("interactive")
    assert 0.3 * 360 / 3600 <= rate <= 2.0 * 360 / 3600, rate


def _calibration_training():
    from atfm.schema.trace import TraceRow, TraceTable
    rows = []
    for k in range(16):
        t0 = k * 3 * 86400.0
        rows.append(TraceRow(session_id=f"s{k}", cls="interactive", tenant="t", turn_index=0, t_request=t0, t_first_token=t0 + 1,
                             t_last_token=t0 + 2, isl=100, osl=10, tool_name="Bash", t_tool_start=t0 + 2, t_tool_end=t0 + 40, source="tracelab"))
        rows.append(TraceRow(session_id=f"s{k}", cls="interactive", tenant="t", turn_index=1, t_request=t0 + 40, t_first_token=t0 + 41,
                             t_last_token=t0 + 42, isl=120, osl=10, tool_name=None, source="tracelab"))
    train = TraceTable.from_rows(rows)
    return train
