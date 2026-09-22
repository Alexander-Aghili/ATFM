import numpy as np
from atfm.schema.trace import TraceRow, TraceTable
from atfm.board.state import SessionState
from atfm.board.predictors import SurvivalPredictor, ConstantSeries
from atfm.board.forecaster import SessionForecaster, SeriesForecaster, ExogenousModel

def _train():
    rows = []
    t = 0.0
    for k in range(20):
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=0, t_request=t,
                             t_first_token=t + 1, t_last_token=t + 2, isl=160, osl=16, tool_name="pytest",
                             t_tool_start=t + 2, t_tool_end=t + 62, source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=1, t_request=t + 62,
                             t_first_token=t + 63, t_last_token=t + 64, isl=320, osl=16, tool_name=None, source="test"))
        t += 500.0
    return TraceTable.from_rows(rows)

def _state(elapsed):
    return SessionState(session_id=f"x{elapsed}", cls="background", tenant="t", parent_session_id=None, phase="tool_running",
                        turn_index=0, tool_name="pytest", backend_id="ci", t_tool_start=100.0 - elapsed, ctx_tokens=176, t_phase_start=100.0 - elapsed)

def test_session_forecaster_shapes_and_empty():
    tr = _train()
    fc = SessionForecaster(SurvivalPredictor().fit(tr), ExogenousModel().fit(tr), horizons=[10.0, 120.0], n=64)
    snap = fc.forecast(100.0, [], np.random.default_rng(0))
    assert snap.samples["kv_blocks"]["background"].shape == (2, 64)
    assert np.all(snap.samples["kv_blocks"]["background"] == 0) and snap.endogenous_fraction["background"].tolist() == [0.0, 0.0]
    assert np.isfinite(snap.quantiles("kv_blocks", "background", 0.9)).all()

def test_session_forecaster_counts_due_sessions():
    tr = _train()
    fc = SessionForecaster(SurvivalPredictor().fit(tr), ExogenousModel().fit(tr), horizons=[10.0, 120.0], n=256)
    states = [_state(59.0), _state(59.5), _state(1.0)]
    snap = fc.forecast(100.0, states, np.random.default_rng(0))
    q50 = snap.quantiles("kv_blocks", "background", 0.5)
    assert 35 <= q50[0] <= 45 and 55 <= q50[1] <= 65
    assert snap.endogenous_fraction["background"][1] > 0.99

def test_series_forecaster():
    sf = SeriesForecaster(ConstantSeries(), horizons=[10.0], n=8)
    sf.observe(0.0, {"kv_blocks": {"interactive": np.array([5.0]), "background": np.array([0.0])},
                     "prefill_tokens": {"interactive": np.array([80.0]), "background": np.array([0.0])}})
    snap = sf.forecast(10.0, [], np.random.default_rng(0))
    assert np.all(snap.samples["kv_blocks"]["interactive"] == 5.0)

def test_series_forecaster_only_uses_realized_windows():
    sf = SeriesForecaster(ConstantSeries(), horizons=[10.0, 100.0], n=4)
    def truth(v):
        return {"kv_blocks": {"interactive": np.array([v, v]), "background": np.array([0.0, 0.0])},
                "prefill_tokens": {"interactive": np.array([v, v]), "background": np.array([0.0, 0.0])}}
    sf.observe(0.0, truth(1.0))
    sf.observe(50.0, truth(2.0))
    s60 = sf.forecast(60.0, [], np.random.default_rng(0)).samples["kv_blocks"]["interactive"]
    assert np.all(s60[0] == 2.0)   # h=10: window (50,60] realized
    assert np.all(s60[1] == 0.0)   # h=100: no realized window yet
    s100 = sf.forecast(100.0, [], np.random.default_rng(0)).samples["kv_blocks"]["interactive"]
    assert np.all(s100[1] == 1.0)  # h=100: only the window (0,100] is realized
