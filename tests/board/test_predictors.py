import numpy as np
from atfm.schema.trace import TraceRow, TraceTable
from atfm.board.state import SessionState
from atfm.board.predictors import ConstantSeries, KalmanSeries, DurationModel, HistoryPredictor, SurvivalPredictor

def _train():
    rows = []
    t = 0.0
    for k in range(40):
        d = 10.0 + k  # pytest durations 10..49
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=0, t_request=t,
                             t_first_token=t + 1, t_last_token=t + 2, isl=100, osl=10, tool_name="pytest",
                             t_tool_start=t + 2, t_tool_end=t + 2 + d, source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=1, t_request=t + 2 + d + 1.0,
                             t_first_token=t + 4 + d, t_last_token=t + 5 + d, isl=150, osl=10, tool_name=None, source="test"))
        t += 1000.0
    return TraceTable.from_rows(rows)

def _state(elapsed, phase="tool_running"):
    return SessionState(session_id="x", cls="background", tenant="t", parent_session_id=None, phase=phase,
                        turn_index=0, tool_name="pytest", backend_id="ci", t_tool_start=100.0 - elapsed,
                        ctx_tokens=110, t_phase_start=100.0 - elapsed)

def test_series_constant_and_kalman():
    rng = np.random.default_rng(0)
    assert np.all(ConstantSeries().predict(np.array([3.0, 5.0]), 4, rng) == 5.0)
    assert np.all(ConstantSeries().predict(np.array([]), 4, rng) == 0.0)
    k = KalmanSeries()
    s = k.predict(np.arange(1.0, 51.0), 2000, rng)
    assert 45.0 < s.mean() < 56.0 and s.min() >= 0.0

def test_duration_model_conditional():
    dm = DurationModel().fit(_train())
    rng = np.random.default_rng(1)
    unc = dm.sample("pytest", 2000, rng)
    assert 25.0 < unc.mean() < 35.0
    cond = dm.sample_conditional("pytest", 40.0, 2000, rng)
    assert cond.min() > 40.0 and cond.max() <= 49.0 + 1e-9
    tail = dm.sample_conditional("pytest", 500.0, 200, rng)
    assert tail.min() > 500.0 and np.isfinite(tail).all()
    assert dm.no_return_prob("pytest") == 0.0
    assert abs(dm.overhead("pytest") - 1.0) < 1e-9
    assert dm.sample("unknown_tool", 10, rng).shape == (10,)

def test_b2_vs_m1():
    tr = _train()
    b2, m1 = HistoryPredictor(), SurvivalPredictor()
    b2.fit(tr); m1.fit(tr)
    rng = np.random.default_rng(2)
    s = _state(elapsed=44.0)
    r2 = b2.resumption(s, 100.0, 2000, rng)
    r1 = m1.resumption(s, 100.0, 2000, rng)
    assert (r2 <= 1.0 + 1e-9).mean() > 0.8
    assert r1.min() > 1.0 and r1.max() <= 6.0 + 1e-9
    assert np.all(m1.resumption(_state(0.0, phase="llm_pending"), 100.0, 5, rng) == 0.0)
    isl = m1.next_call_isl(s, 100, rng)
    assert isl.min() >= 110 and isl.dtype.kind == "i"

def test_llm_running_forecasts_next_call_not_current():
    # while the current call decodes, the next request comes after the remaining decode plus the next tool
    m1 = SurvivalPredictor().fit(_train())   # LLM time 2 s per call, pooled tool durations 10..49 s
    s = SessionState(session_id="x", cls="background", tenant="t", parent_session_id=None, phase="llm_running",
                     turn_index=0, tool_name=None, backend_id=None, ctx_tokens=110, t_phase_start=100.0)
    r = m1.resumption(s, 100.5, 2000, np.random.default_rng(0))
    assert (r > 0).all() and 15.0 < np.median(r) < 50.0
    pending = SessionState(session_id="y", cls="background", tenant="t", parent_session_id=None, phase="llm_pending",
                           turn_index=0, ctx_tokens=110, t_phase_start=100.0)
    assert np.all(m1.resumption(pending, 100.5, 5, np.random.default_rng(0)) == 0.0)
