import numpy as np
from atfm.schema.trace import TraceRow, TraceTable
from atfm.board.state import SessionState
from atfm.board.predictors import ProgressPredictor, BackendPredictor, BackendFactor
from atfm.board.predictors.progress import rate_posterior

def _train():
    rows = []
    t = 0.0
    for k in range(30):
        d = 100.0
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=0, t_request=t,
                             t_first_token=t + 1, t_last_token=t + 2, isl=100, osl=10, tool_name="pytest",
                             t_tool_start=t + 2, t_tool_end=t + 2 + d, backend_id="ci",
                             progress_events=[{"t": t + 2 + 10 * j, "completed": 10 * j, "total": 100, "phase": "run"} for j in range(1, 10)],
                             source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=1, t_request=t + 2 + d,
                             t_first_token=t + 3 + d, t_last_token=t + 4 + d, isl=150, osl=10, tool_name=None, source="test"))
        t += 1000.0
    return TraceTable.from_rows(rows)

def test_rate_posterior_concentrates():
    prog = [{"t": 10.0 * j, "completed": 5.0 * j, "total": 100.0, "phase": "run"} for j in range(1, 6)]
    shape, rate = rate_posterior(prog, t_start=0.0, now=50.0)
    assert abs(shape / rate - 0.5) < 0.1 and shape > 5

def test_m2_uses_progress_m1_fallback():
    tr = _train()
    m2 = ProgressPredictor().fit(tr)
    rng = np.random.default_rng(0)
    s = SessionState(session_id="x", cls="background", tenant="t", parent_session_id=None, phase="tool_running",
                     turn_index=0, tool_name="pytest", backend_id="ci", t_tool_start=0.0, ctx_tokens=110, t_phase_start=0.0,
                     progress=[{"t": 25.0, "completed": 10, "total": 100, "phase": "run"},
                               {"t": 50.0, "completed": 20, "total": 100, "phase": "run"}])
    r = m2.resumption(s, 52.0, 2000, rng)
    assert 120.0 < np.median(r) < 320.0
    s.progress = []
    r1 = m2.resumption(s, 52.0, 2000, rng)
    assert 40.0 < np.median(r1) < 60.0

def test_backend_factor_and_m3():
    bf = BackendFactor()
    assert np.all(bf.draw("nope", 3, np.random.default_rng(0)) == 1.0)
    for _ in range(10):
        bf.update("ci", 2.0)
    f = bf.draw("ci", 1000, np.random.default_rng(0))
    assert 1.7 < np.median(f) < 2.3
    m3 = BackendPredictor().fit(_train())
    for _ in range(10):
        m3.observe_completion("ci", "pytest", 200.0)
    s = SessionState(session_id="x", cls="background", tenant="t", parent_session_id=None, phase="tool_running",
                     turn_index=0, tool_name="pytest", backend_id="ci", t_tool_start=0.0, ctx_tokens=110, t_phase_start=0.0)
    r = m3.resumption(s, 10.0, 2000, np.random.default_rng(1))
    assert np.median(r) > 120.0
    shared = {"ci": np.full(5, 3.0)}
    r_shared = m3.resumption(s, 10.0, 5, np.random.default_rng(1), factors=shared)
    assert np.all(r_shared > 200.0)

def test_m2_residual_not_double_counted():
    # deterministic 100 s tools with progress every 10 s; observed at 55 s with 50% done -> ~45 s remain
    m2 = ProgressPredictor().fit(_train())
    s = SessionState(session_id="x", cls="background", tenant="t", parent_session_id=None, phase="tool_running",
                     turn_index=0, tool_name="pytest", backend_id="ci", t_tool_start=0.0, ctx_tokens=110, t_phase_start=0.0,
                     progress=[{"t": 10.0 * j, "completed": 10.0 * j, "total": 100, "phase": "run"} for j in range(1, 6)])
    r = m2.resumption(s, 55.0, 4000, np.random.default_rng(0))
    assert 40.0 < np.median(r) < 50.0

def _nonlinear_train(n=20, dur=100.0):
    """Tools whose reported progress runs ahead of time: 50% reported at 20% of the duration, 90% at 60%."""
    rows = []
    t = 0.0
    curve = [(0.1, 0.30), (0.2, 0.50), (0.4, 0.75), (0.6, 0.90), (0.8, 0.96)]   # (time fraction, progress fraction)
    for k in range(n):
        prog = [{"t": t + 2 + dur * tf, "completed": 100 * pf, "total": 100, "phase": "run"} for tf, pf in curve]
        rows.append(TraceRow(session_id=f"n{k}", cls="background", tenant="t", turn_index=0, t_request=t, t_first_token=t + 1,
                             t_last_token=t + 2, isl=100, osl=10, tool_name="build", backend_id="ci", t_tool_start=t + 2,
                             t_tool_end=t + 2 + dur, progress_events=prog, source="test"))
        rows.append(TraceRow(session_id=f"n{k}", cls="background", tenant="t", turn_index=1, t_request=t + 2 + dur,
                             t_first_token=t + 3 + dur, t_last_token=t + 4 + dur, isl=150, osl=10, tool_name=None, source="test"))
        t += 1000.0
    return TraceTable.from_rows(rows)

def test_progress_curve_maps_reported_progress_to_time_fraction():
    from atfm.board.predictors.progress import ProgressCurve
    pc = ProgressCurve().fit(_nonlinear_train())
    assert abs(pc.time_fraction("build", 0.50) - 0.20) < 0.05
    assert abs(pc.time_fraction("build", 0.90) - 0.60) < 0.05
    assert pc.time_fraction("unknown_tool", 0.5) == 0.5            # linear fallback
    assert pc.time_fraction("build", 0.0) == 0.0 and pc.time_fraction("build", 1.0) == 1.0

def test_m2_uses_progress_curve_for_nonlinear_tools():
    tr = _nonlinear_train()
    m2 = ProgressPredictor().fit(tr)
    # a 200 s build observed at 40 s reporting 50%: linear extrapolation says ~40 s remain; the curve says 50% = 20% of time -> ~160 s
    s = SessionState(session_id="x", cls="background", tenant="t", parent_session_id=None, phase="tool_running",
                     turn_index=0, tool_name="build", backend_id="ci", t_tool_start=0.0, ctx_tokens=110, t_phase_start=0.0,
                     progress=[{"t": 20.0, "completed": 30, "total": 100, "phase": "run"}, {"t": 40.0, "completed": 50, "total": 100, "phase": "run"}])
    r = m2.resumption(s, 40.0, 2000, np.random.default_rng(0))
    assert 120.0 < np.median(r) < 200.0
