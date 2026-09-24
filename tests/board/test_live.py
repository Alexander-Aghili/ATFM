import numpy as np
from atfm.schema.events import parse_event
from atfm.board.live import SessionRegistry, LiveBoard
from atfm.board.forecaster import SessionForecaster, ExogenousModel
from atfm.board.predictors import SurvivalPredictor
from atfm.schema.trace import TraceRow, TraceTable

def _ev(**d):
    return parse_event(d)

def test_registry_transitions_and_unknown_sessions():
    r = SessionRegistry()
    r.apply(_ev(kind="tool.progress", t=5.0, session_id="ghost", call_id="c0", completed=1, total=3))
    s = {x.session_id: x for x in r.states(5.0)}["ghost"]
    assert s.phase == "tool_running" and s.tool_name == "unknown" and s.progress[0]["completed"] == 1
    r.apply(_ev(kind="session.start", t=0.0, session_id="a", tenant="t", **{"class": "background"}))
    r.apply(_ev(kind="llm.request", t=1.0, session_id="a", turn_index=0, request_id="r1", isl=1000, predicted_osl=100))
    assert {x.session_id: x for x in r.states(1.5)}["a"].phase == "llm_running"
    r.apply(_ev(kind="llm.done", t=2.0, session_id="a", request_id="r1", osl=80))
    r.apply(_ev(kind="tool.start", t=2.1, session_id="a", turn_index=0, call_id="c1", tool_name="pytest", backend_id="ci"))
    r.apply(_ev(kind="tool.progress", t=12.1, session_id="a", call_id="c1", completed=10, total=100, phase="run"))
    s = {x.session_id: x for x in r.states(15.0)}["a"]
    assert s.phase == "tool_running" and abs(s.elapsed(15.0) - 12.9) < 1e-9 and s.progress[-1]["total"] == 100 and s.ctx_tokens == 1100
    r.apply(_ev(kind="tool.end", t=62.1, session_id="a", call_id="c1", exit_status=0))
    s = {x.session_id: x for x in r.states(63.0)}["a"]
    assert s.phase == "llm_pending" and s.tool_history == [("pytest", 60.0)]
    assert r.new_starts_since(0.0) == [(0.0, "background")]
    assert all(x.session_id != "ghost" for x in r.states(5.0 + 7201.0))

def _train():
    rows = []
    for k in range(10):
        t = k * 500.0
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=0, t_request=t, t_first_token=t + 1,
                             t_last_token=t + 2, isl=160, osl=16, tool_name="pytest", t_tool_start=t + 2, t_tool_end=t + 62, source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=1, t_request=t + 62, t_first_token=t + 63,
                             t_last_token=t + 64, isl=320, osl=16, tool_name=None, source="test"))
    return TraceTable.from_rows(rows)

def test_live_board_step_and_predictions():
    tr = _train()
    pred = SurvivalPredictor().fit(tr)
    board = LiveBoard(SessionRegistry(), SessionForecaster(pred, ExogenousModel().fit(tr), horizons=[10.0, 120.0], n=32))
    board.registry.apply(_ev(kind="session.start", t=0.0, session_id="a", tenant="t", **{"class": "background"}))
    board.registry.apply(_ev(kind="tool.start", t=1.0, session_id="a", turn_index=0, call_id="c", tool_name="pytest", backend_id="ci"))
    snap = board.step(59.0, np.random.default_rng(0))
    assert snap.samples["kv_blocks"]["background"].shape == (2, 32)
    assert abs(board.expected_tool_next("a") - 60.0) < 1e-6 and board.expected_service("a", 20000, 60) == 2.0
