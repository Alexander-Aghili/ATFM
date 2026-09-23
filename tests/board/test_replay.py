import numpy as np
from atfm.schema.trace import TraceRow, TraceTable
from atfm.board.replay import FleetReplayer

def _rows():
    a = [TraceRow(session_id="a", cls="interactive", tenant="t", turn_index=0, t_request=0.0, t_first_token=1.0,
                  t_last_token=2.0, isl=160, osl=16, tool_name="pytest", backend_id="ci", t_tool_start=2.0, t_tool_end=62.0, source="test"),
         TraceRow(session_id="a", cls="interactive", tenant="t", turn_index=1, t_request=63.0, t_first_token=64.0,
                  t_last_token=65.0, isl=320, osl=16, tool_name=None, source="test")]
    b = [TraceRow(session_id="b", cls="background", tenant="t", turn_index=0, t_request=30.0, t_first_token=31.0,
                  t_last_token=32.0, isl=800, osl=8, tool_name="build", backend_id="ci", t_tool_start=32.0, t_tool_end=200.0, source="test")]
    return TraceTable.from_rows(a + b)

def test_phases():
    rep = FleetReplayer(_rows())
    s = {x.session_id: x for x in rep.states_at(1.5)}
    assert s["a"].phase == "llm_running" and "b" not in s
    s = {x.session_id: x for x in rep.states_at(40.0)}
    assert s["a"].phase == "tool_running" and s["a"].tool_name == "pytest" and abs(s["a"].elapsed(40.0) - 38.0) < 1e-9
    assert s["b"].phase == "tool_running" and s["b"].ctx_tokens == 808
    s = {x.session_id: x for x in rep.states_at(62.5)}
    assert s["a"].phase == "llm_pending"
    s = {x.session_id: x for x in rep.states_at(64.0)}
    assert s["a"].tool_history == [("pytest", 60.0)] and s["a"].ctx_tokens == 336
    assert all(x.session_id != "a" for x in rep.states_at(70.0))
    assert all(x.session_id != "b" for x in rep.states_at(201.0))

def test_demand_truth():
    rep = FleetReplayer(_rows())
    d = rep.demand_truth(10.0, [30.0, 60.0], block_size=16)
    assert d["kv_blocks"]["background"].tolist() == [50, 50]
    assert d["endogenous_kv_blocks"]["background"].tolist() == [0, 0]
    assert d["kv_blocks"]["interactive"].tolist() == [0, 20]
    assert d["endogenous_kv_blocks"]["interactive"].tolist() == [0, 20]
    assert d["prefill_tokens"]["interactive"].tolist() == [0, 320]

def test_demand_truth_counts_first_call_per_session():
    # session c hops three times inside the window: it must count once (KV required on resumption)
    c = [TraceRow(session_id="c", cls="background", tenant="t", turn_index=i, t_request=20.0 + 5 * i,
                  t_first_token=21.0 + 5 * i, t_last_token=22.0 + 5 * i, isl=160, osl=1,
                  tool_name="bash" if i < 2 else None, t_tool_start=(22.0 + 5 * i) if i < 2 else None,
                  t_tool_end=(25.0 + 5 * i) if i < 2 else None, source="test") for i in range(3)]
    rep = FleetReplayer(TraceTable.from_rows(c))
    d = rep.demand_truth(10.0, [30.0], block_size=16)
    assert d["kv_blocks"]["background"].tolist() == [10]
    assert d["prefill_tokens"]["background"].tolist() == [160]
    d2 = rep.demand_truth(21.0, [30.0], block_size=16)   # active at t=21 -> endogenous, next call at 25
    assert d2["endogenous_kv_blocks"]["background"].tolist() == [10] and d2["kv_blocks"]["background"].tolist() == [10]

def test_one_shot_session_ends():
    t = TraceTable.from_rows([TraceRow(session_id="q", cls="interactive", tenant="t", turn_index=0, t_request=0.0,
                                       t_first_token=1.0, t_last_token=3.0, isl=16, osl=1, tool_name=None, source="test")])
    rep = FleetReplayer(t)
    assert rep.states_at(2.0)[0].phase == "llm_running" and rep.states_at(3.5) == []
