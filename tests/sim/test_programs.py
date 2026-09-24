import numpy as np
from atfm.traces.synthetic import WorkloadSpec, ClassSpec, ToolSpec
from atfm.schema.trace import TraceRow, TraceTable
from atfm.sim.programs import Program, Turn, programs_from_spec, programs_from_table

def _spec():
    tools = [ToolSpec(name="bash", weight=0.7, log_mu=np.log(3.0), log_sigma=0.4, signal="none"),
             ToolSpec(name="pytest", weight=0.3, log_mu=np.log(120.0), log_sigma=0.5, signal="strong", backend_id="ci", spawn_prob=0.5)]
    return WorkloadSpec(duration_s=1800.0, seed=0, classes=[
        ClassSpec(cls="background", rate_per_hour=120.0, turns_mean=6, isl0=2000, isl_growth=400, osl_mean=100, tools=tools),
        ClassSpec(cls="interactive", rate_per_hour=60.0, turns_mean=4, isl0=3000, isl_growth=500, osl_mean=80, tools=tools[:1],
                  think_log_mu=np.log(15.0), think_log_sigma=0.5, deadline_s=600.0)])

def test_programs_from_spec_are_paired_and_closed_loop():
    a = programs_from_spec(_spec(), np.random.default_rng(1))
    b = programs_from_spec(_spec(), np.random.default_rng(1))
    assert [p.t_arrival for p in a] == [p.t_arrival for p in b] and len(a) > 40
    assert all(0.0 <= p.t_arrival < 1800.0 for p in a)
    bg = [p for p in a if p.cls == "background"]
    assert any(p.spawn_at_turn for p in bg)
    p = bg[0]
    assert p.turns[0].isl_new == 2000 and p.turns[-1].tool_name is None
    strong = [t for q in bg for t in q.turns if t.tool_name == "pytest"]
    assert strong and all(len(t.progress) >= 1 and t.progress[-1][2] == 100.0 for t in strong)
    it = [p for p in a if p.cls == "interactive"][0]
    assert it.deadline_s == 600.0

def test_programs_from_table_isl_growth_and_gaps():
    rows = [TraceRow(session_id="a", cls="interactive", tenant="t", turn_index=0, t_request=100.0, t_first_token=101.0, t_last_token=103.0,
                     isl=1000, osl=50, tool_name="pytest", backend_id="ci", t_tool_start=103.0, t_tool_end=163.0,
                     progress_events=[{"t": 133.0, "completed": 50, "total": 100, "phase": "run"}], source="test"),
            TraceRow(session_id="a", cls="interactive", tenant="t", turn_index=1, t_request=163.5, t_first_token=164.0, t_last_token=166.0,
                     isl=1450, osl=30, tool_name="__gap__", backend_id="unknown", t_tool_start=166.0, t_tool_end=200.0, source="test"),
            TraceRow(session_id="a", cls="interactive", tenant="t", turn_index=2, t_request=200.0, t_first_token=201.0, t_last_token=202.0,
                     isl=1400, osl=10, tool_name=None, source="test")]
    progs = programs_from_table(TraceTable.from_rows(rows), rate_per_hour=None, duration_s=1000.0, rng=np.random.default_rng(0))
    assert len(progs) == 1 and progs[0].t_arrival == 100.0
    t0, t1, t2 = progs[0].turns
    assert t0.isl_new == 1000 and t0.tool_duration == 60.0 and t0.progress == [(30.0, 50.0, 100.0)]
    assert t1.isl_new == 400 and t1.think and t1.tool_duration == 34.0
    # context shrank (1450 -> 1400): a reset turn, the whole new context is prefilled fresh
    assert t2.reset and t2.isl_new == 1400 and t2.tool_name is None
    assert not t0.reset and not t1.reset
