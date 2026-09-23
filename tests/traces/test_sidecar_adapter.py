import pandas as pd
from atfm.schema.events import parse_event
from atfm.traces.sidecar import events_to_trace_table

def _e(**d):
    return parse_event(d)

def test_events_to_trace_table_pairs_calls_and_tools():
    ev = [
        _e(kind="session.start", t=0.0, session_id="a", tenant="t1", **{"class": "background"}),
        _e(kind="llm.request", t=1.0, session_id="a", turn_index=0, request_id="r1", isl=1000),
        _e(kind="llm.first_token", t=1.5, session_id="a", request_id="r1"),
        _e(kind="llm.done", t=2.0, session_id="a", request_id="r1", osl=50),
        _e(kind="tool.start", t=2.2, session_id="a", turn_index=0, call_id="c1", tool_name="pytest", backend_id="ci"),
        _e(kind="tool.progress", t=12.2, session_id="a", call_id="c1", completed=5, total=20, phase="run"),
        _e(kind="tool.data", t=15.0, session_id="a", call_id="c1", metric="lines_per_s", value=3.0),
        _e(kind="tool.end", t=42.2, session_id="a", call_id="c1", exit_status=1),
        _e(kind="llm.request", t=42.5, session_id="a", turn_index=1, request_id="r2", isl=1200),
        _e(kind="llm.done", t=44.0, session_id="a", request_id="r2", osl=10),
        _e(kind="tool.start", t=100.0, session_id="b", turn_index=0, call_id="c9", tool_name="build", backend_id="ci"),
        _e(kind="tool.end", t=160.0, session_id="b", call_id="c9", exit_status=0),
    ]
    t = events_to_trace_table(ev)
    df = t.df
    a = df[df.session_id == "a"].sort_values("turn_index")
    assert len(a) == 2 and a.iloc[0]["tool_name"] == "pytest" and a.iloc[0]["t_tool_end"] == 42.2
    assert a.iloc[0]["progress_events"][0]["completed"] == 5 and a.iloc[0]["data_events"][0]["metric"] == "lines_per_s"
    assert a.iloc[0]["tool_exit_status"] == 1 and a.iloc[0]["osl"] == 50 and a.iloc[0]["class"] == "background"
    assert pd.isna(a.iloc[1]["tool_name"])   # pandas 3 str dtype stores missing as NaN
    b = df[df.session_id == "b"]
    assert len(b) == 1 and b.iloc[0]["isl"] == 1 and b.iloc[0]["t_tool_start"] == 100.0 and b.iloc[0]["source"] == "sidecar"
