import pandas as pd
from atfm.schema.events import parse_event
from atfm.traces.sidecar import events_to_trace_table

def _e(**d):
    return parse_event(d)

def test_events_to_trace_table_pairs_calls_and_tools():
    ev = _paired_events()
    t = events_to_trace_table(ev)
    df = t.df
    a = df[df.session_id == "a"].sort_values("turn_index")
    assert len(a) == 2 and a.iloc[0]["tool_name"] == "pytest" and a.iloc[0]["t_tool_end"] == 42.2
    assert a.iloc[0]["progress_events"][0]["completed"] == 5 and a.iloc[0]["data_events"][0]["metric"] == "lines_per_s"
    assert a.iloc[0]["tool_exit_status"] == 1 and a.iloc[0]["osl"] == 50 and a.iloc[0]["class"] == "background"
    assert pd.isna(a.iloc[1]["tool_name"])   # pandas 3 str dtype stores missing as NaN
    b = df[df.session_id == "b"]
    assert len(b) == 1 and b.iloc[0]["isl"] == 1 and b.iloc[0]["t_tool_start"] == 100.0 and b.iloc[0]["source"] == "sidecar"


def _paired_events():
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
    return ev

def test_pre_call_tools_and_multiple_tools_per_turn_are_kept():
    ev = _multiple_tool_events()
    df = events_to_trace_table(ev).df
    assert sorted(df.tool_name.dropna()) == ["build", "clone", "install", "pytest"]
    assert (df[df.tool_name.isin(["install", "clone"])]["turn_index"] < 0).all()      # pre-call setup phases
    assert df[df.tool_name == "pytest"]["isl"].iloc[0] == 500 and df[df.tool_name == "build"]["isl"].iloc[0] == 1
    assert len(df) == 4 + 1    # two setup rows, the pytest call row, the extra build row, the final call row


def _multiple_tool_events():
    ev = [
        _e(kind="session.start", t=0.0, session_id="a", tenant="t1", **{"class": "background"}),
        _e(kind="tool.start", t=1.0, session_id="a", turn_index=0, call_id="s1", tool_name="install", backend_id="pkg"),
        _e(kind="tool.end", t=9.0, session_id="a", call_id="s1", exit_status=0),
        _e(kind="tool.start", t=9.5, session_id="a", turn_index=1, call_id="s2", tool_name="clone", backend_id="git"),
        _e(kind="tool.end", t=12.0, session_id="a", call_id="s2", exit_status=0),
        _e(kind="llm.request", t=13.0, session_id="a", turn_index=0, request_id="r1", isl=500),
        _e(kind="llm.done", t=14.0, session_id="a", request_id="r1", osl=5),
        _e(kind="tool.start", t=14.2, session_id="a", turn_index=2, call_id="c1", tool_name="pytest", backend_id="ci"),
        _e(kind="tool.end", t=30.0, session_id="a", call_id="c1", exit_status=0),
        _e(kind="tool.start", t=30.5, session_id="a", turn_index=3, call_id="c2", tool_name="build", backend_id="ci"),
        _e(kind="tool.end", t=40.0, session_id="a", call_id="c2", exit_status=0),
        _e(kind="llm.request", t=41.0, session_id="a", turn_index=1, request_id="r2", isl=600),
        _e(kind="llm.done", t=42.0, session_id="a", request_id="r2", osl=5),
    ]
    return ev
