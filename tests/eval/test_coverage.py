from atfm.schema.trace import TraceRow, TraceTable
from atfm.eval.coverage import signal_class, coverage_report

def _row(tool, prog, data=(), dur=100.0, i=0):
    return TraceRow(session_id=f"s{i}", cls="background", tenant="t", turn_index=0, t_request=0.0, t_first_token=1.0, t_last_token=2.0,
                    isl=10, osl=1, tool_name=tool, t_tool_start=2.0, t_tool_end=2.0 + dur,
                    progress_events=[{"t": 3.0 + j, "completed": j, "total": 10 if tot else None, "phase": "run"} for j, tot in prog],
                    data_events=[{"t": 3.0, "metric": m, "value": 1.0} for m in data], source="test")

def test_signal_class():
    assert signal_class(_row("pytest", [(1, True), (2, True)]).model_dump(by_alias=True)) == "strong"
    assert signal_class(_row("build", [(9, True)]).model_dump(by_alias=True)) == "weak"
    assert signal_class(_row("bash", [], data=("lines_per_s",)).model_dump(by_alias=True)) == "weak"
    assert signal_class(_row("bash", []).model_dump(by_alias=True)) == "none"

def test_coverage_report_time_shares():
    t = TraceTable.from_rows([_row("pytest", [(1, True), (2, True)], dur=300.0, i=0), _row("bash", [], dur=100.0, i=1),
                              _row("bash", [(1, True)], dur=100.0, i=2)])
    df = coverage_report(t).set_index("tool_name")
    assert abs(df.loc["ALL", "strong_time_share"] - 0.6) < 1e-9 and abs(df.loc["bash", "weak_time_share"] - 0.5) < 1e-9
    assert df.loc["ALL", "calls"] == 3
