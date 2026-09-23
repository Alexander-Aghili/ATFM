import numpy as np
import pandas as pd
from atfm.schema.trace import TraceRow, TraceTable, ProgressEvent, TRACE_COLUMNS
from atfm.schema.forecast import ForecastSnapshot

def _row(**kw):
    base = dict(session_id="s1", cls="interactive", tenant="t1", turn_index=0,
                t_request=100.0, t_first_token=101.0, t_last_token=102.0,
                isl=1000, osl=50, source="test")
    base.update(kw)
    return TraceRow(**base)

def test_row_alias_class():
    r = TraceRow(**{"session_id": "s", "class": "background", "tenant": "t", "turn_index": 0,
                    "t_request": 1.0, "isl": 16, "osl": 1, "source": "test"})
    assert r.cls == "background"
    assert r.model_dump(by_alias=True)["class"] == "background"

def test_table_roundtrip(tmp_path):
    rows = [_row(), _row(turn_index=1, t_request=200.0, tool_name="pytest",
                      t_tool_start=102.5, t_tool_end=199.0,
                      progress_events=[ProgressEvent(t=150.0, completed=10, total=20, phase="run")])]
    t = TraceTable.from_rows(rows)
    assert list(t.df.columns) == TRACE_COLUMNS
    p = tmp_path / "t.parquet"
    t.to_parquet(p)
    t2 = TraceTable.from_parquet(p)
    assert len(t2.df) == 2
    assert t2.df.loc[1, "progress_events"][0]["completed"] == 10
    assert t2.time_range() == (100.0, 200.0)

def test_kv_blocks():
    t = TraceTable.from_rows([_row(isl=17), _row(isl=16, turn_index=1, t_request=101.0)])
    assert t.kv_blocks(16).tolist() == [2, 1]

def test_sessions_sorted():
    t = TraceTable.from_rows([_row(t_request=300.0, turn_index=1), _row(t_request=100.0)])
    sid, df = next(t.sessions())
    assert sid == "s1" and df["t_request"].tolist() == [100.0, 300.0]

def test_snapshot_quantiles():
    s = ForecastSnapshot(t=0.0, horizons=[10.0, 30.0], model_id="m",
                         samples={"kv_blocks": {"interactive": np.array([[1, 2, 3, 4], [10, 20, 30, 40]], float)}},
                         endogenous_fraction={"interactive": np.array([1.0, 0.5])})
    q = s.quantiles("kv_blocks", "interactive", 0.5)
    assert q.shape == (2,) and q[0] == 2.5
