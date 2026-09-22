import numpy as np
from atfm.schema.trace import TraceRow, TraceTable
from atfm.traces.transform import split_by_time_blocks, overlay_sessions

def _sess(sid, t0, n=3, gap=100.0):
    rows = []
    for i in range(n):
        rows.append(TraceRow(session_id=sid, cls="interactive", tenant="t", turn_index=i,
                             t_request=t0 + i * gap, t_first_token=t0 + i * gap + 1, t_last_token=t0 + i * gap + 2,
                             isl=100, osl=10, tool_name="Bash" if i < n - 1 else None,
                             t_tool_start=(t0 + i * gap + 2) if i < n - 1 else None,
                             t_tool_end=(t0 + (i + 1) * gap) if i < n - 1 else None, source="test"))
    return rows

def test_split_keeps_sessions_whole():
    rows = []
    for k in range(20):
        rows += _sess(f"s{k}", t0=k * 86400.0 * 2)
    t = TraceTable.from_rows(rows)
    tr, te = split_by_time_blocks(t, block_seconds=7 * 86400, test_fraction=0.3, seed=1)
    assert set(tr.df.session_id) & set(te.df.session_id) == set()
    assert len(tr) + len(te) == len(t) and len(te) > 0

def test_overlay_preserves_internal_timing():
    t = TraceTable.from_rows(_sess("a", 5000.0) + _sess("b", 9000.0, gap=50.0))
    o = overlay_sessions(t, rate_per_hour=600.0, duration_s=3600.0, seed=3)
    assert o.df.session_id.nunique() > 50
    for sid, g in o.sessions():
        d = np.diff(g["t_request"].to_numpy())
        assert np.allclose(d, 100.0) or np.allclose(d, 50.0)
        assert g["t_request"].iloc[0] >= 0.0 and g["t_request"].iloc[0] < 3600.0
        assert g["t_tool_end"].iloc[0] - g["t_tool_start"].iloc[0] > 0

def test_overlay_deterministic():
    t = TraceTable.from_rows(_sess("a", 5000.0))
    a = overlay_sessions(t, 100.0, 3600.0, seed=7).df
    b = overlay_sessions(t, 100.0, 3600.0, seed=7).df
    assert a.equals(b)
