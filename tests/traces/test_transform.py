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

def test_overlay_keeps_missing_parent_missing(tmp_path):
    import pandas as pd
    child = [r.model_copy(update={"session_id": "a/kid", "parent_session_id": "a"}) for r in _sess("a", 5100.0)]
    t = TraceTable.from_rows(_sess("a", 5000.0) + child)
    p = tmp_path / "t.parquet"
    t.to_parquet(p)                       # parquet + pandas 3 store the missing parents as NaN in a str column
    back = TraceTable.from_parquet(p)
    o = overlay_sessions(back, 100.0, 3600.0, seed=1)
    roots = o.df[~o.df.session_id.str.contains("/kid")]
    kids = o.df[o.df.session_id.str.contains("/kid")]
    assert roots["parent_session_id"].isna().all(), roots["parent_session_id"].unique()[:3]
    assert kids["parent_session_id"].str.endswith(tuple(f"#{k}" for k in range(1000))).all()

def _family(root, t0):
    kid = [r.model_copy(update={"session_id": f"{root}/kid", "parent_session_id": root, "t_request": r.t_request + 50.0,
                                "t_first_token": r.t_first_token + 50.0, "t_last_token": r.t_last_token + 50.0,
                                "t_tool_start": None if r.t_tool_start is None else r.t_tool_start + 50.0,
                                "t_tool_end": None if r.t_tool_end is None else r.t_tool_end + 50.0})
           for r in _sess(root, t0)]
    return _sess(root, t0) + kid

def test_split_by_family_keeps_children_with_roots():
    from atfm.traces.transform import split_by_session_families
    rows = []
    for k in range(20):
        rows += _family(f"s{k}", 0.0)          # all roots start at t=0, like AgentX
    t = TraceTable.from_rows(rows)
    tr, te = split_by_session_families(t, test_fraction=0.3, seed=1)
    fam = lambda df: set(df.session_id.str.split("/").str[0])
    assert fam(tr.df) & fam(te.df) == set() and 4 <= len(fam(te.df)) <= 8
    assert te.df.session_id.str.contains("/kid").any() and not te.df.session_id.str.contains("/kid").all()

def test_overlay_shifts_family_together():
    t = TraceTable.from_rows(_family("a", 1000.0))
    o = overlay_sessions(t, rate_per_hour=30.0, duration_s=3600.0, seed=5)
    fams = o.df.assign(fam=o.df.session_id.str.split("#").str[0].str.split("/").str[0], k=o.df.session_id.str.split("#").str[1])
    for (f, k), g in fams.groupby(["fam", "k"]):
        root = g[~g.session_id.str.contains("/")]; kid = g[g.session_id.str.contains("/")]
        assert len(root) and len(kid)
        assert abs((kid.t_request.min() - root.t_request.min()) - 50.0) < 1e-6      # child offset preserved
        assert (kid.parent_session_id == root.session_id.iloc[0]).all()
