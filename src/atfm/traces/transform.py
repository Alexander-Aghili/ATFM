from __future__ import annotations

import numpy as np
import pandas as pd

from atfm.schema.trace import TraceTable

TIME_COLS = ["t_request", "t_first_token", "t_last_token", "t_tool_start", "t_tool_end"]


def family_of(session_id: str) -> str:
    """Root session id: child sessions are named '<root>/<child>' by the adapters."""
    return session_id.split("/", 1)[0]


def split_by_session_families(table: TraceTable, test_fraction: float = 0.3, seed: int = 0) -> tuple[TraceTable, TraceTable]:
    """Hold out whole families (a root session with its children). Use when timestamps are not on a
    shared calendar (AgentX traces each start at t=0), where calendar blocks would split roots from children."""
    df = table.df
    fams = np.array(sorted(set(family_of(s) for s in df["session_id"])))
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(fams) * test_fraction)))
    test = set(rng.choice(fams, size=n_test, replace=False).tolist())
    is_test = df["session_id"].map(family_of).isin(test)
    return TraceTable(df[~is_test].copy()), TraceTable(df[is_test].copy())


def split_by_time_blocks(table: TraceTable, block_seconds: float = 7 * 86400, test_fraction: float = 0.3,
                         seed: int = 0) -> tuple[TraceTable, TraceTable]:
    """Assign each session to the calendar block of its first request; hold out whole blocks."""
    df = table.df
    first = df.groupby("session_id")["t_request"].min()
    t0 = float(first.min())
    block = ((first - t0) // block_seconds).astype(int)
    blocks = np.array(sorted(block.unique()))
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(blocks) * test_fraction)))
    test_blocks = set(rng.choice(blocks, size=n_test, replace=False).tolist())
    test_sessions = set(block[block.isin(test_blocks)].index)
    is_test = df["session_id"].isin(test_sessions)
    return TraceTable(df[~is_test].copy()), TraceTable(df[is_test].copy())


def _shift_events(events: list, delta: float) -> list:
    return [{**e, "t": e["t"] + delta} for e in events]


def overlay_sessions(table: TraceTable, rate_per_hour: float, duration_s: float, seed: int,
                     t0: float = 0.0) -> TraceTable:
    """Replay real session families as a Poisson fleet: a root and its children are shifted together so
    the parent's spawn phase still coincides with the child's activity; internal timing is preserved."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate_per_hour * duration_s / 3600.0)
    starts = np.sort(rng.uniform(t0, t0 + duration_s, size=n))
    df = table.df
    fam_col = df["session_id"].map(family_of)
    groups = {f: g for f, g in df.groupby(fam_col, sort=False)}
    fams = np.array(list(groups.keys()))
    picks = rng.choice(len(fams), size=n, replace=True)
    out = []
    for k, (start, idx) in enumerate(zip(starts, picks)):
        g = groups[fams[idx]].copy()
        root_t0 = float(g.loc[g["session_id"] == fams[idx], "t_request"].min()) if (g["session_id"] == fams[idx]).any() \
            else float(g["t_request"].min())
        delta = start - root_t0
        for c in TIME_COLS:
            g[c] = g[c] + delta
        g["progress_events"] = g["progress_events"].apply(lambda ev: _shift_events(ev, delta))
        g["data_events"] = g["data_events"].apply(lambda ev: _shift_events(ev, delta))
        g["session_id"] = g["session_id"].apply(lambda s: f"{s}#{k}")
        g["parent_session_id"] = g["parent_session_id"].apply(lambda p: None if pd.isna(p) else f"{p}#{k}")
        out.append(g)
    if not out:
        return TraceTable(table.df.iloc[0:0].copy())
    return TraceTable(pd.concat(out, ignore_index=True))
