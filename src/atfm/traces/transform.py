from __future__ import annotations

import numpy as np
import pandas as pd

from atfm.schema.trace import TraceTable

TIME_COLS = ["t_request", "t_first_token", "t_last_token", "t_tool_start", "t_tool_end"]


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
    """Replay real sessions as a Poisson fleet: new start times, internal timing preserved."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate_per_hour * duration_s / 3600.0)
    starts = np.sort(rng.uniform(t0, t0 + duration_s, size=n))
    groups = {sid: g for sid, g in table.sessions()}
    sids = np.array(list(groups.keys()))
    picks = rng.choice(len(sids), size=n, replace=True)
    out = []
    for k, (start, idx) in enumerate(zip(starts, picks)):
        g = groups[sids[idx]].copy()
        delta = start - float(g["t_request"].iloc[0])
        for c in TIME_COLS:
            g[c] = g[c] + delta
        g["progress_events"] = g["progress_events"].apply(lambda ev: _shift_events(ev, delta))
        g["data_events"] = g["data_events"].apply(lambda ev: _shift_events(ev, delta))
        g["session_id"] = f"{sids[idx]}#{k}"
        g["parent_session_id"] = g["parent_session_id"].apply(lambda p: None if pd.isna(p) else f"{p}#{k}")
        out.append(g)
    if not out:
        return TraceTable(table.df.iloc[0:0].copy())
    return TraceTable(pd.concat(out, ignore_index=True))
