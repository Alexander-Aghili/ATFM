"""Progress-signal coverage: how much tool time carries strong, weak or no live signal."""
from __future__ import annotations

import pandas as pd

from atfm.schema.trace import TraceTable


def signal_class(row: dict) -> str:
    prog = row.get("progress_events") or []
    data = row.get("data_events") or []
    with_total = [p for p in prog if p.get("total") not in (None, 0)]
    if len(with_total) >= 2:
        return "strong"
    if with_total or data or prog:
        return "weak"
    return "none"


def coverage_report(table: TraceTable) -> pd.DataFrame:
    df = table.df
    tools = df[df["tool_name"].notna() & (df["tool_name"] != "__think__")].copy()
    tools["dur"] = (tools["t_tool_end"] - tools["t_tool_start"]).clip(lower=0.0)
    tools["sig"] = tools.apply(lambda r: signal_class(r.to_dict()), axis=1) if len(tools) else pd.Series(dtype=object)
    rows = []
    groups = list(tools.groupby("tool_name")) + [("ALL", tools)]
    for name, g in groups:
        tot = float(g["dur"].sum())
        rows.append({"tool_name": name, "calls": int(len(g)), "tool_time_s": tot,
                     **{f"{s}_time_share": (float(g.loc[g["sig"] == s, "dur"].sum()) / tot if tot > 0 else 0.0)
                        for s in ("strong", "weak", "none")}})
    return pd.DataFrame(rows)


def coverage_markdown(df: pd.DataFrame) -> str:
    lines = ["| tool | calls | tool time (s) | strong | weak | none |", "|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.tool_name} | {r.calls} | {r.tool_time_s:.0f} | {r.strong_time_share:.0%} | "
                     f"{r.weak_time_share:.0%} | {r.none_time_share:.0%} |")
    return "\n".join(lines)
