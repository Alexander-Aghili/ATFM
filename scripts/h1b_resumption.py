"""H1b: per-session remaining-time scoring on collected long tools, leave-one-job-family-out.

    uv run python scripts/h1b_resumption.py runs/collect/l1_varied_events.jsonl [more.jsonl ...] --min-duration 30
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from atfm.bus.jsonl import read_events
from atfm.eval.resumption import leave_one_family_out
from atfm.traces.sidecar import events_to_trace_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="+")
    ap.add_argument("--min-duration", type=float, default=30.0)
    ap.add_argument("--offset", type=float, default=15.0, help="score every OFFSET seconds inside a phase")
    ap.add_argument("--max-offset", type=float, default=3600.0)
    ap.add_argument("--models", default="B2,M1,M2")
    ap.add_argument("--families", default="", help="comma list: keep only these job families (default all)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    events = [e for path in a.events for e in read_events(path)]
    table = events_to_trace_table(events)
    if a.families:
        from atfm.eval.resumption import job_family
        keep = set(a.families.split(","))
        table = type(table)(table.df[table.df["session_id"].map(job_family).isin(keep)].reset_index(drop=True))
    offsets = list(np.arange(a.offset, a.max_offset, a.offset))
    df = leave_one_family_out(table, a.models.split(","), offsets, a.min_duration, rng=np.random.default_rng(0))
    if df.empty:
        print("no phases longer than --min-duration"); return
    per = df.groupby(["family", "model"]).agg(pinball90=("pinball90", "mean"), crps=("crps", "mean"), n=("n", "sum")).reset_index()
    print(per.pivot(index="family", columns="model", values="pinball90").round(1).to_string())
    print("\nall phases (q90 pinball / CRPS, n per model):")
    print(df.groupby("model").agg(pinball90=("pinball90", "mean"), crps=("crps", "mean"), n=("n", "sum")).round(1).to_string())
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
