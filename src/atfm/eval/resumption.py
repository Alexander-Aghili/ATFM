"""Per-session resumption evaluation (H1b): at fixed offsets inside every tool phase, score each
predictor's distribution of the remaining time against the true remaining time.

This is the right unit for comparing progress-aware (M2) with elapsed-time (M1) prediction: every
long tool phase yields many observation points, independent of fleet concurrency.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import re

from atfm.board.predictors import HistoryPredictor, ProgressPredictor, SurvivalPredictor
from atfm.board.state import SessionState
from atfm.eval.coverage import signal_class
from atfm.eval.forecast import crps, pinball
from atfm.schema.trace import TraceTable


def resumption_records(table: TraceTable, offsets_s: list[float], min_duration_s: float = 0.0) -> list[dict]:
    """One record per (tool phase, offset) with the session state as the board would see it then."""
    recs = []
    for sid, rows in table.session_records():
        history: list[tuple[str, float]] = []
        for r in rows:
            ts, te = r["t_tool_start"], r["t_tool_end"]
            if r["tool_name"] is None or ts is None or te is None or (isinstance(ts, float) and math.isnan(ts)) \
                    or (isinstance(te, float) and math.isnan(te)):
                continue
            dur = float(te - ts)
            if dur < min_duration_s:
                history.append((r["tool_name"], dur))
                continue
            _phase_records(sid, r, ts, dur, history, offsets_s, recs)
            history.append((r["tool_name"], dur))
    return recs


def _phase_records(sid, r, ts, dur, history, offsets_s, recs):
    sig = signal_class(r)
    for off in offsets_s:
        if off >= dur:
            continue
        now = float(ts) + off
        prog = [e for e in (r["progress_events"] or []) if e["t"] <= now]
        data = [e for e in (r["data_events"] or []) if e["t"] <= now]
        state = SessionState(session_id=sid, cls=r["class"], tenant=r["tenant"], parent_session_id=None,
                             phase="tool_running", turn_index=int(r["turn_index"]), tool_name=r["tool_name"],
                             backend_id=r["backend_id"], t_tool_start=float(ts), progress=prog, data=data,
                             ctx_tokens=int(r["isl"]) + int(r["osl"]), tool_history=list(history), t_phase_start=float(ts),
                             tool_args_hash=None if pd.isna(r.get("tool_args_hash")) else r.get("tool_args_hash"))
        recs.append({"session_id": sid, "tool_name": r["tool_name"], "signal": sig, "elapsed": off,
                     "duration": dur, "true_remaining": dur - off, "now": now, "state": state})


def score_resumption(predictors: dict, records: list[dict], n: int = 256, rng=None) -> pd.DataFrame:
    """CRPS and q90 pinball of each predictor's remaining-time samples per record (long format)."""
    rng = np.random.default_rng(0) if rng is None else rng
    rows = []
    for name, p in predictors.items():
        for r in records:
            s = p.resumption(r["state"], r["now"], n, rng)
            s = np.where(np.isfinite(s), s, 1e6)[None, :]
            y = np.array([r["true_remaining"]])
            rows.append({"model": name, "session_id": r["session_id"], "tool_name": r["tool_name"], "signal": r["signal"],
                         "elapsed": r["elapsed"], "true_remaining": r["true_remaining"],
                         "crps": float(crps(s, y)[0]), "pinball90": float(pinball(s, y, 0.9)[0]),
                         "q50": float(np.quantile(s, 0.5)), "n": 1})
    return pd.DataFrame(rows)


def summarize_resumption(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "tool_name", "signal"]
    out = df.groupby(keys).agg(crps=("crps", "mean"), pinball90=("pinball90", "mean"), n=("n", "sum")).reset_index()
    return out


_SID_SUFFIX = re.compile(r"-\d+-[0-9a-f]{6}$")


def job_family(session_id: str) -> str:
    """Collection driver ids are `<job name>-<repeat>-<hex6>`; the family is the job name."""
    return _SID_SUFFIX.sub("", session_id)


_PREDICTORS = {"B2": HistoryPredictor, "M1": SurvivalPredictor, "M2": ProgressPredictor}


def leave_one_family_out(table: TraceTable, models: list[str], offsets_s: list[float], min_duration_s: float,
                         n: int = 256, rng=None) -> pd.DataFrame:
    """Score each job family with predictors fit on every other family (same tool at other sizes stays in)."""
    rng = np.random.default_rng(0) if rng is None else rng
    df = table.df
    fam = df["session_id"].map(job_family)
    out = []
    for family in sorted(fam.unique()):
        train = TraceTable(df[fam != family].reset_index(drop=True))
        test = TraceTable(df[fam == family].reset_index(drop=True))
        recs = resumption_records(test, offsets_s, min_duration_s)
        if not recs:
            continue
        preds = {m: _PREDICTORS[m]().fit(train) for m in models}
        scored = score_resumption(preds, recs, n=n, rng=rng)
        scored["family"] = family
        out.append(scored)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
