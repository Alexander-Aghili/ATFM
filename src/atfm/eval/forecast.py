from __future__ import annotations

import numpy as np
import pandas as pd

INF_SENTINEL = 1e12  # np.inf samples ("never resumes") are scored as a very large value


def crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """CRPS from samples: E|X - y| - 0.5 E|X - X'| along the last axis."""
    s = np.asarray(samples, float)
    s = np.where(np.isfinite(s), s, INF_SENTINEL)
    y = np.asarray(truth, float)[..., None]
    n = s.shape[-1]
    term1 = np.abs(s - y).mean(axis=-1)
    srt = np.sort(s, axis=-1)
    idx = np.arange(1, n + 1)
    term2 = (2.0 / (n * n)) * ((2 * idx - n - 1) * srt).sum(axis=-1)  # = E|X - X'|
    return term1 - 0.5 * term2


def pinball(samples: np.ndarray, truth: np.ndarray, q: float) -> np.ndarray:
    s = np.where(np.isfinite(samples), samples, INF_SENTINEL)
    pred = np.quantile(np.asarray(s, float), q, axis=-1)
    y = np.asarray(truth, float)
    diff = y - pred
    return np.where(diff >= 0, q * diff, (q - 1) * diff)


def coverage(samples: np.ndarray, truth: np.ndarray, level: float) -> np.ndarray:
    s = np.where(np.isfinite(samples), samples, INF_SENTINEL)
    lo = np.quantile(s, (1 - level) / 2, axis=-1)
    hi = np.quantile(s, 1 - (1 - level) / 2, axis=-1)
    y = np.asarray(truth, float)
    return (y >= lo) & (y <= hi)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def surge_events(q_series: np.ndarray, truth_series: np.ndarray, capacity: float, tick_s: float,
                 horizon_s: float = 300.0) -> dict:
    """Lead time of forecast capacity crossings ahead of true crossings; false alarms and misses."""
    fq = np.asarray(q_series) > capacity
    ft = np.asarray(truth_series) > capacity
    truth_runs = _runs(ft)
    fc_runs = _runs(fq)
    w = int(round(horizon_s / tick_s))
    lead, missed = [], 0
    for ts, _ in truth_runs:
        cands = [fs for (fs, _) in fc_runs if ts - w <= fs <= ts]
        if cands:
            lead.append(float((ts - min(cands)) * tick_s))
        else:
            missed += 1
    false_alarms = sum(1 for (fs, fe) in fc_runs
                       if not any(fs - w <= ts <= fe + w for (ts, _) in truth_runs))
    return {"lead_times": lead, "false_alarms": false_alarms, "missed": missed}


def score_run(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        s = r["samples"][None, :]
        y = np.array([r["truth"]])
        rows.append({"model": r["model"], "target": r["target"], "class": r["class"], "h": r["h"],
                     "perturbed": bool(r.get("perturbed", False)),
                     "crps": float(crps(s, y)[0]), "pinball90": float(pinball(s, y, 0.9)[0]),
                     "pinball95": float(pinball(s, y, 0.95)[0]),
                     "cov80": float(coverage(s, y, 0.8)[0]), "cov90": float(coverage(s, y, 0.9)[0]),
                     "endogenous_fraction": r.get("endogenous_fraction", np.nan)})
    df = pd.DataFrame(rows)
    keys = ["model", "target", "class", "h"]
    metrics = ["crps", "pinball90", "pinball95", "cov80", "cov90", "endogenous_fraction"]
    agg = df.groupby(keys)[metrics].mean()
    agg["n"] = df.groupby(keys).size()
    pert = df[df["perturbed"]].groupby(keys)[metrics[:5]].mean().add_suffix("_pert")
    return agg.join(pert, how="left").reset_index()


def score_tick(t: float, model: str, target: str, cls: str, horizons: list[float], samples: np.ndarray,
               truth: np.ndarray, perturbed: bool, endogenous_fraction: np.ndarray) -> pd.DataFrame:
    """Score one (tick, model, target, class) across horizons; keeps scalars only, never the samples."""
    return pd.DataFrame({
        "t": t, "model": model, "target": target, "class": cls, "h": list(horizons), "perturbed": perturbed,
        "crps": crps(samples, truth), "pinball90": pinball(samples, truth, 0.9),
        "pinball95": pinball(samples, truth, 0.95),
        "cov80": coverage(samples, truth, 0.8).astype(float), "cov90": coverage(samples, truth, 0.9).astype(float),
        "endogenous_fraction": np.asarray(endogenous_fraction, float),
    })


def aggregate_scores(frames: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    keys = ["model", "target", "class", "h"]
    metrics = ["crps", "pinball90", "pinball95", "cov80", "cov90", "endogenous_fraction"]
    agg = df.groupby(keys)[metrics].mean()
    agg["n"] = df.groupby(keys).size()
    pert = df[df["perturbed"]].groupby(keys)[metrics[:5]].mean().add_suffix("_pert")
    return agg.join(pert, how="left").reset_index()
