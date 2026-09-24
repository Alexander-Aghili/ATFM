"""Serving metrics from the simulator's lifecycle log (spec section 12) and paired bootstrap across arms."""
from __future__ import annotations

import numpy as np
import pandas as pd


def serving_metrics(log: pd.DataFrame, sessions: list[dict], slo_ttft_s: float, sim_duration_s: float, gpu_count: int,
                    makespan_s: float | None = None) -> dict:
    """`makespan_s` (realized end of the run) drives throughput and GPU hours; `sim_duration_s` is the arrival window."""
    span = float(makespan_s) if makespan_s else float(sim_duration_s)
    hold_col = "hold_s" if "hold_s" in log.columns else "held_s"
    it = log[(log["class"] == "interactive") & (log["turn_index"] > 0)]
    ttft = (it["t_first_token"] - it["t_arrival"]).to_numpy(float) if len(it) else np.array([np.nan])
    sess = pd.DataFrame(sessions)
    bg = sess[sess["class"] == "background"] if len(sess) else sess
    bg_jct = (bg["t_end"] - bg["t_start"]).to_numpy(float) if len(bg) else np.array([np.nan])
    with_dl = sess[sess["deadline"].notna()] if len(sess) else sess
    wait_proxy, wait_worker = float(log["queue_proxy_s"].sum()), float(log["queue_worker_s"].sum())
    tot_wait = wait_proxy + wait_worker
    bg_rows = log[log["class"] == "background"]
    return {
        "ttft_after_tool_p50": float(np.nanpercentile(ttft, 50)), "ttft_after_tool_p95": float(np.nanpercentile(ttft, 95)),
        "ttft_after_tool_p99": float(np.nanpercentile(ttft, 99)),
        "slo_attainment_calls": float(np.mean(ttft <= slo_ttft_s)) if len(it) else float("nan"),
        "bg_jct_mean": float(np.nanmean(bg_jct)), "bg_jct_p95": float(np.nanpercentile(bg_jct, 95)),
        "deadline_hit_rate": float(1.0 - with_dl["missed"].mean()) if len(with_dl) else float("nan"),
        "tasks_per_hour": float(len(sess) / (span / 3600.0)),
        "max_imposed_delay_by_tenant": {t: float(v) for t, v in log.groupby("tenant")[hold_col].max().items()},
        "mean_held_s_background": float(bg_rows[hold_col].mean()) if len(bg_rows) else 0.0,
        "gpu_hours": float(gpu_count * span / 3600.0),
        "recomputed_prefill_tokens": int(log["recomputed_tokens"].sum()),
        "kv_hit_rate": float(log["prefix_hit_tokens"].sum() / max(1, log["isl"].sum())),
        "hold_kv_block_s": float(log.groupby("session_id")["hold_kv_block_s"].max().sum()),
        "evictions_caused_by_holds": int(log.groupby("session_id")["evictions_caused"].max().sum()),
        "evictions_to_admit": int(log["evictions_to_admit"].sum()) if "evictions_to_admit" in log.columns else 0,
        "queue_proxy_share": float(wait_proxy / tot_wait) if tot_wait > 0 else 0.0,
        "queue_worker_share": float(wait_worker / tot_wait) if tot_wait > 0 else 0.0,
        "sessions_completed": int(len(sess)),
    }


def paired_bootstrap(per_session: dict, metric_fn, n_boot: int = 500, rng=None) -> pd.DataFrame:
    """Resample the same session ids for every arm; report each arm's CI and its paired difference to the first arm."""
    rng = np.random.default_rng(0) if rng is None else rng
    arms = list(per_session)
    ids = sorted(set.intersection(*[set(df["session_id"]) for df in per_session.values()]))
    idx = {a: per_session[a].set_index("session_id").loc[ids] for a in arms}
    draws = {a: [] for a in arms}
    for _ in range(n_boot):
        sample = rng.choice(ids, size=len(ids), replace=True)
        for a in arms:
            draws[a].append(metric_fn(idx[a].loc[sample].reset_index()))
    base = np.asarray(draws[arms[0]], float)
    rows = []
    for a in arms:
        d = np.asarray(draws[a], float)
        diff = d - base
        rows.append({"arm": a, "mean": float(metric_fn(idx[a].reset_index())), "ci_lo": float(np.nanquantile(d, 0.025)),
                     "ci_hi": float(np.nanquantile(d, 0.975)), "diff_mean": float(np.nanmean(diff)),
                     "diff_ci_lo": float(np.nanquantile(diff, 0.025)), "diff_ci_hi": float(np.nanquantile(diff, 0.975))})
    return pd.DataFrame(rows)
