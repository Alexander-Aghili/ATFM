"""Serving metrics from the simulator's lifecycle log (spec section 12) and paired bootstrap across arms."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass(frozen=True)
class MeanMetric:
    """A row-wise mean that can be prepared once before session resampling.

    ``values`` returns one numeric value per input row, using NaN to exclude a
    row. The statistic is ``offset + scale * mean(nonmissing values)``; an
    empty selection yields NaN. Arbitrary DataFrame metrics remain supported
    by the bootstrap functions without this optimization.
    """

    values: Callable[[pd.DataFrame], pd.Series | np.ndarray]
    offset: float = 0.0
    scale: float = 1.0

    def __call__(self, frame: pd.DataFrame) -> float:
        return float(self.offset + self.scale * pd.Series(self.values(frame)).mean())


def _paired_draws(per_session: dict, metric_fn, n_boot: int, rng) -> tuple[dict, dict]:
    """Bootstrap identical session selections across arms without repeated label lookup."""
    arms = list(per_session)
    ids = sorted(set.intersection(*[set(df["session_id"]) for df in per_session.values()]))
    idx = {a: per_session[a].set_index("session_id").loc[ids] for a in arms}
    unique = all(frame.index.is_unique for frame in idx.values())
    values = None
    if isinstance(metric_fn, MeanMetric) and unique:
        values = {a: np.asarray(metric_fn.values(idx[a].reset_index()), dtype=float) for a in arms}
        if any(array.shape != (len(ids),) for array in values.values()):
            raise ValueError("MeanMetric must produce one value per session row")
    draws = {a: [] for a in arms}
    labels = np.asarray(ids)
    for _ in range(n_boot):
        positions = rng.choice(len(ids), size=len(ids), replace=True)
        for a in arms:
            if values is not None:
                sample = values[a][positions]
                sample = sample[~np.isnan(sample)]
                mean = float(sample.mean()) if len(sample) else float("nan")
                draws[a].append(metric_fn.offset + metric_fn.scale * mean)
            else:
                frame = idx[a].take(positions) if unique else idx[a].loc[labels[positions]]
                draws[a].append(metric_fn(frame.reset_index()))
    return draws, idx


def paired_contrasts(per_session: dict, metric_fn, pairs: list[tuple[str, str]], n_boot: int = 500, rng=None) -> pd.DataFrame:
    """Direct paired differences a - b for named arm pairs, from one set of session resamples."""
    rng = np.random.default_rng(0) if rng is None else rng
    draws, idx = _paired_draws(per_session, metric_fn, n_boot, rng)
    rows = []
    for a, b in pairs:
        diff = np.asarray(draws[a], float) - np.asarray(draws[b], float)
        rows.append({"contrast": f"{a} vs {b}", "a": a, "b": b,
                     "diff_mean": float(metric_fn(idx[a].reset_index()) - metric_fn(idx[b].reset_index())),
                     "diff_ci_lo": float(np.nanquantile(diff, 0.025)), "diff_ci_hi": float(np.nanquantile(diff, 0.975))})
    return pd.DataFrame(rows)


def paired_bootstrap(per_session: dict, metric_fn, n_boot: int = 500, rng=None) -> pd.DataFrame:
    """Resample the same session ids for every arm; report each arm's CI and its paired difference to the first arm."""
    rng = np.random.default_rng(0) if rng is None else rng
    arms = list(per_session)
    draws, idx = _paired_draws(per_session, metric_fn, n_boot, rng)
    base = np.asarray(draws[arms[0]], float)
    rows = []
    for a in arms:
        d = np.asarray(draws[a], float)
        diff = d - base
        rows.append({"arm": a, "mean": float(metric_fn(idx[a].reset_index())), "ci_lo": float(np.nanquantile(d, 0.025)),
                     "ci_hi": float(np.nanquantile(d, 0.975)), "diff_mean": float(np.nanmean(diff)),
                     "diff_ci_lo": float(np.nanquantile(diff, 0.025)), "diff_ci_hi": float(np.nanquantile(diff, 0.975))})
    return pd.DataFrame(rows)
