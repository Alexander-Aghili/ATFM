"""Per-eviction diagnostic. Instruments a policy's eviction choices and scores each against the truth
after the run: the victim's true time to its next call, whether it was the true latest returner among the
candidates, and how long that next call then waited. Built to explain why forecast-ranked eviction beats
exact-return-time eviction; usable on any arm."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import Simulator
from .engine import EngineConfig


def eviction_records(programs, engines: list[EngineConfig], policy, seed: int = 0, returned_within_s: float = 60.0) -> pd.DataFrame:
    sim = Simulator(programs, engines, policy, rng=np.random.default_rng(seed))
    events: list[dict] = []
    has_order = hasattr(policy, "kv_victims")
    orig = policy.kv_victims if has_order else None

    def record(w, candidates, order):
        eta = getattr(policy, "_eta", {}) if has_order else {}
        now = sim.now
        for v in order[:1] if order else []:
            events.append({"t": now, "victim": v, "worker": w.worker_id, "n_candidates": len(candidates),
                           "blocks": w.resident_blocks(v), "cls": sim.sessions[v].program.cls if v in sim.sessions else "?",
                           "estimated_absence": (eta[v] - now) if (v in eta and np.isfinite(eta[v])) else np.nan,
                           "candidates": list(candidates),
                           "cand_cls": {c: (sim.sessions[c].program.cls if c in sim.sessions else "?") for c in candidates}})

    if has_order:
        def kv(sim_, w, cands):
            order = orig(sim_, w, cands)
            record(w, cands, order)
            return order
        policy.kv_victims = kv
        for w in sim.workers:
            w.victim_policy = (lambda cands, w=w: policy.kv_victims(sim, w, cands))
    else:
        for w in sim.workers:                       # LRU: the head of the resident order is the victim
            def lru_policy(cands, w=w):
                order = list(cands)
                record(w, cands, order)
                return order
            w.victim_policy = lru_policy
    log = sim.run()
    # the next *use* of a session's KV is when its next call starts on the worker (a call queued at the proxy
    # at eviction time has already arrived but has not used the KV yet), so truth is measured on t_start
    starts = log.groupby("session_id")["t_start"].apply(lambda s: np.sort(s.values))
    queue = log.set_index(["session_id", "t_start"])["queue_proxy_s"] if "queue_proxy_s" in log.columns else None
    rows = []
    for e in events:
        def next_call(sid):
            a = starts.get(sid)
            if a is None:
                return np.inf, None
            n = a[a > e["t"]]
            return (n[0] - e["t"], n[0]) if len(n) else (np.inf, None)
        true_abs, t_next = next_call(e["victim"])
        truths = {c: next_call(c)[0] for c in e["candidates"]}
        latest = max(truths.values()) if truths else np.inf
        q = float(queue.get((e["victim"], t_next), np.nan)) if (queue is not None and t_next is not None) else np.nan
        other = [truths[c] for c in e["candidates"] if e["cand_cls"].get(c) != e["cls"]]
        rows.append({"arm": getattr(policy, "name", type(policy).__name__), "t": e["t"], "victim": e["victim"], "cls": e["cls"],
                     "blocks": e["blocks"], "n_candidates": e["n_candidates"], "estimated_absence": e["estimated_absence"],
                     "true_absence": true_abs, "victim_was_true_latest": bool(true_abs >= latest - 1e-9),
                     "returned_within_60": bool(true_abs < returned_within_s), "next_call_queue_s": q,
                     "n_other_class_candidates": len(other),
                     "other_class_max_true_absence": (max(other) if other else np.nan),
                     "forced": bool(not other or max(other) < true_abs)})   # no other-class candidate would have been a later returner
    return pd.DataFrame(rows)


def summarize_evictions(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"evictions": 0}
    return {"evictions": int(len(df)), "victim_true_latest_share": float(df["victim_was_true_latest"].mean()),
            "returned_within_60_share": float(df["returned_within_60"].mean()),
            "mean_next_call_queue_s": float(np.nanmean(df["next_call_queue_s"])) if df["next_call_queue_s"].notna().any() else float("nan"),
            "never_returned_share": float(np.isinf(df["true_absence"]).mean()),
            "by_class": {c: int(n) for c, n in df["cls"].value_counts().items()}}
