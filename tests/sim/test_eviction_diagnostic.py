"""Per-eviction diagnostic (the tool for the open oracle-gap question): for every eviction under an arm,
record the victim, its class and blocks, the policy's estimated absence, the true time to its next call,
and whether that next call landed in a contended window. A tool, not an experiment; the test runs it on a
tiny workload."""
import numpy as np

from atfm.sim.diagnostics import eviction_records, summarize_evictions
from atfm.sim.engine import EngineConfig
from tests.sim.test_touch_arm import _prog


def _progs():
    return [_prog(f"s{i}", "background" if i % 2 else "interactive", 5.0 * i, 800, "bash", 40.0, turns=2) for i in range(6)]


def test_eviction_records_have_truth_and_estimate_per_victim():
    from atfm.experiments.h2sim import H2SimConfig, _arm
    cfg = H2SimConfig(name="d", regime="short_tool", engines=[{"kv_blocks": 200, "max_batch": 4, "prefill_tps": 20000.0, "decode_tps": 40.0}], window=4)
    engines = [EngineConfig(**e) for e in cfg.engines]
    pol = _arm("oracle_kv", cfg, engines, None, np.random.default_rng(0))
    df = eviction_records(_progs(), engines, pol, seed=0)
    assert len(df) > 0
    for col in ("t", "victim", "cls", "blocks", "estimated_absence", "true_absence", "returned_within_60", "victim_was_true_latest",
                "next_call_queue_s", "n_candidates"):
        assert col in df.columns
    assert (df["true_absence"] >= 0).all() and df["cls"].isin(["interactive", "background"]).all()
    err = (df["estimated_absence"] - df["true_absence"]).abs()[np.isfinite(df["estimated_absence"]) & np.isfinite(df["true_absence"])]
    assert err.median() < 1.0                                                            # oracle: exact up to the harness overhead and tick staleness
    s = summarize_evictions(df)
    assert set(s) >= {"evictions", "victim_true_latest_share", "returned_within_60_share", "mean_next_call_queue_s", "by_class"}


def test_diagnostic_works_for_lru_and_forecast_arms_too():
    from atfm.experiments.h2sim import H2SimConfig, _arm, regime_spec
    from atfm.sim.programs import programs_from_spec
    cfg = H2SimConfig(name="d", regime="short_tool", engines=[{"kv_blocks": 600, "max_batch": 4, "prefill_tps": 20000.0, "decode_tps": 40.0}], window=4)
    engines = [EngineConfig(**e) for e in cfg.engines]
    train = programs_from_spec(regime_spec("short_tool", 300.0, 1), np.random.default_rng(1))
    for arm in ("proxy_rules", "forecast_M1_kv"):
        df = eviction_records(_progs(), engines, _arm(arm, cfg, engines, train, np.random.default_rng(0)), seed=0)
        assert len(df) > 0 and df["arm"].iloc[0] == arm
        if arm == "proxy_rules":
            assert df["estimated_absence"].isna().all()                                     # LRU has no estimate
