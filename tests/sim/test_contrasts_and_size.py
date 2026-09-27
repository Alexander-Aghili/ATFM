"""Direct paired contrasts between arms, and size-aware eviction ordering. Each test names the production
change that makes it pass."""
import numpy as np
import pandas as pd

from atfm.proxy.config import ProxyConfig
from atfm.sim.engine import EngineConfig


def _per_session(vals: dict[str, list[float]]):
    return {a: pd.DataFrame({"session_id": [f"s{i}" for i in range(len(v))], "x": v}) for a, v in vals.items()}


def test_paired_contrasts_report_named_pairs_from_the_same_draws():
    from atfm.eval.serving import paired_contrasts
    per = _per_session({"native": [1, 2, 3, 4, 5, 6], "a": [2, 3, 4, 5, 6, 7], "b": [2, 3, 4, 5, 6, 7]})
    df = paired_contrasts(per, lambda d: float(d["x"].mean()), pairs=[("a", "b"), ("b", "native")], n_boot=50,
                          rng=np.random.default_rng(0))
    assert list(df["contrast"]) == ["a vs b", "b vs native"]
    ab = df.iloc[0]
    assert ab["diff_mean"] == 0.0 and ab["diff_ci_lo"] == 0.0 and ab["diff_ci_hi"] == 0.0     # identical arms
    bn = df.iloc[1]
    assert abs(bn["diff_mean"] - 1.0) < 1e-9 and bn["diff_ci_lo"] > 0.99 and bn["diff_ci_hi"] < 1.01


def test_runner_adds_m2_vs_m1_contrasts_when_both_arms_present():
    from atfm.experiments.h2sim import H2SimConfig, default_contrasts
    assert default_contrasts(["native", "forecast_M1", "forecast_M2", "forecast_M1_kv", "forecast_M2_kv", "oracle_kv"]) == \
        [("forecast_M2", "forecast_M1"), ("forecast_M2_kv", "forecast_M1_kv"), ("forecast_M2_kv", "oracle_kv"), ("forecast_M1_kv", "oracle_kv")]
    assert default_contrasts(["native", "proxy_rules"]) == []
    cfg = H2SimConfig(name="t", regime="short_tool", contrasts=[["forecast_M2", "forecast_M1"]])
    assert cfg.contrasts == [["forecast_M2", "forecast_M1"]]


def test_size_aware_ordering_evicts_the_most_idle_block_seconds_first():
    """Score = predicted absence x resident blocks: a large context away for a while goes before a small
    context away slightly longer. Without size awareness the order is by absence alone."""
    from atfm.sim.kv_placement import order_victims
    eta = {"big": 100.0, "small": 120.0, "unknown": float("inf")}
    blocks = {"big": 400, "small": 50, "unknown": 10}
    assert order_victims(["big", "small", "unknown"], eta, blocks, now=0.0, size_aware=False) == ["unknown", "small", "big"]
    assert order_victims(["big", "small", "unknown"], eta, blocks, now=0.0, size_aware=True) == ["unknown", "big", "small"]
    # a session that already returned (eta in the past) is kept: zero absence
    eta2 = {"back": -5.0, "away": 30.0}
    assert order_victims(["back", "away"], eta2, {"back": 500, "away": 10}, now=0.0, size_aware=True) == ["away", "back"]


def test_size_aware_arms_registered():
    from atfm.experiments.h2sim import ARMS, H2SimConfig, _arm
    from atfm.sim.kv_placement import OracleKvPolicy
    assert {"forecast_M1_kv_size", "forecast_M2_kv_size", "oracle_kv_size"} <= set(ARMS)
    cfg = H2SimConfig(name="t", regime="short_tool")
    engines = [EngineConfig(**e) for e in cfg.engines]
    pol = _arm("oracle_kv_size", cfg, engines, None, np.random.default_rng(0))
    assert isinstance(pol, OracleKvPolicy) and pol.size_aware and pol.name == "oracle_kv_size"
    pol2 = _arm("oracle_kv", cfg, engines, None, np.random.default_rng(0))
    assert not pol2.size_aware
