from atfm.experiments.h2sim import H2SimConfig, run_h2sim, regime_spec

def test_h2sim_smoke_all_arms(tmp_path):
    cfg = H2SimConfig(name="smoke", regime="long_tool", seeds=[0], arms=["native", "proxy_rules", "forecast_M1", "forecast_M2", "oracle", "working_set"],
                      engines=[{"kv_blocks": 4000, "max_batch": 4, "prefill_tps": 20000.0, "decode_tps": 40.0}], window=4, beta=0.5,
                      slo_ttft_s=2.0, working_set_budget=3000, duration_s=600.0, out_dir=str(tmp_path))
    df = run_h2sim(cfg)
    assert set(df.arm) == set(cfg.arms) and (df.seed == 0).all()
    assert (tmp_path / "smoke" / "metrics.csv").exists() and (tmp_path / "smoke" / "paired.csv").exists()
    assert (df.groupby("arm").sessions_completed.first() > 20).all()
    assert regime_spec("short_tool").classes[0].tools[0].log_mu < regime_spec("long_tool").classes[0].tools[0].log_mu
