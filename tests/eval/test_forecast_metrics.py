import numpy as np
from atfm.eval.forecast import crps, pinball, coverage, surge_events, score_run

def test_crps_point_forecast_is_abs_error():
    s = np.full((3, 50), 4.0); y = np.array([4.0, 6.0, 1.0])
    assert np.allclose(crps(s, y), [0.0, 2.0, 3.0])

def test_crps_matches_gaussian_closed_form():
    rng = np.random.default_rng(0)
    s = rng.normal(0.0, 1.0, size=(1, 200000)); y = np.array([0.0])
    # CRPS of N(0,1) at y=0: 2*phi(0) - 1/sqrt(pi) = 0.2339
    assert abs(crps(s, y)[0] - (2 * 0.3989422804 - 1 / np.sqrt(np.pi))) < 0.01

def test_pinball_asymmetry():
    s = np.full((1, 100), 10.0)
    under = pinball(s, np.array([19.0]), 0.9)
    over = pinball(s, np.array([1.0]), 0.9)
    assert abs(under[0] - 8.1) < 1e-9 and abs(over[0] - 0.9) < 1e-9 and abs(under[0] / over[0] - 9.0) < 1e-9

def test_coverage():
    s = np.arange(100.0)[None, :].repeat(2, 0)
    assert coverage(s, np.array([50.0, 200.0]), 0.8).tolist() == [True, False]

def test_surge_events():
    truth = np.array([0, 0, 5, 5, 5, 0, 0, 0, 0, 0], float)
    q90 = np.array([0, 3, 5, 5, 0, 0, 0, 3, 3, 0], float)
    ev = surge_events(q90, truth, capacity=2.0, tick_s=10.0, horizon_s=30.0)
    assert ev["lead_times"] == [10.0] and ev["false_alarms"] == 1 and ev["missed"] == 0

def test_score_run_aggregates():
    recs = []
    for t in range(4):
        recs.append({"t": float(t), "model": "m", "target": "kv_blocks", "class": "background", "h_index": 0, "h": 10.0,
                     "samples": np.full(8, 5.0), "truth": 5.0 + t, "perturbed": t >= 2, "endogenous_fraction": 0.5})
    df = score_run(recs)
    row = df.iloc[0]
    assert row["n"] == 4 and abs(row["crps"] - 1.5) < 1e-9 and abs(row["crps_pert"] - 2.5) < 1e-9
    assert abs(row["endogenous_fraction"] - 0.5) < 1e-9
