import numpy as np
from atfm.board.calibrate import inflate, fit_inflation, CalibratedForecaster
from atfm.schema.forecast import ForecastSnapshot
from atfm.eval.forecast import coverage

class _Narrow:
    """Forecaster whose samples are unbiased but four times too narrow."""
    def __init__(self, truth_fn):
        self.truth_fn = truth_fn
        self.model_id = "narrow"
    def forecast(self, t, states, rng):
        mu = self.truth_fn(t)
        s = mu[:, None] + rng.normal(0, 1.0, size=(len(mu), 64))   # true noise sd is 4
        return ForecastSnapshot(t=t, horizons=[10.0, 30.0], model_id=self.model_id,
                                samples={"kv_blocks": {"background": s, "interactive": np.zeros_like(s)},
                                         "prefill_tokens": {"background": s * 16, "interactive": np.zeros_like(s)}},
                                endogenous_fraction={"background": np.ones(2), "interactive": np.zeros(2)})

def test_inflate_keeps_mean_and_scales_spread():
    s = np.array([[10.0, 12.0, 14.0]])
    out = inflate(s, np.array([3.0]))
    assert np.allclose(out.mean(axis=1), 12.0) and np.allclose(out, [[6.0, 12.0, 18.0]])
    assert np.allclose(inflate(s, np.array([1.0])), s)
    assert np.allclose(inflate(np.array([[0.0, 2.0, 4.0]]), np.array([3.0])), [[0.0, 2.0, 8.0]])   # clipped at 0

def test_fit_inflation_reaches_target_coverage():
    rng = np.random.default_rng(0)
    truth = {t: np.array([100.0, 200.0]) + rng.normal(0, 4.0, size=2) for t in range(200)}
    fc = _Narrow(lambda t: np.array([100.0, 200.0]))
    k = fit_inflation(fc, ticks=list(range(200)), truth_fn=lambda t: {"kv_blocks": {"background": truth[t], "interactive": np.zeros(2)}},
                      target=0.9, rng=np.random.default_rng(1))
    assert k.shape == (2,) and (k > 2.5).all() and (k < 6.0).all()
    cal = CalibratedForecaster(fc, k)
    hits = []
    for t in range(200):
        snap = cal.forecast(t, [], np.random.default_rng(t))
        hits.append(coverage(snap.samples["kv_blocks"]["background"], truth[t], 0.9))
    cov = np.mean(hits, axis=0)
    assert (cov > 0.8).all() and cal.model_id == "narrow+cal"
