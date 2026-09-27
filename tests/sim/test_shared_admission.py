"""Forecast and oracle share the rule but have different completion knowledge."""
from types import SimpleNamespace

import numpy as np
import pytest

from atfm.proxy.config import ProxyConfig
from atfm.schema.forecast import ForecastSnapshot
from atfm.sim.forecast_arm import GdpLite, forecast_hold_until


@pytest.mark.parametrize("true_end, oracle, expected", [
    (200.0, False, None),
    (200.0, True, 130.0),
    (120.0, True, None),
])
def test_only_oracle_can_use_true_completion_times(true_end, oracle, expected):
    worker = SimpleNamespace(
        cfg=SimpleNamespace(max_batch=1),
        running={"r": (SimpleNamespace(session_id="s"), 99.0, true_end)},
        capacity_free_blocks=lambda: 0,
        resident_blocks=lambda sid: 100,
    )
    sim = SimpleNamespace(now=100.0, workers=[worker])
    snapshot = ForecastSnapshot(t=100.0, horizons=[30.0], model_id="test", samples={
        "kv_blocks": {"interactive": np.full((1, 4), 100.0)},
        "prefill_tokens": {"interactive": np.full((1, 4), 100.0)},
    })
    cfg = ProxyConfig(upstream_url="x", prefill_tps=100.0, default_osl=60, decode_tps=60.0)
    assert forecast_hold_until(sim, snapshot, cfg, GdpLite(), 100.0, oracle=oracle) == expected
