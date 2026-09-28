"""Compare optimized planning with the frozen pre-optimization sample scan."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from atfm.control.gdp import Deferrable, GdpPlanner
from atfm.schema.forecast import ForecastSnapshot


def reference_planner(**kwargs):
    spec = importlib.util.spec_from_file_location("gdp_original", Path(__file__).parent / "reference/gdp_original.py")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GdpPlanner(**kwargs)


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.3, 0.5, 0.999, 1.0, -0.1])
@pytest.mark.parametrize("draws", [1, 7, 128])
def test_plan_matches_sample_scan(eps, draws):
    rng = np.random.default_rng(34)
    for _ in range(8):
        slots = 17
        snap = ForecastSnapshot(0.0, [float(i + 1) for i in range(slots)], "test", {
            r: {"interactive": np.cumsum(rng.uniform(0, 100, (slots, draws)), axis=0)}
            for r in ("kv_blocks", "prefill_tokens")})
        defs = [Deferrable(str(i), str(i % 3), float(rng.uniform(-1, 25)),
                           int(rng.integers(0, 100)), int(rng.integers(0, 100))) for i in range(40)]
        options = dict(slot_s=1.0, horizon_s=slots, eps=eps, max_hold_s=5.5)
        new, old = GdpPlanner(**options), reference_planner(**options)
        cap = {r: float(rng.integers(50, 220)) for r in snap.samples}
        args = (100.0, snap, cap, defs, {"1": 2.3})
        assert new.plan(*args) == old.plan(*args)
        assert new.last_assignment == old.last_assignment
        assert new.max_imposed_delay == old.max_imposed_delay


def test_exact_probability_and_floating_point_boundaries():
    for eps in [0.1, 0.3, np.nextafter(0.3, 0), np.nextafter(0.3, 1)]:
        values = np.array([[0.0] * 7 + [1.0] * 3])
        snap = ForecastSnapshot(0, [1.0], "test", {r: {"interactive": values} for r in ("kv_blocks", "prefill_tokens")})
        for cap in [0.0, 1.0, np.nextafter(1.0, 0), float("inf"), float("nan")]:
            options = dict(slot_s=1, horizon_s=1, eps=eps)
            args = (0, snap, {"kv_blocks": cap}, [Deferrable("s", "t", 0, 0, 0)])
            assert GdpPlanner(**options).plan(*args) == reference_planner(**options).plan(*args)
