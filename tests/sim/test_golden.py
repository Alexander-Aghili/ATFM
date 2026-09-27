"""Golden run (spec 11): a fixed tiny workload under a fixed seed produces fixed serving metrics. Any change
to the engine, the event loop or the policies that moves these numbers must be deliberate."""
import json
from pathlib import Path

import numpy as np

from atfm.experiments.h2sim import H2SimConfig, regime_spec, _arm, build_simulator
from atfm.eval.serving import serving_metrics
from atfm.sim.engine import EngineConfig
from atfm.sim.programs import programs_from_spec

GOLDEN = Path(__file__).parent / "golden_h2sim.json"
KEYS = ["ttft_after_tool_p95", "bg_jct_mean", "deadline_hit_rate", "recomputed_prefill_tokens", "sessions_completed"]


def _run(arm):
    cfg = H2SimConfig(name="golden", regime="long_tool", duration_s=300.0, interactive_rate_per_hour=240.0,
                      background_rate_per_hour=240.0, engines=[{"kv_blocks": 2000, "max_batch": 4, "prefill_tps": 20000.0,
                                                                "decode_tps": 40.0, "priority": True}], window=4)
    engines = [EngineConfig(**e) for e in cfg.engines]
    spec = regime_spec(cfg.regime, cfg.duration_s, 0, cfg.interactive_rate_per_hour, cfg.background_rate_per_hour)
    train = programs_from_spec(spec.model_copy(update={"seed": 1000}), np.random.default_rng(1000))
    progs = programs_from_spec(spec, np.random.default_rng(0))
    sim = build_simulator(cfg, progs, engines, _arm(arm, cfg, engines, train, np.random.default_rng(7)), seed=0)
    log = sim.run()
    m = serving_metrics(log, sim.session_log, cfg.slo_ttft_s, cfg.duration_s, len(engines), makespan_s=sim.now)
    return {k: (round(float(m[k]), 6) if isinstance(m[k], float) else m[k]) for k in KEYS}


def test_golden_metrics_are_reproducible_under_seed():
    got = {arm: _run(arm) for arm in ("native", "proxy_rules", "forecast_M2_kv")}
    if not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(got, indent=2, sort_keys=True))
    want = json.loads(GOLDEN.read_text())
    assert got == want, f"golden metrics moved; if deliberate, delete {GOLDEN} and rerun"
    assert _run("native") == got["native"]                                          # determinism across calls
