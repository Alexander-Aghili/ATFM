"""Per-eviction diagnostic for one regime and a list of arms (short run by default).

    uv run python scripts/eviction_diagnostic.py experiments/h2sim_loaded_kv.yaml --arms oracle_kv forecast_M2_kv proxy_rules --duration 900
"""
import argparse
import json

import numpy as np
import yaml

from atfm_experiments.h2sim import H2SimConfig, _arm, _programs_for
from atfm.sim.diagnostics import eviction_records, summarize_evictions
from atfm.sim.engine import EngineConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--arms", nargs="+", default=["oracle_kv", "forecast_M2_kv", "proxy_rules"])
    ap.add_argument("--duration", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write per-eviction rows (parquet)")
    a = ap.parse_args()
    cfg = H2SimConfig(**yaml.safe_load(open(a.config)))
    cfg.duration_s = a.duration
    engines = [EngineConfig(**e) for e in cfg.engines]
    train = _programs_for(cfg, a.seed, train=True)
    frames = []
    for arm in a.arms:
        progs = _programs_for(cfg, a.seed, train=False)
        df = eviction_records(progs, engines, _arm(arm, cfg, engines, train, np.random.default_rng(a.seed + 7)), seed=a.seed)
        frames.append(df)
        print(arm, json.dumps(summarize_evictions(df)))
    if a.out and frames:
        import pandas as pd
        pd.concat(frames, ignore_index=True).to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
