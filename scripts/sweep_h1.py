"""Run an H1 config over seeds and overlay rates; write one CSV with the mean and CI of pinball90 per model/horizon."""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from atfm_experiments.h1 import H1Config, run_h1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rates", type=float, nargs="+", default=[None])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = yaml.safe_load(open(a.config))
    frames = []
    for seed, rate in itertools.product(a.seeds, a.rates):
        cfg = dict(base, seed=seed, name=f"{base['name']}_s{seed}" + (f"_r{int(rate)}" if rate else ""))
        if rate:
            cfg["overlay_rate_per_hour"] = rate
        df = run_h1(H1Config(**cfg))
        df["seed"], df["rate"] = seed, rate if rate else base.get("overlay_rate_per_hour")
        frames.append(df)
        print(cfg["name"], "done")
    all_ = pd.concat(frames, ignore_index=True)
    kv = all_[all_["target"] == "kv_blocks"]
    keys = ["class", "model", "h", "rate"]
    g = kv.groupby(keys)["pinball90"]
    summary = g.agg(mean="mean", sd="std", n="count").reset_index()
    summary["ci95"] = 1.96 * summary["sd"] / np.sqrt(summary["n"].clip(lower=1))
    out = Path(a.out or f"runs/{base['name']}_sweep.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.pivot_table(index=["class", "model", "rate"], columns="h", values="mean").round(1).to_string())
    print("written", out)


if __name__ == "__main__":
    main()
