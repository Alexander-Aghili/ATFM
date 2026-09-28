"""GDP CPU scaling with independent session, slot, and sample dimensions."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from atfm.control.gdp import Deferrable, GdpPlanner
from atfm.schema.forecast import ForecastSnapshot


def workload(sessions: int, slots: int, draws: int, saturated: bool):
    rng = np.random.default_rng(7)
    samples = np.cumsum(rng.uniform(800, 1000, (slots, draws)), axis=0)
    snap = ForecastSnapshot(0.0, list(np.arange(1, slots + 1, dtype=float)), "benchmark",
                            {r: {"interactive": samples} for r in ("kv_blocks", "prefill_tokens")})
    planner = GdpPlanner(slot_s=1.0, horizon_s=float(slots), max_hold_s=float(slots))
    defs = [Deferrable(str(i), "t", 0.0, 10, 10) for i in range(sessions)]
    cap = {r: 500.0 if saturated else 1e9 for r in snap.samples}
    return lambda: planner.plan(0.0, snap, cap, defs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--slots", type=int, nargs="+", default=[30, 300])
    parser.add_argument("--draws", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min([args.repeats, *args.sessions, *args.slots, *args.draws]) < 1:
        parser.error("all dimensions and repeats must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sessions in args.sessions:
        for slots in args.slots:
            for draws in args.draws:
                for saturated in (False, True):
                    run = workload(sessions, slots, draws, saturated)
                    def digest(result):
                        return hashlib.sha256(json.dumps([d.__dict__ for d in result], sort_keys=True).encode()).hexdigest()
                    expected = digest(run())
                    times = []
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        result = run()
                        times.append(time.perf_counter() - start)
                        assert digest(result) == expected
                    row = dict(sessions=sessions, slots=slots, draws=draws, saturated=saturated,
                               median_s=float(np.median(times)), min_s=min(times), max_s=max(times),
                               repeats=args.repeats, result_sha256=expected)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(args.out / "timings.csv", index=False)
                    print(row, flush=True)
    source = Path("src/atfm/control/gdp.py")
    (args.out / "environment.json").write_text(json.dumps(dict(
        python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        setup="excluded; one untimed warmup; three repeats by default; serialization excluded"), indent=2))


if __name__ == "__main__":
    main()
