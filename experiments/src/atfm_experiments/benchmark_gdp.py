"""GDP CPU scaling with independent session, slot, and sample dimensions."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from atfm.control.gdp import Deferrable, GdpPlanner
from atfm.schema.forecast import ForecastSnapshot


def workload(sessions: int, slots: int, draws: int, saturated: bool, mixed: bool = False):
    rng = np.random.default_rng(7)
    samples = np.cumsum(rng.uniform(800, 1000, (slots, draws)), axis=0)
    snap = ForecastSnapshot(0.0, list(np.arange(1, slots + 1, dtype=float)), "benchmark",
                            {r: {"interactive": samples} for r in ("kv_blocks", "prefill_tokens")})
    planner = GdpPlanner(slot_s=1.0, horizon_s=float(slots), max_hold_s=float(slots))
    defs = [Deferrable(str(i), "t", 0.0, 10, 10) for i in range(sessions)]
    if mixed:
        defs = [Deferrable(str(i), f"t{i % 16}", float(rng.integers(0, slots)),
                           int(rng.integers(1, 100)), int(rng.integers(1, 100))) for i in range(sessions)]
    cap = {r: (1000.0 if mixed else 500.0 if saturated else 1e9) for r in snap.samples}
    return lambda: planner.plan(0.0, snap, cap, defs)


def digest(result) -> str:
    """Hash the canonical JSON list without allocating a second full result tree."""
    checksum = hashlib.sha256(b"[")
    for i, directive in enumerate(result):
        if i:
            checksum.update(b", ")
        checksum.update(json.dumps(directive.__dict__, sort_keys=True).encode())
    checksum.update(b"]")
    return checksum.hexdigest()


def main():
    args = _arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sessions in args.sessions:
        for slots in args.slots:
            for draws in args.draws:
                for regime in args.regimes:
                    row = _trial(sessions, slots, draws, regime, args)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(args.out / "timings.csv", index=False)
                    print(row, flush=True)
    _environment(args)


def _environment(args):
    source = Path("src/atfm/control/gdp.py")
    (args.out / "environment.json").write_text(json.dumps(dict(
        python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arguments=vars(args) | {"out": str(args.out)},
        rss="Process high-water RSS including imports, setup, results and hashing; cumulative across cases. Run one case per process for attribution.",
        setup="excluded; one untimed warmup; three repeats by default; serialization excluded"), indent=2))


def _trial(sessions, slots, draws, regime, args):
    saturated = regime == "saturated"
    run = workload(sessions, slots, draws, saturated, mixed=regime == "mixed")
    expected, times, cpu_times = _measure(run, args)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row = dict(sessions=sessions, slots=slots, draws=draws, saturated=saturated, regime=regime,
               cpu_median_s=float(np.median(cpu_times)),
               process_peak_rss_mib=peak_rss / (1024 ** 2 if sys.platform == "darwin" else 1024),
               median_s=float(np.median(times)), min_s=min(times), max_s=max(times),
               repeats=args.repeats, result_sha256=expected)
    return row


def _measure(run, args):
    expected = digest(run())
    times, cpu_times = [], []
    for _ in range(args.repeats):
        cpu_start = time.process_time()
        start = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - start)
        cpu_times.append(time.process_time() - cpu_start)
        assert digest(result) == expected
    return expected, times, cpu_times


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--slots", type=int, nargs="+", default=[30, 300])
    parser.add_argument("--draws", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--regimes", nargs="+", choices=["open", "saturated", "mixed"], default=["open", "saturated"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min([args.repeats, *args.sessions, *args.slots, *args.draws]) < 1:
        parser.error("all dimensions and repeats must be positive")
    return args


if __name__ == "__main__":
    main()
