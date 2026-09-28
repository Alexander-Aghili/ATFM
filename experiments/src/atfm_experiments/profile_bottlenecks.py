"""Profile current control hot paths; local component evidence, not fleet capacity."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import pstats
import time

import numpy as np

from atfm.bus import JsonlBus
from atfm.proxy.queue import Entry, HoldQueue
from atfm.schema.events import ToolProgress, event_to_dict
from atfm_experiments.benchmark_gdp import workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=["gdp", "jsonl", "queue"], default=["gdp", "jsonl", "queue"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    _profile_gdp(args)
    _profile_jsonl(args, rows)
    _profile_queue(args, rows)
    (args.out / "timings.json").write_text(json.dumps(rows, indent=2) + "\n")
    _environment(args)


def _environment(args):
    paths = ["src/atfm/control/gdp.py", "src/atfm/proxy/queue.py", "src/atfm/bus/jsonl.py", __file__]
    (args.out / "environment.json").write_text(json.dumps(dict(
        python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
        source_sha256={str(Path(p).relative_to(Path.cwd()) if Path(p).is_absolute() else p):
                       hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        notes="GDP times are instrumented profiles, not repeated wall-time benchmarks. JSONL warmup + 3 runs; queue setup excluded, warmup + 3 runs."), indent=2) + "\n")


def _profile_queue(args, rows):
    for size in ((1000, 4000, 100000) if "queue" in args.cases else []):
        timings = []
        for repeat in range(4):
            queue = HoldQueue(1, clock=lambda: 0.0, release_order_max=size)
            queue.pending = [Entry(str(i), i % 3, float(i % 7), float(-i)) for i in range(size)]
            start = time.perf_counter()
            queue.tick()
            while queue.queued:
                queue.complete()
            elapsed = time.perf_counter() - start
            assert len(queue.release_order) == size
            if repeat:
                timings.append(elapsed)
        rows.append(dict(case="single_slot_queue_drain", size=size, median_s=float(np.median(timings)),
                         min_s=min(timings), max_s=max(timings)))
        print(rows[-1], flush=True)


def _profile_jsonl(args, rows):
    for size in ((10000, 100000, 1000000) if "jsonl" in args.cases else []):
        path = args.out / "events.jsonl"
        line = json.dumps(event_to_dict(ToolProgress(t=1, session_id="s", call_id="c", completed=1))) + "\n"
        with path.open("w") as stream:
            for _ in range(size):
                stream.write(line)
        bus = JsonlBus(path)
        assert len(bus.drain()) == size
        timings = _drain_timings(bus)
        rows.append(dict(case="unchanged_jsonl_drain", size=size, median_s=float(np.median(timings)),
                         min_s=min(timings), max_s=max(timings), bytes=path.stat().st_size))
        path.unlink()
        print(rows[-1], flush=True)


def _drain_timings(bus):
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        events = bus.drain()
        timings.append(time.perf_counter() - start)
        assert events == []
        del events
    return timings


def _profile_gdp(args):
    for sessions, slots, draws in ([(100000, 10000, 1024), (1000000, 300, 128)] if "gdp" in args.cases else []):
        run = workload(sessions, slots, draws, True)
        profile = cProfile.Profile()
        result = profile.runcall(run)
        assert len(result) == sessions
        with (args.out / f"gdp-{sessions}-profile.txt").open("w") as stream:
            stats = pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative")
            stats.print_stats(25)
        del result


if __name__ == "__main__":
    main()
