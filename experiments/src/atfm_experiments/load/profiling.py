"""Opt-in main-thread profiles; instrumented timings are not load benchmarks."""
from __future__ import annotations

import cProfile
from contextlib import contextmanager
import csv
from pathlib import Path
import pstats


@contextmanager
def profile_event_loop(directory: Path):
    """Profile the serving thread, excluding construction and shutdown cleanup."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        profiler.dump_stats(str(directory / 'proxy.pstats'))
        stats = pstats.Stats(profiler)
        with (directory / 'proxy-profile.txt').open('w') as stream:
            stats.stream = stream
            for order in ('tottime', 'cumulative'):
                stats.sort_stats(order).print_stats(60)
        with (directory / 'proxy-profile.csv').open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['file', 'line', 'function', 'primitive_calls', 'calls', 'self_s', 'cumulative_s'])
            for (file, line, function), (primitive, calls, own, cumulative, _) in sorted(stats.stats.items()):
                writer.writerow([file, line, function, primitive, calls, own, cumulative])
