"""Opt-in per-thread CPU profiles; instrumented timings are not load benchmarks."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from importlib.metadata import version
from pathlib import Path

from .runtime import write_json


@contextmanager
def profile_serving(directory: Path):
    """Profile serving and prediction threads, excluding construction and cleanup."""
    import yappi

    if yappi.is_running():
        raise RuntimeError('proxy profiling requires an isolated profiler')
    yappi.clear_stats()
    yappi.set_clock_type('cpu')
    yappi.start(builtins=True, profile_threads=True)
    try:
        yield
    finally:
        yappi.stop()
        yappi.get_func_stats().save(str(directory / 'proxy.pstats'), type='pstat')
        threads = yappi.get_thread_stats()
        write_json(directory / 'proxy-profile.json', {
            'profiler': 'yappi', 'version': version('yappi'), 'clock': yappi.get_clock_type(),
            'threads': [{'id': t.id, 'name': t.name, 'cpu_s': t.ttot, 'schedules': t.sched_count}
                        for t in threads]})
        with (directory / 'proxy-profile.txt').open('w') as text, \
             (directory / 'proxy-profile.csv').open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['thread', 'thread_name', 'file', 'line', 'function',
                             'primitive_calls', 'calls', 'self_s', 'cumulative_s'])
            for thread in threads:
                stats = yappi.get_func_stats(filter={'ctx_id': thread.id})
                text.write(f'\nThread {thread.id}: {thread.name}, CPU {thread.ttot:.6f}s\n')
                for order in ('tsub', 'ttot'):
                    text.write(f'\nOrdered by {order}\n')
                    stats.sort(order).print_all(out=text, limit=50)
                for stat in stats:
                    writer.writerow([thread.id, thread.name, stat.module, stat.lineno, stat.name,
                                     stat.nactualcall, stat.ncall, stat.tsub, stat.ttot])
        yappi.clear_stats()
