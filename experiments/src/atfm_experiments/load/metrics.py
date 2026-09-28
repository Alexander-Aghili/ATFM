"""Bounded process diagnostics; full client/request records are written separately."""
from __future__ import annotations

from collections import defaultdict, deque
import os
from pathlib import Path
import resource
import time

import numpy as np


def distribution(values) -> dict:
    values = list(values)
    if not values:
        return {'count': 0, 'p50': None, 'p95': None, 'p99': None, 'max': None}
    return dict(count=len(values), **dict(zip(('p50', 'p95', 'p99'), map(float, np.percentile(values, [50, 95, 99])))),
                max=float(max(values)))


class Diagnostics:
    def __init__(self):
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.lag = deque(maxlen=4096)
        self.latencies = defaultdict(lambda: deque(maxlen=4096))
        self.counts = defaultdict(int)
        self.errors = defaultdict(int)
        self.in_flight = 0

    def snapshot(self) -> dict:
        rss = None
        statm = Path('/proc/self/statm')
        if statm.exists():
            rss = int(statm.read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        return {'uptime_s': time.perf_counter() - self.started,
                'cpu_s': time.process_time() - self.cpu_started, 'rss_bytes': rss,
                'peak_rss_native': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'in_flight_http': self.in_flight, 'event_loop_lag_s': distribution(self.lag),
                'endpoints': {path: {'total': count, 'errors': self.errors[path],
                                    'recent_duration_s': distribution(self.latencies[path])}
                              for path, count in self.counts.items()}}


class RequestTiming:
    def __init__(self, app, diagnostics: Diagnostics):
        self.app, self.diagnostics = app, diagnostics

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'].startswith('/__load/'):
            return await self.app(scope, receive, send)
        start, status = time.perf_counter(), 500
        d = self.diagnostics
        d.in_flight += 1

        async def measured_send(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
            await send(message)

        try:
            await self.app(scope, receive, measured_send)
        finally:
            self._record(scope, d, status, start)


    def _record(self, scope, d, status, start):
        path = scope['path']
        d.in_flight -= 1
        d.counts[path] += 1
        d.errors[path] += status >= 400
        d.latencies[path].append(time.perf_counter() - start)
