"""Single-owner control execution, with bounded admission and stage timings."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock
import time

from fastapi import HTTPException


class StageTimings:
    def __init__(self):
        self.lock, self.stages = Lock(), {}

    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self.lock:
                stage = self.stages.setdefault(name, dict(count=0, total_s=0., max_s=0., last_s=0.))
                stage.update(count=stage['count'] + 1, total_s=stage['total_s'] + elapsed,
                             max_s=max(stage['max_s'], elapsed), last_s=elapsed)

    def snapshot(self):
        with self.lock:
            return {name: dict(stage) for name, stage in self.stages.items()}


class ControlWorker:
    """At most one unfinished mutation; cancellation cannot release running work."""

    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='atfm-board')
        self.lock, self.closed, self.busy = Lock(), False, False
        self.accepted, self.rejected, self.completed = 0, 0, 0
        self.timings = StageTimings()

    def submit(self, function, *args):
        with self.lock:
            if self.closed or self.busy:
                self.rejected += 1
                return None
            future = self.pool.submit(function, *args)
            self.busy = True
            self.accepted += 1
        future.add_done_callback(self._completed)
        return future

    def _completed(self, future):
        with self.lock:
            self.busy = False
            self.completed += 1

    async def run(self, function, *args):
        future = self.submit(function, *args)
        if future is None:
            raise HTTPException(503, 'board control worker busy or closed', headers={'Retry-After': '1'})
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(wrapped)

    def snapshot(self):
        with self.lock:
            return dict(busy=self.busy, closed=self.closed, accepted=self.accepted,
                        rejected=self.rejected, completed=self.completed)

    def close(self):
        with self.lock:
            self.closed = True
        self.pool.shutdown(wait=True)
