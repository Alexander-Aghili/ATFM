"""Bound synchronous prediction work independently of callers waiting for results."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time


class PredictionRunner:
    """One permit per unfinished job; caller timeout never releases a running job's permit."""

    def __init__(self, predictor, limit: int, budget_s: float):
        self.predictor, self.limit, self.budget_s = predictor, limit, budget_s
        self.pool = ThreadPoolExecutor(max_workers=limit, thread_name_prefix='atfm-predict')
        self.lock, self.closed = Lock(), False
        self.counts = dict.fromkeys(('disabled', 'attempted', 'used', 'timeout', 'error', 'cancelled',
                                    'pending', 'accepted', 'rejected', 'outstanding', 'peak_outstanding',
                                    'running', 'completed', 'cancelled_before_start', 'expired_before_start',
                                    'late_completed'), 0)

    def snapshot(self):
        with self.lock:
            return dict(self.counts)

    def _count(self, **changes):
        with self.lock:
            for name, amount in changes.items():
                self.counts[name] += amount

    async def predict(self, meta, fallback):
        if self.predictor is None:
            self._count(disabled=1)
            return fallback
        deadline = time.monotonic() + self.budget_s
        self._count(attempted=1, pending=1)
        try:
            future = self._submit(meta, deadline)
            if future is None:
                self._count(pending=-1, rejected=1)
                return fallback
            return await self._wait(future, deadline)
        except asyncio.CancelledError:
            self._count(pending=-1, cancelled=1)
            raise
        except Exception as exc:
            self._count(pending=-1, **{'timeout' if isinstance(exc, TimeoutError) else 'error': 1})
            return fallback

    def _submit(self, meta, deadline):
        with self.lock:
            if self.closed or self.counts['outstanding'] >= self.limit:
                return None
            future = self.pool.submit(self._invoke, meta, deadline)
            self.counts['accepted'] += 1
            self.counts['outstanding'] += 1
            self.counts['peak_outstanding'] = max(self.counts['peak_outstanding'], self.counts['outstanding'])
        future.add_done_callback(lambda done: self._completed(done, deadline))
        return future

    def _invoke(self, meta, deadline):
        if time.monotonic() >= deadline:
            self._count(expired_before_start=1)
            raise TimeoutError('prediction expired before execution')
        self._count(running=1)
        try:
            return self._compute(meta, deadline)
        finally:
            self._count(running=-1)

    def _compute(self, meta, deadline):
        args = (meta.session_id, meta.isl, meta.predicted_osl)
        deadline_method = getattr(type(self.predictor), 'expected_times_until', None)
        if deadline_method is not None:
            return deadline_method(self.predictor, *args, deadline=deadline)
        combined = getattr(self.predictor, 'expected_times', None)
        if combined is not None:
            return combined(*args)
        service = float(self.predictor.expected_service(*args))
        if time.monotonic() >= deadline:
            raise TimeoutError("prediction expired before next-tool estimate")
        return service, float(self.predictor.expected_tool_next(meta.session_id))

    def _completed(self, future, deadline):
        self._count(outstanding=-1, completed=1, cancelled_before_start=int(future.cancelled()),
                    late_completed=int(not future.cancelled() and time.monotonic() >= deadline))

    async def _wait(self, future, deadline):
        try:
            remaining = max(0., deadline - time.monotonic())
            result = await asyncio.wait_for(asyncio.wrap_future(future), remaining)
            if time.monotonic() >= deadline:
                raise TimeoutError('prediction arrived after caller deadline')
            self._count(pending=-1, used=1)
            return result
        finally:
            future.cancel()

    def close(self):
        with self.lock:
            self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)
