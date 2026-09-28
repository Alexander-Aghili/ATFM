"""Admission permits follow worker lifetime, including abandoned callers."""
import asyncio
from concurrent.futures import Future
from threading import Event
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from atfm.proxy.app import create_app
from atfm.proxy.board_client import BoardClient
from atfm.proxy.config import ProxyConfig
from atfm.proxy.prediction import PredictionRunner

META = SimpleNamespace(session_id='s', isl=12, predicted_osl=3)
FALLBACK = (1., 0.)


class BlockedPredictor:
    def __init__(self):
        self.started, self.release = Event(), Event()
        self.calls = 0

    def expected_times(self, *args):
        self.calls += 1
        self.started.set()
        assert self.release.wait(3)
        return 2., 3.


@pytest.fixture
def blocked():
    predictor = BlockedPredictor()
    runner = PredictionRunner(predictor, limit=1, budget_s=.02)
    try:
        yield runner, predictor
    finally:
        predictor.release.set()
        runner.close()


async def test_timeout_keeps_permit_until_worker_finishes(blocked):
    runner, predictor = blocked
    assert await runner.predict(META, FALLBACK) == FALLBACK
    assert predictor.started.is_set()
    for _ in range(20):
        assert await runner.predict(META, FALLBACK) == FALLBACK
    counts = runner.snapshot()
    assert counts['outstanding'] == counts['running'] == counts['peak_outstanding'] == 1
    assert counts['timeout'] == counts['accepted'] == predictor.calls == 1
    assert counts['rejected'] == 20 and counts['pending'] == 0
    predictor.release.set()
    await _drain(runner)
    assert runner.snapshot()['late_completed'] == 1
    assert await runner.predict(META, FALLBACK) == (2., 3.)
    assert runner.snapshot()['used'] == 1


async def _drain(runner):
    async with asyncio.timeout(2):
        while runner.snapshot()['outstanding']:
            await asyncio.sleep(.001)


async def test_cancellation_keeps_running_work_bounded(blocked):
    runner, predictor = blocked
    runner.budget_s = 2
    task = asyncio.create_task(runner.predict(META, FALLBACK))
    assert await asyncio.to_thread(predictor.started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await runner.predict(META, FALLBACK) == FALLBACK
    assert runner.snapshot()['cancelled'] == runner.snapshot()['rejected'] == 1
    assert runner.snapshot()['outstanding'] == 1
    predictor.release.set()
    await _drain(runner)
    assert runner.snapshot()['completed'] == 1


async def test_burst_never_admits_more_than_limit():
    predictor = BlockedPredictor()
    runner = PredictionRunner(predictor, limit=3, budget_s=.05)
    try:
        results = await asyncio.gather(*(runner.predict(META, FALLBACK) for _ in range(40)))
        assert results == [FALLBACK] * 40
        counts = runner.snapshot()
        assert counts['accepted'] == counts['peak_outstanding'] == 3
        assert counts['rejected'] == 37 and counts['timeout'] == 3
    finally:
        predictor.release.set()
        runner.close()
    assert runner.snapshot()['completed'] == 3 and runner.snapshot()['outstanding'] == 0


class DeferredExecutor:
    def submit(self, fn, *args):
        self.future = Future()
        return self.future


async def test_cancel_before_dispatch_returns_permit(blocked):
    runner, predictor = blocked
    runner.pool.shutdown()
    runner.pool = DeferredExecutor()
    task = asyncio.create_task(runner.predict(META, FALLBACK))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner.snapshot()['cancelled_before_start'] == 1
    assert runner.snapshot()['outstanding'] == predictor.calls == 0
    runner.pool = Mock()


def test_expired_job_skips_predictor(blocked):
    runner, predictor = blocked
    future = runner._submit(META, time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        future.result(timeout=1)
    runner.close()
    assert predictor.calls == 0
    assert runner.snapshot()['expired_before_start'] == runner.snapshot()['completed'] == 1


async def test_submission_failure_does_not_leak_capacity(blocked):
    runner, _ = blocked
    runner.pool.shutdown()
    assert await runner.predict(META, FALLBACK) == FALLBACK
    assert runner.snapshot()['error'] == 1
    assert runner.snapshot()['outstanding'] == runner.snapshot()['accepted'] == 0


async def test_closed_runner_falls_back_without_submitting(blocked):
    runner, _ = blocked
    runner.close()
    assert await runner.predict(META, FALLBACK) == FALLBACK
    assert runner.snapshot()['rejected'] == 1
    assert runner.snapshot()['accepted'] == 0


@pytest.mark.parametrize('field,value', [('prediction_limit', 0), ('prediction_limit', 1.5),
                                        ('board_timeout_s', 0), ('board_timeout_s', float('inf'))])
def test_invalid_prediction_configuration(field, value):
    with pytest.raises(ValueError):
        ProxyConfig(upstream_url='http://up', **{field: value})


def test_board_transport_uses_remaining_deadline():
    seen = []
    def respond(request):
        seen.append(request.extensions['timeout'])
        return httpx.Response(200, json={'e_service_s': 2, 'e_tool_next_s': 3})
    board = BoardClient('http://board', timeout_s=.5)
    board.client.close()
    with httpx.Client(transport=httpx.MockTransport(respond)) as board.client:
        assert board.expected_times_until('s', 1, 2, deadline=time.monotonic() + .1) == (2., 3.)
        with pytest.raises(TimeoutError):
            board.expected_times_until('s', 1, 2, deadline=time.monotonic() - 1)
    assert len(seen) == 1
    assert all(0 < value <= .1 for value in seen[0].values())


def test_board_transport_timeout_is_a_deadline_failure():
    board = BoardClient('http://board')
    board.client.close()
    board.client = Mock(post=Mock(side_effect=httpx.ReadTimeout('slow')))
    with pytest.raises(TimeoutError):
        board.expected_times_until('s', 1, 2, deadline=time.monotonic() + 1)


async def test_app_shutdown_owns_board_but_not_injected_upstream():
    upstream = httpx.AsyncClient()
    app = create_app(ProxyConfig(upstream_url='http://up', board_url='http://board'), upstream_client=upstream)
    async with app.router.lifespan_context(app):
        assert app.state.predictor.client.timeout.read == .05
    assert app.state.predictor.client.is_closed
    assert app.state.prediction_runner.closed
    assert not upstream.is_closed
    await upstream.aclose()


def test_legacy_predictor_skips_second_call_after_deadline():
    class Legacy:
        def expected_service(self, *args):
            return 2.
        def expected_tool_next(self, *args):
            pytest.fail('expired work must not start a second estimate')
    runner = PredictionRunner(Legacy(), limit=1, budget_s=.05)
    try:
        with pytest.raises(TimeoutError):
            runner._compute(META, time.monotonic() - 1)
    finally:
        runner.close()
