"""Control contention, publication coherence, freshness, and worker lifetime."""
import asyncio
from contextlib import asynccontextmanager
import threading

import httpx
import numpy as np
import pytest

from atfm.board.execution import ControlWorker
from atfm.board.forecaster import ExogenousModel, SessionForecaster
from atfm.board.live import LiveBoard, SessionRegistry
from atfm.board.publication import PredictionView
from atfm.board.service import create_board_app
from atfm.bus import InMemoryBus
from atfm.schema.events import ToolStart
from tests.board.test_shared_live import Predictor, start


@asynccontextmanager
async def serving():
    board = LiveBoard(SessionRegistry(), SessionForecaster(Predictor(), ExogenousModel(), [10.], n=4))
    bus = InMemoryBus()
    app = create_board_app(board, bus=bus, clock=lambda: 10., prediction_max_age_s=2.)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://board') as client:
            yield app, board, bus, client


def pause_step(board):
    entered, release = threading.Event(), threading.Event()
    original = board.step
    def blocked(*args):
        entered.set()
        assert release.wait(5), 'test did not release worker'
        return original(*args)
    board.step = blocked
    return entered, release


async def wait_entered(event):
    for _ in range(500):
        if event.is_set():
            return
        await asyncio.sleep(.002)
    pytest.fail('worker did not start')


async def test_predictions_and_health_serve_old_version_during_tick():
    async with serving() as (app, board, bus, client):
        await client.post('/tick')
        bus.publish(ToolStart(t=9, session_id='new', turn_index=0, call_id='c', tool_name='current'))
        entered, release = pause_step(board)
        tick = asyncio.create_task(client.post('/tick'))
        try:
            await wait_entered(entered)
            p = (await asyncio.wait_for(client.post('/predict', json={'session_id': 'new'}), .5)).json()
            assert p['prediction_version'] == 1 and p['e_tool_next_s'] == 5.
            assert (await client.get('/healthz')).status_code == 200
            assert (await client.post('/tick')).status_code == 503
            assert (await client.post('/directives')).status_code == 503
        finally:
            release.set()
            await tick
        p = (await client.post('/predict', json={'session_id': 'new'})).json()
        assert p['prediction_version'] == 2 and p['e_tool_next_s'] == 10.


async def test_cancelled_tick_keeps_worker_busy_until_completion():
    async with serving() as (app, board, bus, client):
        entered, release = pause_step(board)
        tick = asyncio.create_task(client.post('/tick'))
        try:
            await wait_entered(entered)
            tick.cancel()
            with pytest.raises(asyncio.CancelledError):
                await tick
            assert (await client.post('/tick')).status_code == 503
            assert app.state.control_worker.snapshot()['busy']
        finally:
            release.set()
        while app.state.control_worker.snapshot()['busy']:
            await asyncio.sleep(.002)
        assert app.state.reader.status()['version'] == 1
        assert (await client.post('/tick')).status_code == 200


async def test_freshness_uses_capture_time_and_monotonic_clock():
    async with serving() as (app, board, bus, client):
        mono = [100.]
        app.state.reader.monotonic = lambda: mono[0]
        await client.post('/tick')
        mono[0] += 2.
        fresh = (await client.post('/predict', json={'isl': 10})).json()
        assert not fresh['stale'] and not fresh['over_budget']
        mono[0] += .001
        stale = (await client.post('/predict', json={'isl': 10})).json()
        assert stale['stale'] and stale['over_budget'] and stale['e_service_s'] == 0.
        await client.post('/tick')
        assert not (await client.post('/predict', json={})).json()['stale']
        assert app.state.reader.status()['predictions']['stale'] == 1


async def test_failed_forecast_leaves_publication_intact():
    async with serving() as (app, board, bus, client):
        await client.post('/tick')
        previous = app.state.reader.published.read()
        def fail(*args):
            raise ValueError('forecast failed')
        board.step = fail
        bus.publish(start('new', 9.))
        with pytest.raises(ValueError, match='forecast failed'):
            await client.post('/tick')
        assert app.state.reader.published.read() is previous
        assert not app.state.control_worker.snapshot()['busy']


def test_projection_is_detached_and_matches_live_predictions():
    board = LiveBoard(SessionRegistry(), SessionForecaster(Predictor(), ExogenousModel(), [10.]))
    for sid in ('running', 'pending', 'empty'):
        board.registry.apply(start(sid, 0.))
    running, pending = board.registry.get('running'), board.registry.get('pending')
    running.phase, running.tool_name = 'tool_running', 'current'
    pending.tool_history = [('previous', 1.)]
    view = PredictionView.capture(board, 1.)
    for sid in ('running', 'pending', 'empty', 'unknown'):
        assert view.predict(sid, 200, 10) == (board.expected_service(sid, 200, 10), board.expected_tool_next(sid))
    running.tool_name = 'previous'
    pending.tool_history.clear()
    assert view.predict('running', 0, 0)[1] == 10.
    assert view.predict('pending', 0, 0)[1] == 20.
    with pytest.raises(TypeError):
        view.tools['running'] = 0.


async def test_snapshot_is_preencoded_and_directives_preserve_rng_order():
    async with serving() as (app, board, bus, client):
        bus.publish(start('a', 0.))
        reference = LiveBoard(SessionRegistry(), SessionForecaster(Predictor(), ExogenousModel(), [10.], n=4))
        reference.registry.apply(start('a', 0.))
        expected = reference.step(10., np.random.default_rng(0))
        await client.post('/tick')
        actual = app.state.reader.published.read().snapshot
        for target, classes in expected.samples.items():
            for cls, values in classes.items():
                np.testing.assert_array_equal(actual.samples[target][cls], values)
        body = (await client.get('/snapshot')).content
        actual.quantiles = lambda *args: pytest.fail('quantiles recomputed in request')
        assert (await client.get('/snapshot')).content == body
        a = (await client.post('/directives')).json()
        assert (await client.post('/directives')).json() == a


@pytest.mark.parametrize('age', [0, -1, float('inf'), float('nan')])
async def test_invalid_freshness_configuration(age):
    async with serving() as (_, board, _, _):
        with pytest.raises(ValueError, match='prediction_max_age_s'):
            create_board_app(board, prediction_max_age_s=age)


def test_control_worker_releases_failed_submission_and_rejects_after_close(monkeypatch):
    worker = ControlWorker()
    def fail(*args):
        raise RuntimeError('submit failed')
    monkeypatch.setattr(worker.pool, 'submit', fail)
    with pytest.raises(RuntimeError, match='submit failed'):
        worker.submit(lambda: None)
    assert not worker.snapshot()['busy']
    worker.close()
    assert worker.submit(lambda: None) is None


async def test_directive_computation_does_not_block_prediction(monkeypatch):
    async with serving() as (app, board, bus, client):
        await client.post('/tick')
        entered, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return {}
        monkeypatch.setattr('atfm.board.service.resumption_quantiles', blocked)
        task = asyncio.create_task(client.post('/directives'))
        try:
            await wait_entered(entered)
            p = await asyncio.wait_for(client.post('/predict', json={}), .5)
            assert p.status_code == 200 and not p.json()['over_budget']
            assert (await asyncio.wait_for(client.get('/snapshot'), .5)).json()['t'] == 10.
        finally:
            release.set()
            await task


async def test_slow_tick_publication_is_already_stale():
    async with serving() as (app, board, bus, client):
        mono = [100.]
        app.state.reader.monotonic = lambda: mono[0]
        original = board.step
        def slow(*args):
            mono[0] += 3.
            return original(*args)
        board.step = slow
        await client.post('/tick')
        result = (await client.post('/predict', json={})).json()
        assert result['prediction_version'] == 1
        assert result['prediction_age_s'] == 3. and result['stale']


async def test_metrics_update_is_skipped_while_control_worker_busy():
    from atfm.board.service import configure_controllers
    async with serving() as (app, board, bus, client):
        scrape = configure_controllers(app, {'metrics': {'default_total_blocks': 100, 'default_worker_id': 'w0'}},
                                       fetch=lambda: 'vllm:gpu_cache_usage_perc{model_name="m"} 0.5\n')
        entered, release = pause_step(board)
        task = asyncio.create_task(client.post('/tick'))
        try:
            await wait_entered(entered)
            await asyncio.to_thread(scrape)
            assert app.state.worker_metrics == {}
        finally:
            release.set()
            await task
        await asyncio.to_thread(scrape)
        assert app.state.worker_metrics


def test_shutdown_drains_control_and_prevents_new_work():
    worker = ControlWorker()
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    def blocked():
        entered.set()
        assert release.wait(5)
    future = worker.submit(blocked)
    assert entered.wait(1)
    closer = threading.Thread(target=lambda: (worker.close(), closed.set()))
    closer.start()
    try:
        assert not closed.wait(.02)
        assert worker.submit(lambda: None) is None
    finally:
        release.set()
        closer.join(2)
    assert closed.is_set() and future.done()
    assert worker.snapshot()['accepted'] == worker.snapshot()['completed'] == 1


async def test_projection_failure_for_one_tool_falls_back_without_failing_tick():
    async with serving() as (app, board, bus, client):
        bus.publish(ToolStart(t=9., session_id='bad', turn_index=0, call_id='c', tool_name='unknown'))
        assert (await client.post('/tick')).status_code == 200
        result = (await client.post('/predict', json={'session_id': 'bad'})).json()
        assert result['over_budget'] and result['e_service_s'] == result['e_tool_next_s'] == 0.
        assert not (await client.post('/predict', json={'session_id': 'missing'})).json()['over_budget']
