from unittest.mock import Mock

import httpx
import pytest

from atfm.proxy.app import create_app
from atfm.proxy.board_client import BoardClient
from atfm.proxy.config import ProxyConfig
from atfm.proxy.queue import Entry


async def test_batch_validates_before_mutation_ticks_once_and_filters_expiry():
    app = create_app(ProxyConfig(upstream_url='http://up'), clock=lambda: 10.0)
    queue = app.state.queue
    queue.tick = Mock(wraps=queue.tick)
    holds = [dict(session_id=str(i), release_not_before=20, expires_at=30) for i in range(100)]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://proxy') as client:
        bad = await client.post('/directives/batch', json={'holds': holds + [{'session_id': 'bad'}]})
        assert bad.status_code == 422
        assert queue.directives == {}
        queue.tick.assert_not_called()
        holds.append(dict(session_id='expired', release_not_before=20, expires_at=10))
        result = await client.post('/directives/batch', json={'holds': holds})
        assert result.json() == {'ok': True, 'applied': 100, 'expired': 1}
        queue.tick.assert_called_once()
        assert 'expired' not in queue.directives
        assert (await client.post('/directives/batch', json={'holds': holds * 11})).status_code == 422
        assert (await client.post('/directives/batch', json={'holds': []})).json()['applied'] == 0
        queue.tick.assert_called_once()
    await app.state.client.aclose()


async def test_batch_and_single_hold_state_match_without_retiming_waiters():
    app = create_app(ProxyConfig(upstream_url='http://up', window=0), clock=lambda: 10.0)
    queue = app.state.queue
    waiting = Entry('s', 0, 0, 10)
    queue.submit(waiting)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://proxy') as client:
        hold = dict(session_id='s', release_not_before=20, expires_at=30)
        assert (await client.post('/directives/batch', json={'holds': [hold]})).status_code == 200
        assert waiting.not_before == 0
        expected = dict(queue.directives)
        assert (await client.post('/directives', json=hold)).json() == {'ok': True}
        assert queue.directives == expected
        new = Entry('s', 0, 0, 10)
        queue.submit(new)
        assert new.not_before == 20
    if queue._timer:
        queue._timer.cancel()
    await app.state.client.aclose()


def test_combined_prediction_uses_one_request():
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={'e_service_s': 2, 'e_tool_next_s': 3, 'over_budget': False})
    board = BoardClient('http://board')
    board.client.close()
    with httpx.Client(transport=httpx.MockTransport(respond)) as board.client:
        assert board.expected_times('s', 20, 30) == (2, 3)
    assert len(calls) == 1


@pytest.mark.parametrize('body', [{}, {'e_service_s': 0, 'e_tool_next_s': 0, 'over_budget': True},
                                  {'e_service_s': -1, 'e_tool_next_s': 2}])
def test_bad_combined_prediction_signals_fallback(body):
    board = BoardClient('http://board')
    board.client.close()
    board._predict = lambda *args: body
    with pytest.raises((KeyError, ValueError)):
        board.expected_times('s', 1, 1)


@pytest.mark.parametrize('status', [200, 503])
async def test_proxy_combines_board_calls_and_uses_local_fallback(status):
    from tests.proxy.test_hardening import _upstream
    from atfm.proxy.index import estimate_isl

    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(status, json={'e_service_s': 2, 'e_tool_next_s': 3})
    board = BoardClient('http://board')
    board.client.close()
    cfg = ProxyConfig(upstream_url='http://up', beta=1, board_timeout_s=1)
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=_upstream()), base_url='http://up')
    body = {'model': 'm', 'messages': [{'role': 'user', 'content': 'hello'}]}
    with httpx.Client(transport=httpx.MockTransport(respond)) as board.client:
        app = create_app(cfg, upstream_client=upstream, predictor=board)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://proxy') as client:
            response = await client.post('/v1/chat/completions', json=body,
                                         headers={'x-atfm-session': 's', 'x-atfm-class': 'background'})
            assert response.status_code == 200
        expected = 2 if status == 200 else 1 / (estimate_isl(body) / cfg.prefill_tps + cfg.default_osl / cfg.decode_tps)
        assert app.state.last_index['s'] == pytest.approx(expected * cfg.w_background)
        assert len(calls) == 1
        app.state.pool.shutdown()
    await upstream.aclose()


@pytest.mark.parametrize('outcome', ['used', 'error', 'timeout', 'disabled', 'cancelled'])
async def test_prediction_outcomes_account_for_each_request(outcome):
    import asyncio
    import threading
    from tests.proxy.test_hardening import _upstream

    started, release = threading.Event(), threading.Event()
    def predict(*args):
        started.set()
        if outcome in ('timeout', 'cancelled'):
            release.wait(2)
        if outcome == 'error':
            raise ValueError('bad prediction')
        return 2., 3.

    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=_upstream()), base_url='http://up')
    app = create_app(ProxyConfig(upstream_url='http://up', board_timeout_s=.02 if outcome == 'timeout' else 1),
                     upstream_client=upstream,
                     predictor=None if outcome == 'disabled' else Mock(expected_times=predict))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://proxy') as client:
            task = asyncio.create_task(client.post('/v1/chat/completions', json={'messages': []}))
            if outcome == 'cancelled':
                assert await asyncio.to_thread(started.wait, 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                assert (await task).status_code == 200
            counts = (await client.get('/state')).json()['predictions']
            assert counts[outcome] == 1
            assert counts['pending'] == 0
            assert counts['attempted'] == (outcome != 'disabled')
            assert sum(counts[k] for k in ('used', 'error', 'timeout', 'cancelled')) == counts['attempted']
    finally:
        release.set()
        app.state.pool.shutdown()
        await upstream.aclose()
