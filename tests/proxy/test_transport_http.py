"""Real socket checks for partial streaming response cleanup and pool reuse."""
import asyncio
from contextlib import asynccontextmanager
import socket

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from atfm.proxy.transport import ShardedTransport


@asynccontextmanager
async def origin():
    uvicorn = pytest.importorskip('uvicorn')
    app, released = streaming_app()
    server = uvicorn.Server(uvicorn.Config(app, log_level='critical', lifespan='off'))
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    await asyncio.sleep(.001)
            yield f'http://127.0.0.1:{listener.getsockname()[1]}', released
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 5)


def streaming_app():
    app, released = FastAPI(), asyncio.Event()
    async def chunks():
        try:
            yield b'first\n'
            await asyncio.Event().wait()
        finally:
            released.set()
    @app.get('/stream')
    async def stream():
        return StreamingResponse(chunks())
    @app.get('/fast')
    async def fast():
        return {'ok': True}
    return app, released


async def test_partial_real_http_stream_does_not_leak_balancing_occupancy():
    async with origin() as (url, released):
        transport = ShardedTransport(shards=2, connections=2)
        async with httpx.AsyncClient(base_url=url, transport=transport, trust_env=False) as client:
            async with client.stream('GET', '/stream') as response:
                assert await anext(response.aiter_bytes()) == b'first\n'
                assert sum(transport.active) == 1
                for _ in range(8):
                    assert (await client.get('/fast')).json() == {'ok': True}
                assert sum(transport.active) == 1
            await asyncio.wait_for(released.wait(), 2)
            assert not any(transport.active)
            assert (await client.get('/fast')).status_code == 200
        assert transport.closed


async def test_real_http_connections_are_reused_without_exceeding_capacity():
    async with origin() as (url, _):
        transport, connects = ShardedTransport(shards=2, connections=4), []
        async def trace(name, info):
            if name == 'connection.connect_tcp.complete':
                connects.append(name)
        async with httpx.AsyncClient(base_url=url, transport=transport, trust_env=False) as client:
            for _ in range(3):
                responses = await asyncio.gather(*(client.get('/fast', extensions={'trace': trace}) for _ in range(4)))
                assert all(r.status_code == 200 for r in responses)
                assert not any(transport.active)
        assert 1 <= len(connects) <= 4
