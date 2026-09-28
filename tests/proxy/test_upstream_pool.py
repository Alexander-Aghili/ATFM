"""Exercise connection reuse over real HTTP/1.1, not private HTTPX internals."""
import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest

from atfm.proxy.config import ProxyConfig
from atfm.proxy.upstream import create_upstream_client


class KeepaliveServer:
    def __init__(self, width):
        self.width, self.connections, self.requests = width, 0, 0
        self.ready, self.handlers = asyncio.Event(), set()

    async def handle(self, reader, writer):
        self.connections += 1
        task = asyncio.current_task()
        self.handlers.add(task)
        try:
            while True:
                await reader.readuntil(b'\r\n\r\n')
                await self.wait_for_wave()
                writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok')
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            self.handlers.remove(task)

    async def wait_for_wave(self):
        ready = self.ready
        self.requests += 1
        if self.requests % self.width == 0:
            self.ready = asyncio.Event()
            ready.set()
        await ready.wait()

    @asynccontextmanager
    async def running(self):
        server = await asyncio.start_server(self.handle, '127.0.0.1', 0)
        try:
            yield f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}'
        finally:
            self.ready.set()
            server.close()
            await server.wait_closed()
            await asyncio.gather(*self.handlers)


@pytest.mark.parametrize('keepalive,extra_connections', [(None, False), (0, True), (20, True)])
async def test_pool_reuses_admission_window_and_honors_override(keepalive, extra_connections):
    server = KeepaliveServer(32)
    async with server.running() as url:
        cfg = ProxyConfig(upstream_url=url, window=32, upstream_keepalive_connections=keepalive)
        async with create_upstream_client(cfg) as client:
            async with asyncio.timeout(5):
                for _ in range(2):
                    responses = await asyncio.gather(*(client.get('/') for _ in range(32)))
                    assert all(r.text == 'ok' for r in responses)
        assert server.requests == 64
        assert (server.connections > 32) == extra_connections
        if not extra_connections:
            assert server.connections == 32


@pytest.mark.parametrize('value', [-1, 1.5, True])
def test_keepalive_configuration_rejects_invalid_counts(value):
    with pytest.raises(ValueError):
        ProxyConfig(upstream_url='http://up', upstream_keepalive_connections=value)


async def test_injected_client_remains_unchanged_and_caller_owned():
    from atfm.proxy.app import create_app
    async with httpx.AsyncClient() as injected:
        app = create_app(ProxyConfig(upstream_url='http://up', window=64), upstream_client=injected)
        async with app.router.lifespan_context(app):
            assert app.state.client is injected
        assert not injected.is_closed
