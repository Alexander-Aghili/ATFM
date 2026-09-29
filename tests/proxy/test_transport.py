"""Transport ownership and balancing contracts, including cancelled streams."""
import asyncio

import httpx
import pytest

from atfm.proxy.transport import ShardedTransport


class Stream(httpx.AsyncByteStream):
    def __init__(self, error=None):
        self.closed, self.error = 0, error

    async def __aiter__(self):
        yield b'first'
        if self.error:
            raise self.error
        yield b'last'

    async def aclose(self):
        self.closed += 1


class Pool(httpx.AsyncBaseTransport):
    def __init__(self, **kwargs):
        self.options, self.requests, self.streams = kwargs, [], []
        self.closed, self.error, self.gate = 0, None, None

    async def handle_async_request(self, request):
        self.requests.append(request)
        if self.gate:
            await self.gate.wait()
        if self.error:
            raise self.error
        stream = Stream()
        self.streams.append(stream)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)

    async def aclose(self):
        self.closed += 1
        if self.error:
            raise self.error


@pytest.mark.parametrize('shards,connections', [(0, 100), (101, 100), (1, 0)])
def test_invalid_shard_capacity(shards, connections):
    with pytest.raises(ValueError):
        ShardedTransport(shards, connections, Pool)


async def test_balancing_preserves_total_capacity_and_holds_open_streams():
    transport = ShardedTransport(16, 100, Pool)
    responses = [await transport.handle_async_request(httpx.Request('GET', 'http://worker')) for _ in range(100)]
    assert transport.active == transport.limits and sum(transport.limits) == 100
    assert sum(p.options['limits'].max_connections for p in transport.pools) == 100
    await responses[0].aclose()
    replacement = await transport.handle_async_request(httpx.Request('GET', 'http://worker'))
    assert transport.active == transport.limits
    await asyncio.gather(*(r.aclose() for r in responses), replacement.aclose())
    assert not any(transport.active)
    assert all(s.closed == 1 for p in transport.pools for s in p.streams)
    await transport.aclose()


async def test_request_cancellation_and_failure_release_occupancy():
    transport = ShardedTransport(1, 1, Pool)
    pool = transport.pools[0]
    pool.gate = asyncio.Event()
    task = asyncio.create_task(transport.handle_async_request(httpx.Request('GET', 'http://worker')))
    await asyncio.sleep(0)
    assert transport.active == [1]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.active == [0]
    pool.gate, pool.error = None, httpx.ConnectError('injected')
    with pytest.raises(httpx.ConnectError):
        await transport.handle_async_request(httpx.Request('GET', 'http://worker'))
    assert transport.active == [0]
    pool.error = None
    await transport.aclose()


async def test_streaming_preserves_body_headers_and_request_extensions():
    transport = ShardedTransport(2, 4, Pool)
    async with httpx.AsyncClient(transport=transport) as client:
        async with client.stream('POST', 'http://worker/path?q=1', content=b'body', headers={'x-test': 'ok'},
                                 extensions={'example': 42}) as response:
            assert sum(transport.active) == 1
            assert response.headers['content-type'] == 'text/event-stream'
            assert await response.aread() == b'firstlast'
        assert not any(transport.active)
    request = transport.pools[0].requests[0]
    assert request.url.raw_path == b'/path?q=1' and request.content == b'body'
    assert request.headers['x-test'] == 'ok' and request.extensions['example'] == 42
    assert all(p.closed == 1 for p in transport.pools)


@pytest.mark.parametrize('failure', [httpx.ReadError('injected'), asyncio.CancelledError()])
async def test_stream_error_closes_response_and_returns_occupancy(failure):
    transport = ShardedTransport(1, 1, Pool)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(type(failure)):
            async with client.stream('GET', 'http://worker') as response:
                transport.pools[0].streams[0].error = failure
                await response.aread()
        assert transport.active == [0]
        assert transport.pools[0].streams[0].closed == 1


async def test_close_attempts_every_pool_and_rejects_new_requests():
    transport = ShardedTransport(3, 6, Pool)
    transport.pools[0].error = RuntimeError('close failed')
    with pytest.raises(RuntimeError, match='close failed'):
        await transport.aclose()
    assert all(p.closed == 1 for p in transport.pools)
    await transport.aclose()
    with pytest.raises(RuntimeError, match='transport is closed'):
        await transport.handle_async_request(httpx.Request('GET', 'http://worker'))


async def test_shared_tls_context_verifies_certificates_and_hostnames():
    import ssl
    transport = ShardedTransport(4, 8, Pool)
    contexts = [p.options['verify'] for p in transport.pools]
    assert all(c is contexts[0] for c in contexts)
    assert contexts[0].verify_mode == ssl.CERT_REQUIRED and contexts[0].check_hostname
    await transport.aclose()


@pytest.mark.parametrize('shards,proxies,sharded', [(16, {}, True), (1, {}, False),
                                                  (16, {'https': 'http://proxy'}, False),
                                                  (16, {'no': 'localhost'}, True)])
async def test_upstream_factory_preserves_stock_proxy_discovery(monkeypatch, shards, proxies, sharded):
    from atfm.proxy.config import ProxyConfig
    from atfm.proxy.transport import upstream_client
    monkeypatch.setattr('atfm.proxy.transport.getproxies', lambda: proxies)
    async with upstream_client(ProxyConfig(upstream_url='http://worker', upstream_pool_shards=shards)) as client:
        assert isinstance(client._transport, ShardedTransport) == sharded
        assert client.timeout.read == 600 and str(client.base_url) == 'http://worker'


@pytest.mark.parametrize('shards', [0, 101, True, 1.5])
def test_proxy_rejects_invalid_transport_configuration(shards):
    from pydantic import ValidationError
    from atfm.proxy.config import ProxyConfig
    with pytest.raises(ValidationError):
        ProxyConfig(upstream_url='http://worker', upstream_pool_shards=shards)


async def test_failed_stream_close_releases_occupancy_once(monkeypatch):
    transport = ShardedTransport(1, 1, Pool)
    response = await transport.handle_async_request(httpx.Request('GET', 'http://worker'))
    async def fail():
        raise httpx.ReadError('close failed')
    monkeypatch.setattr(transport.pools[0].streams[0], 'aclose', fail)
    with pytest.raises(httpx.ReadError, match='close failed'):
        await response.aclose()
    await response.aclose()
    assert transport.active == [0]
    await transport.aclose()


@pytest.mark.parametrize('window,shards', [(8, 1), (20, 1), (21, 16), (64, 16)])
async def test_automatic_transport_keeps_small_windows_on_stock_httpx(monkeypatch, window, shards):
    from atfm.proxy.config import ProxyConfig
    from atfm.proxy.transport import upstream_client
    monkeypatch.setattr('atfm.proxy.transport.getproxies', lambda: {})
    async with upstream_client(ProxyConfig(upstream_url='http://worker', window=window)) as client:
        if shards == 1:
            assert isinstance(client._transport, httpx.AsyncHTTPTransport)
        else:
            assert isinstance(client._transport, ShardedTransport)
            assert len(client._transport.pools) == shards
