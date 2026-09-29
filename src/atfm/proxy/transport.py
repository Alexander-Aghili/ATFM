"""Bounded upstream HTTP pools with response-lifetime load balancing."""
from __future__ import annotations

import asyncio
from urllib.request import getproxies

import httpx


class ShardedTransport(httpx.AsyncBaseTransport):
    """Balance open response lifetimes across independent HTTP connection pools."""

    def __init__(self, shards=16, connections=100, factory=httpx.AsyncHTTPTransport):
        if not 1 <= shards <= connections:
            raise ValueError('shards must be between one and the total connection limit')
        self.limits = [connections // shards + (i < connections % shards) for i in range(shards)]
        context = httpx.create_ssl_context()
        self.pools = [factory(verify=context, limits=httpx.Limits(max_connections=n, max_keepalive_connections=n))
                      for n in self.limits]
        self.active = [0] * shards
        self.closed = False

    async def handle_async_request(self, request):
        if self.closed:
            raise RuntimeError('transport is closed')
        shard = min(range(len(self.pools)), key=lambda i: self.active[i] / self.limits[i])
        self.active[shard] += 1
        try:
            response = await self.pools[shard].handle_async_request(request)
            response.stream = _TrackedStream(response.stream, self, shard)
            return response
        except BaseException:
            self.release(shard)
            raise

    def release(self, shard):
        self.active[shard] -= 1

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        outcomes = await asyncio.gather(*(pool.aclose() for pool in self.pools), return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome


class _TrackedStream(httpx.AsyncByteStream):
    def __init__(self, stream, owner, shard):
        self.stream, self.owner, self.shard = stream, owner, shard
        self.closed = False

    async def __aiter__(self):
        async for chunk in self.stream:
            yield chunk

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        try:
            await self.stream.aclose()
        finally:
            self.owner.release(self.shard)


def upstream_client(cfg):
    """Keep HTTPX proxy discovery intact; custom pools apply only to direct traffic."""
    proxies = any(key != 'no' and value for key, value in getproxies().items())
    shards = cfg.upstream_pool_shards or (16 if cfg.window > 20 else 1)
    transport = ShardedTransport(shards) if shards > 1 and not proxies else None
    return httpx.AsyncClient(base_url=cfg.upstream_url, timeout=httpx.Timeout(600.0), transport=transport)
