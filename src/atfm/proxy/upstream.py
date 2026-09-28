"""Owned upstream HTTP transport sized for the proxy's admission window."""
import httpx

from atfm.proxy.config import ProxyConfig


def create_upstream_client(cfg: ProxyConfig) -> httpx.AsyncClient:
    keepalive = cfg.upstream_keepalive_connections
    if keepalive is None:
        keepalive = max(20, cfg.window)
    limits = httpx.Limits(max_connections=max(100, cfg.window, keepalive), max_keepalive_connections=keepalive)
    return httpx.AsyncClient(base_url=cfg.upstream_url, timeout=httpx.Timeout(600.0), limits=limits)
