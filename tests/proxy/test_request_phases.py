"""Trace phase boundaries separate prediction, admission, and dispatch waits."""
import asyncio
import json

import httpx
import pytest

from atfm.proxy.app import create_app
from atfm.proxy.config import ProxyConfig
from atfm_experiments.load.report import _request_phases


async def test_trace_records_prediction_and_queue_boundaries(tmp_path):
    path = tmp_path / 'trace.jsonl'
    async def upstream(request):
        await asyncio.sleep(.01)
        return httpx.Response(200, json={'usage': {'completion_tokens': 1}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        app = create_app(ProxyConfig(upstream_url='http://up', window=1, trace_path=str(path)), upstream_client=client)
        client.base_url = 'http://up'
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://p') as caller:
                results = await asyncio.gather(*(caller.post('/v1/chat/completions', json={}) for _ in range(2)))
                assert all(r.status_code == 200 for r in results)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        assert row['prediction_s'] >= 0
        assert row['t_request'] <= row['t_enqueued'] <= row['t_admitted'] <= row['t_release'] <= row['t_last_token']
    assert max(r['t_admitted'] - r['t_enqueued'] for r in rows) >= .005


def test_phase_summary_preserves_legacy_traces_without_inventing_zeroes():
    legacy = dict(t_request=0., t_release=1.)
    current = dict(t_enqueued=.2, t_admitted=.7, t_release=.9, prediction_s=.1)
    metrics = _request_phases([legacy, current])
    assert metrics['proxy_prediction_wait_s']['p50'] == .1
    assert metrics['proxy_admission_wait_s']['p50'] == pytest.approx(.5)
    assert metrics['proxy_release_dispatch_s']['p50'] == pytest.approx(.2)
    assert all(metric['count'] == 1 for metric in metrics.values())
    assert _request_phases([legacy])['proxy_prediction_wait_s']['p95'] is None
