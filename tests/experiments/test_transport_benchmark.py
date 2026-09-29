import json
import os

import httpx
import pytest
from pydantic import ValidationError

from atfm_experiments.load.transport_benchmark import TransportConfig, request, run, transport


@pytest.mark.parametrize('values', [{'concurrency': 0}, {'turns': 0}, {'repeats': 0},
                                   {'service_s': float('nan')}, {'unknown': 1}])
def test_transport_config_validation(values):
    with pytest.raises(ValidationError):
        TransportConfig(**values)


async def test_transport_request_retains_failures_instead_of_counting_success():
    async def fail(request):
        raise httpx.ConnectError('injected')
    records = []
    async with httpx.AsyncClient(base_url='http://worker', transport=httpx.MockTransport(fail)) as client:
        await request(client, records, 2, 3)
    assert records[0]['status'] is None and records[0]['error'] == 'ConnectError: injected'
    assert records[0]['lane'] == 2 and records[0]['turn'] == 3
    with pytest.raises(ValueError):
        transport('unknown')


def test_real_http_transport_comparison_retains_records_and_reaps_worker(tmp_path):
    pytest.importorskip('uvicorn')
    if os.name != 'posix':
        pytest.skip('inherited socket harness requires POSIX')
    directory = tmp_path / 'transport'
    assert run(TransportConfig(concurrency=4, turns=3, repeats=1), directory, ('default', 'shards16'))
    summaries = json.loads((directory / 'summary.json').read_text())
    assert all(r['attempted'] == r['ok'] == 12 and r['connects'] > 0 for r in summaries)
    assert len((directory / '0-shards16.jsonl').read_text().splitlines()) == 12
    assert json.loads((directory / 'shutdown.json').read_text())['worker']['exit_code'] is not None
    with pytest.raises(FileExistsError):
        run(TransportConfig(), directory)
