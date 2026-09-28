import asyncio
import json
import os
import subprocess

import httpx
import pytest
from pydantic import ValidationError

from atfm_experiments.load.config import LoadConfig
from atfm_experiments.load.metrics import distribution
from atfm_experiments.load.server import fake_worker
from atfm_experiments.load.workload import tool_deadline


@pytest.mark.parametrize('overrides', [{'sessions': 0}, {'turns': 0}, {'arrival_window_s': 0},
                                      {'worker_service_s': float('nan')}, {'draws': -1},
                                      {'sessions': 100000, 'turns': 100}, {'unknown': 1}])
def test_load_config_rejects_invalid_or_unbounded_runs(overrides):
    with pytest.raises(ValidationError):
        LoadConfig(**overrides)


def test_tool_bursts_use_shared_boundaries_and_never_finish_early():
    assert tool_deadline(10.1, .2, 'staggered', 10, 1) == pytest.approx(10.3)
    assert tool_deadline(10.1, .2, 'burst', 10, 1) == 11
    assert tool_deadline(10.4, .2, 'burst', 10, 1) == 11
    assert tool_deadline(10.9, .2, 'burst', 10, 1) == 12
    assert distribution([])['p99'] is None
    assert distribution([1, 2, 3])['p50'] == 2


async def test_fake_worker_enforces_concurrency_and_injects_failures():
    app = fake_worker(LoadConfig(worker_slots=2, worker_service_s=.005, worker_fail_every=3))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://worker') as client:
        responses = await asyncio.gather(*(client.post('/v1/chat/completions', json={}) for _ in range(9)))
        assert sum(r.status_code == 503 for r in responses) == 3
        assert (await client.post('/v1/chat/completions', json={'stream': True})).status_code == 400
    assert app.state.worker == {'received': 9, 'waiting': 0, 'active': 0, 'completed': 6, 'failed': 3, 'max_active': 2}


async def test_fake_worker_releases_slots_and_waiters_on_cancellation():
    app = fake_worker(LoadConfig(worker_slots=1, worker_service_s=10))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://worker') as client:
        tasks = [asyncio.create_task(client.post('/v1/chat/completions', json={})) for _ in range(2)]
        await asyncio.sleep(.01)
        assert app.state.worker['active'] == app.state.worker['waiting'] == 1
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert app.state.worker['active'] == app.state.worker['waiting'] == 0


def require_servers():
    pytest.importorskip('uvicorn')
    if os.name != 'posix':
        pytest.skip('local harness uses POSIX inherited sockets')


@pytest.mark.parametrize('pattern,keepalive', [('staggered', 0), ('burst', 4)])
def test_real_http_stack_runs_agent_turns_and_control_and_stops_children(tmp_path, pattern, keepalive):
    require_servers()
    from atfm_experiments.load.__main__ import run_case
    from atfm.bus import read_events

    cfg = LoadConfig(sessions=4, turns=2, arrival_window_s=.04, pattern=pattern, worker_slots=1,
                     worker_service_s=.01, tool_mean_s=.08, burst_period_s=.1, draws=4,
                     control_interval_s=.025, monitor_interval_s=.025, max_hold_s=.01, client_keepalive_connections=keepalive)
    directory = tmp_path / 'case'
    result = run_case(cfg, directory)
    assert result['requests_ok'] == result['requests_attempted'] == 8
    assert result['client_errors'] == {} and result['session_errors'] == []
    assert result['client_to_headers_sent_s']['count'] == 8
    assert result['client_to_proxy_timestamp_s']['count'] == 8
    requests = [json.loads(line) for line in (directory / 'requests.jsonl').read_text().splitlines()]
    assert all('http11.receive_response_headers.complete' in r['transport_s'] for r in requests)
    assert result['control_steps'] >= 2 and result['control_errors'] == 0
    assert result['worker_active_peak'] == 1
    assert result['tool_publish_errors'] == 0
    assert result['final_observations']['proxy']['metrics']['queue']['in_flight'] == 0
    assert len(json.loads((directory / 'tools.json').read_text())) == 4
    events = read_events(directory / 'events.jsonl')
    assert sum(e.kind == 'llm.request' for e in events) == 8
    assert sum(e.kind == 'tool.end' for e in events) == 4
    assert sum(e.kind == 'session.start' for e in events) == 4
    shutdown = json.loads((directory / 'shutdown.json').read_text())
    assert set(shutdown) == {'worker', 'board', 'proxy'}
    assert all(p['exit_code'] is not None for p in shutdown.values())
    with pytest.raises(FileExistsError):
        run_case(cfg, directory)


def test_startup_failure_still_reaps_started_children(tmp_path, monkeypatch):
    require_servers()
    from atfm_experiments.load.runtime import local_stack

    popen = subprocess.Popen
    def fail_board(command, **kwargs):
        if '--role' in command and command[command.index('--role') + 1] == 'board':
            command = [command[0], '-c', 'raise SystemExit(7)']
        return popen(command, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', fail_board)
    with pytest.raises(RuntimeError, match='board exited'):
        with local_stack(LoadConfig(), tmp_path):
            pytest.fail('startup unexpectedly succeeded')
    shutdown = json.loads((tmp_path / 'shutdown.json').read_text())
    assert shutdown['board']['exit_code'] == 7
    assert shutdown['worker']['exit_code'] is not None


def test_http_failures_are_reported_without_losing_request_records(tmp_path):
    require_servers()
    from atfm_experiments.load.__main__ import run_case

    result = run_case(LoadConfig(sessions=6, turns=1, arrival_window_s=.04, worker_slots=2,
                                worker_service_s=.005, worker_fail_every=2, draws=4), tmp_path / 'failure')
    assert result['requests_attempted'] == 6 and result['requests_ok'] == 3
    assert result['status_counts'] == {'200': 3, '503': 3}
    assert result['proxy_traces_available'] == 6
    assert result['final_observations']['proxy']['metrics']['queue']['in_flight'] == 0


def test_drain_deadline_reports_incomplete_sessions_and_reaps_children(tmp_path):
    require_servers()
    from atfm_experiments.load.__main__ import run_case

    directory = tmp_path / 'deadline'
    result = run_case(LoadConfig(sessions=2, turns=2, arrival_window_s=.01, drain_timeout_s=.3,
                                worker_service_s=.001, tool_mean_s=10, draws=4), directory)
    assert result['drain_deadline_reached'] and result['cancelled_sessions'] == 2
    assert result['requests_ok'] == 2 and result['maximum_requests'] == 4
    assert all(p['exit_code'] is not None for p in json.loads((directory / 'shutdown.json').read_text()).values())


def test_attribution_pairs_seeds_and_alternates_order():
    from atfm_experiments.load.attribution import cases

    planned = list(cases(LoadConfig(seed=7), [4, 4], 2))
    assert len({name for name, _ in planned}) == 4
    assert [(c.seed, c.control_enabled) for _, c in planned] == [(7, True), (7, False), (8, False), (8, True)]
    with pytest.raises(ValueError):
        list(cases(LoadConfig(), [4], 0))


def test_control_off_keeps_predictions_but_never_ticks_board(tmp_path):
    require_servers()
    from atfm_experiments.load.__main__ import run_case

    result = run_case(LoadConfig(sessions=4, turns=2, arrival_window_s=.04, worker_service_s=.005,
                                tool_mean_s=.02, draws=4, control_enabled=False), tmp_path / 'off')
    assert result['requests_ok'] == 8
    assert result['control_steps'] == result['holds_applied'] == result['control_http_errors'] == 0
    assert result['final_observations']['board']['metrics']['snapshot_t'] is None
    counts = result['final_observations']['proxy']['metrics']['predictions']
    assert counts['attempted'] == 8 and counts['pending'] == 0
    assert counts['used'] + counts['error'] + counts['timeout'] == 8


def test_provenance_records_inherited_descriptor_limits():
    import resource
    from atfm_experiments.load.runtime import provenance

    assert provenance()['rlimit_nofile'] == dict(zip(('soft', 'hard'), resource.getrlimit(resource.RLIMIT_NOFILE)))


def test_profile_is_saved_after_real_http_shutdown(tmp_path):
    import pstats
    pytest.importorskip('yappi')
    require_servers()
    from atfm_experiments.load.__main__ import run_case

    directory = tmp_path / 'profile'
    result = run_case(LoadConfig(sessions=4, turns=1, arrival_window_s=.04,
                                worker_service_s=.005, control_enabled=False, profile_proxy=True), directory)
    assert result['requests_ok'] == 4
    stats = pstats.Stats(str(directory / 'proxy.pstats'))
    assert any(file.endswith('/atfm/proxy/app.py') and function == 'chat'
               for file, _, function in stats.stats)
    assert 'primitive_calls' in (directory / 'proxy-profile.csv').read_text()
    assert 'Thread' in (directory / 'proxy-profile.txt').read_text()
    metadata = json.loads((directory / 'proxy-profile.json').read_text())
    assert metadata['clock'] == 'cpu'
    assert any(t['name'] == '_MainThread' for t in metadata['threads'])
    assert any(t['name'] != '_MainThread' for t in metadata['threads'])
    assert not (directory / 'board.pstats').exists()
