"""Scheduled session arrivals, agent turns, observations, and control-loop trials."""
from __future__ import annotations

import asyncio
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import httpx

from atfm.bus import JsonlBus
from atfm.control.loop import ControlLoop
from atfm.schema.events import ToolStart, ToolProgress, ToolEnd
from .config import LoadConfig
from .metrics import distribution
from .runtime import write_json


def append_record(stream, record) -> None:
    stream.write(json.dumps(record, allow_nan=False) + '\n')
    stream.flush()


def tool_deadline(start: float, duration: float, pattern: str, epoch: float, period: float) -> float:
    end = start + duration
    return epoch + math.ceil((end - epoch) / period) * period if pattern == 'burst' else end


class TimedControlClient:
    def __init__(self, stream, timeout: float):
        self.client = httpx.Client(timeout=timeout, trust_env=False)
        self.stream = stream
        self.errors = 0

    def post(self, url, **kwargs):
        start = time.perf_counter()
        status, error = None, None
        try:
            response = self.client.post(url, **kwargs)
            status = response.status_code
            return response
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            self.errors += status is None or status >= 400
            append_record(self.stream, {'t': time.time(), 'url': url, 'duration_s': time.perf_counter() - start,
                                        'status': status, 'error': error})


async def exercise(cfg: LoadConfig, endpoints: dict[str, str], directory: Path) -> dict:
    loop = asyncio.get_running_loop()
    origin = loop.time()
    wall_origin = time.time()
    stop = asyncio.Event()
    bus = JsonlBus(directory / 'events.jsonl')
    requests, controls, observations, arrivals, tools = [], [], [], [], []
    client_limits = httpx.Limits(max_connections=cfg.sessions, max_keepalive_connections=min(cfg.client_keepalive_connections, cfg.sessions))

    async def pause_until(deadline):
        await asyncio.sleep(max(0., deadline - loop.time()))

    with (directory / 'requests.jsonl').open('w') as request_log, \
         (directory / 'control-http.jsonl').open('w') as control_http, \
         (directory / 'control-steps.jsonl').open('w') as control_steps, \
         (directory / 'observations.jsonl').open('w') as observation_log:
        control_client = TimedControlClient(control_http, cfg.control_timeout_s)
        control = ControlLoop(endpoints['board'], endpoints['proxy'], client=control_client,
                              interval_s=cfg.control_interval_s, log_path=directory / 'control.jsonl')

        async def control_task():
            while not stop.is_set():
                start = loop.time()
                result = await asyncio.to_thread(control.step)
                record = {'t': time.time(), 'duration_s': loop.time() - start, **result}
                controls.append(record)
                append_record(control_steps, record)
                try:
                    await asyncio.wait_for(stop.wait(), cfg.control_interval_s)
                except TimeoutError:
                    pass

        async with httpx.AsyncClient(limits=client_limits, timeout=cfg.request_timeout_s, trust_env=False) as client, \
                         httpx.AsyncClient(timeout=min(2., cfg.request_timeout_s), trust_env=False) as probe:
            async def observe():
                async def one(role, url):
                    start = loop.time()
                    record = {'t': time.time(), 'role': role}
                    try:
                        response = await probe.get(url + '/__load/metrics')
                        response.raise_for_status()
                        record['metrics'] = response.json()
                    except (httpx.HTTPError, ValueError) as exc:
                        record['error'] = type(exc).__name__
                    record['probe_s'] = loop.time() - start
                    observations.append(record)
                    append_record(observation_log, record)
                await asyncio.gather(*(one(role, url) for role, url in endpoints.items()))

            async def monitor():
                while not stop.is_set():
                    await observe()
                    try:
                        await asyncio.wait_for(stop.wait(), cfg.monitor_interval_s)
                    except TimeoutError:
                        pass

            async def session(index):
                rng = random.Random(cfg.seed + index)
                scheduled = origin + index * cfg.arrival_window_s / cfg.sessions
                await pause_until(scheduled)
                sid = f'load-{index}'
                cls = 'interactive' if rng.random() < cfg.interactive_fraction else 'background'
                arrivals.append({'session_id': sid, 'scheduled_s': scheduled - origin,
                                 'started_s': loop.time() - origin, 'lateness_s': max(0., loop.time() - scheduled)})
                for turn in range(cfg.turns):
                    start = loop.time()
                    record = {'session_id': sid, 'turn': turn, 'class': cls, 't': time.time(),
                              'status': None, 'error': None, 'transport_s': {}}

                    async def trace(name, info):
                        if name.endswith(('.started', '.complete', '.failed')):
                            record['transport_s'][name] = loop.time() - start

                    try:
                        response = await client.post(endpoints['proxy'] + '/v1/chat/completions',
                                                     json={'model': 'fake', 'messages': [{'role': 'user', 'content': 'x' * 1024}],
                                                           'max_tokens': 16, 'stream': False},
                                                     headers={'x-atfm-session': sid, 'x-atfm-class': cls, 'x-atfm-tenant': 'load'},
                                                     extensions={'trace': trace})
                        record['status'] = response.status_code
                    except httpx.HTTPError as exc:
                        record['error'] = type(exc).__name__
                    except asyncio.CancelledError:
                        record['error'] = 'drain_deadline'
                        raise
                    finally:
                        record['duration_s'] = loop.time() - start
                        requests.append(record)
                        append_record(request_log, record)
                    if record['status'] != 200 or turn + 1 == cfg.turns:
                        return
                    call_id = f'{sid}-tool-{turn}'
                    started = loop.time()
                    end = tool_deadline(started, cfg.tool_mean_s * rng.uniform(.5, 1.5),
                                        cfg.pattern, origin, cfg.burst_period_s)
                    bus.publish(ToolStart(t=time.time(), session_id=sid, turn_index=turn, call_id=call_id,
                                          tool_name='fake-tool', backend_id='load'))
                    await pause_until((started + end) / 2)
                    bus.publish(ToolProgress(t=time.time(), session_id=sid, call_id=call_id, completed=50, total=100))
                    await pause_until(end)
                    tools.append({'session_id': sid, 'call_id': call_id, 'scheduled_end_s': end - origin,
                                  'end_s': loop.time() - origin, 'lateness_s': max(0., loop.time() - end)})
                    bus.publish(ToolEnd(t=time.time(), session_id=sid, call_id=call_id, exit_status=0))

            origin = loop.time() + .05
            wall_origin = time.time() + .05
            controller = asyncio.create_task(control_task())
            monitoring = asyncio.create_task(monitor())
            sessions = [asyncio.create_task(session(i)) for i in range(cfg.sessions)]
            timed_out = False
            try:
                _, pending = await asyncio.wait(sessions, timeout=cfg.arrival_window_s + cfg.drain_timeout_s)
                timed_out = bool(pending)
                for task in pending:
                    task.cancel()
                results = await asyncio.gather(*sessions, return_exceptions=True)
                workload_elapsed = max(0., loop.time() - origin)
                cancelled_sessions = sum(isinstance(result, asyncio.CancelledError) for result in results)
                failures = [repr(result) for result in results if isinstance(result, Exception)]
            finally:
                for task in sessions:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*sessions, return_exceptions=True)
                stop.set()
                await asyncio.gather(controller, monitoring)
                await observe()
                control_client.client.close()
    elapsed = loop.time() - origin
    write_json(directory / 'arrivals.json', arrivals)
    write_json(directory / 'tools.json', tools)
    traces = [json.loads(line) for line in (directory / 'proxy-trace.jsonl').read_text().splitlines()]
    successful = [r for r in requests if r['status'] == 200]
    by_request = {(t['session_id'], t['turn_index']): t for t in traces}
    paired = [(r, by_request[(r['session_id'], r['turn'])]) for r in requests
              if (r['session_id'], r['turn']) in by_request]
    metrics = [o for o in observations if 'metrics' in o]
    def maximum(role, section, key):
        return max((o['metrics'][section][key] for o in metrics if o['role'] == role), default=0)
    report = {'config': cfg.model_dump(), 'wall_start': wall_origin, 'elapsed_s': elapsed,
              'workload_elapsed_s': workload_elapsed, 'cancelled_sessions': cancelled_sessions,
              'proxy_traces_available': len(traces),
              'scheduled_session_rate_s': cfg.sessions / cfg.arrival_window_s,
              'sessions_started': len(arrivals), 'maximum_requests': cfg.sessions * cfg.turns,
              'requests_attempted': len(requests), 'requests_ok': len(successful),
              'status_counts': dict(Counter(str(r['status']) for r in requests)),
              'client_errors': dict(Counter(r['error'] for r in requests if r['error'])),
              'drain_deadline_reached': timed_out, 'session_errors': failures,
              'completed_request_rate_s': len(successful) / max(workload_elapsed, 1e-9),
              'client_duration_s': distribution(r['duration_s'] for r in requests),
              'client_to_headers_sent_s': distribution(r['transport_s']['http11.send_request_headers.complete']
                                                       for r in requests if 'http11.send_request_headers.complete' in r['transport_s']),
              'client_to_proxy_timestamp_s': distribution(t['t_request'] - r['t'] for r, t in paired),
              'proxy_upstream_s': distribution(t['t_last_token'] - t['t_release'] for t in traces),
              'successful_client_duration_s': distribution(r['duration_s'] for r in successful),
              'proxy_arrival_to_release_s': distribution(t['t_release'] - t['t_request'] for t in traces),
              'session_launch_lateness_s': distribution(r['lateness_s'] for r in arrivals),
              'tool_completion_lateness_s': distribution(r['lateness_s'] for r in tools),
              'control_http_errors': control_client.errors,
              'class_results': {cls: {'requests_ok': sum(r['status'] == 200 for r in requests if r['class'] == cls),
                                      'client_duration_s': distribution(r['duration_s'] for r in requests if r['class'] == cls)}
                                for cls in ('interactive', 'background')},
              'control_steps': len(controls), 'control_errors': sum(r['errors'] for r in controls),
              'holds_applied': sum(r['holds'] for r in controls),
              'control_step_s': distribution(r['duration_s'] for r in controls),
              'proxy_queue_peak_observed': maximum('proxy', 'queue', 'queued'),
              'worker_waiting_peak_observed': maximum('worker', 'worker', 'waiting'),
              'worker_active_peak': maximum('worker', 'worker', 'max_active'),
              'probe_errors': sum('error' in o for o in observations), 'tool_publish_errors': bus.errors,
              'final_observations': {role: next((o for o in reversed(observations) if o['role'] == role), None)
                                     for role in endpoints}}
    write_json(directory / 'summary.json', report)
    return report
