"""Scheduled session arrivals, agent turns, observations, and control-loop trials."""
from __future__ import annotations

import asyncio
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
from .report import write_report
from contextlib import ExitStack


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
    return await _LoadTrial(cfg, endpoints, directory).run()


class _LoadTrial:
    def __init__(self, cfg, endpoints, directory):
        self.cfg, self.endpoints, self.directory = cfg, endpoints, directory
        self.loop = asyncio.get_running_loop()
        self.origin, self.wall_origin = self.loop.time(), time.time()
        self.stop = asyncio.Event()
        self.bus = JsonlBus(directory / 'events.jsonl')
        self.requests, self.controls, self.observations, self.arrivals, self.tools = [], [], [], [], []

    async def run(self):
        limits = httpx.Limits(max_connections=self.cfg.sessions,
                             max_keepalive_connections=min(self.cfg.client_keepalive_connections, self.cfg.sessions))
        with ExitStack() as stack:
            self._open_logs(stack)
            async with httpx.AsyncClient(limits=limits, timeout=self.cfg.request_timeout_s, trust_env=False) as client, \
                    httpx.AsyncClient(timeout=min(2., self.cfg.request_timeout_s), trust_env=False) as probe:
                self.client, self.probe = client, probe
                await self._run_sessions()
        return write_report(self)

    def _open_logs(self, stack):
        names = {'request_log': 'requests.jsonl', 'control_steps': 'control-steps.jsonl',
                 'observation_log': 'observations.jsonl', 'control_http': 'control-http.jsonl'}
        for attr, filename in names.items():
            setattr(self, attr, stack.enter_context((self.directory / filename).open('w')))
        self.control_client = TimedControlClient(self.control_http, self.cfg.control_timeout_s)
        self.control = ControlLoop(self.endpoints['board'], self.endpoints['proxy'], client=self.control_client,
                                  interval_s=self.cfg.control_interval_s, log_path=self.directory / 'control.jsonl')

    async def _run_sessions(self):
        self.origin = self.loop.time() + .05
        self.wall_origin = time.time() + .05
        controller = asyncio.create_task(self.control_task())
        monitoring = asyncio.create_task(self.monitor())
        sessions = [asyncio.create_task(self.session(i)) for i in range(self.cfg.sessions)]
        self.timed_out = False
        try:
            _, pending = await asyncio.wait(sessions, timeout=self.cfg.arrival_window_s + self.cfg.drain_timeout_s)
            self.timed_out = bool(pending)
            for task in pending:
                task.cancel()
            results = await asyncio.gather(*sessions, return_exceptions=True)
            self.workload_elapsed = max(0., self.loop.time() - self.origin)
            self.cancelled_sessions = sum(isinstance(result, asyncio.CancelledError) for result in results)
            self.failures = [repr(result) for result in results if isinstance(result, Exception)]
        finally:
            await self._finish_sessions(sessions, controller, monitoring)


    async def _finish_sessions(self, sessions, controller, monitoring):
        for task in sessions:
            if not task.done():
                task.cancel()
        await asyncio.gather(*sessions, return_exceptions=True)
        self.stop.set()
        await asyncio.gather(controller, monitoring)
        await self.observe()
        self.control_client.client.close()

    async def one(self, role, url):
        start = self.loop.time()
        record = {'t': time.time(), 'role': role}
        try:
            response = await self.probe.get(url + '/__load/metrics')
            response.raise_for_status()
            record['metrics'] = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            record['error'] = type(exc).__name__
        record['probe_s'] = self.loop.time() - start
        self.observations.append(record)
        append_record(self.observation_log, record)

    async def pause_until(self, deadline):
        await asyncio.sleep(max(0., deadline - self.loop.time()))

    async def session(self, index):
        rng = random.Random(self.cfg.seed + index)
        scheduled = self.origin + index * self.cfg.arrival_window_s / self.cfg.sessions
        await self.pause_until(scheduled)
        sid = f'load-{index}'
        cls = 'interactive' if rng.random() < self.cfg.interactive_fraction else 'background'
        self.arrivals.append({'session_id': sid, 'scheduled_s': scheduled - self.origin,
                         'started_s': self.loop.time() - self.origin, 'lateness_s': max(0., self.loop.time() - scheduled)})
        for turn in range(self.cfg.turns):
            record = await self._request(sid, cls, turn)
            if record['status'] != 200 or turn + 1 == self.cfg.turns:
                return
            await self._tool(sid, turn, rng)


    async def _tool(self, sid, turn, rng):
        call_id = f'{sid}-tool-{turn}'
        started = self.loop.time()
        end = tool_deadline(started, self.cfg.tool_mean_s * rng.uniform(.5, 1.5),
                            self.cfg.pattern, self.origin, self.cfg.burst_period_s)
        self.bus.publish(ToolStart(t=time.time(), session_id=sid, turn_index=turn, call_id=call_id,
                              tool_name='fake-tool', backend_id='load'))
        await self.pause_until((started + end) / 2)
        self.bus.publish(ToolProgress(t=time.time(), session_id=sid, call_id=call_id, completed=50, total=100))
        await self.pause_until(end)
        self.tools.append({'session_id': sid, 'call_id': call_id, 'scheduled_end_s': end - self.origin,
                      'end_s': self.loop.time() - self.origin, 'lateness_s': max(0., self.loop.time() - end)})
        self.bus.publish(ToolEnd(t=time.time(), session_id=sid, call_id=call_id, exit_status=0))


    async def _request(self, sid, cls, turn):
        start = self.loop.time()
        record = {'session_id': sid, 'turn': turn, 'class': cls, 't': time.time(),
                  'status': None, 'error': None, 'transport_s': {}}

        async def trace(name, info):
            if name.endswith(('.started', '.complete', '.failed')):
                record['transport_s'][name] = self.loop.time() - start

        await self._exchange(sid, cls, record, start, trace)
        return record


    async def _exchange(self, sid, cls, record, start, trace):
        try:
            response = await self.client.post(self.endpoints['proxy'] + '/v1/chat/completions',
                                         json={'model': 'fake', 'messages': [{'role': 'user', 'content': 'x' * 1024}],
                                               'max_tokens': 16, 'stream': False},
                                         headers={'x-atfm-session': sid, 'x-atfm-class': cls, 'x-atfm-tenant': 'load'},
                                         extensions={'trace': trace})
            record['status'] = response.status_code
        except httpx.HTTPError as exc:
            record['error'] = type(exc).__name__
            record['error_detail'] = str(exc)
        except asyncio.CancelledError:
            record['error'] = 'drain_deadline'
            raise
        finally:
            record['duration_s'] = self.loop.time() - start
            self.requests.append(record)
            append_record(self.request_log, record)

    async def observe(self):
        await asyncio.gather(*(self.one(role, url) for role, url in self.endpoints.items()))

    async def control_task(self):
        if not self.cfg.control_enabled:
            return
        while not self.stop.is_set():
            start = self.loop.time()
            result = await asyncio.to_thread(self.control.step)
            record = {'t': time.time(), 'duration_s': self.loop.time() - start, **result}
            self.controls.append(record)
            append_record(self.control_steps, record)
            try:
                await asyncio.wait_for(self.stop.wait(), self.cfg.control_interval_s)
            except TimeoutError:
                pass

    async def monitor(self):
        while not self.stop.is_set():
            await self.observe()
            try:
                await asyncio.wait_for(self.stop.wait(), self.cfg.monitor_interval_s)
            except TimeoutError:
                pass
