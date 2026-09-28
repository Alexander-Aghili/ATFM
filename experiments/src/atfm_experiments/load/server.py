"""Child-process apps for the load harness, bound only to inherited local sockets."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager, nullcontext, suppress
import json
from pathlib import Path
import socket
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np

from atfm.board.forecaster import ExogenousModel, SessionForecaster
from atfm.board.live import LiveBoard, SessionRegistry
from atfm.board.predictors import ProgressPredictor
from atfm.board.service import create_board_app
from atfm.bus import JsonlBus
from atfm.control.gdp import GdpPlanner
from atfm.proxy.app import create_app
from atfm.proxy.config import ProxyConfig
from atfm.schema.trace import TraceRow, TraceTable
from .config import LoadConfig
from .metrics import Diagnostics, RequestTiming
from .profiling import profile_serving


def training_trace(cfg: LoadConfig) -> TraceTable:
    rows = []
    rng = np.random.default_rng(cfg.seed)
    for i in range(40):
        t = i * 10.
        duration = cfg.tool_mean_s * rng.uniform(.5, 1.5)
        end = t + cfg.worker_service_s
        common = dict(session_id=f'train-{i}', cls='interactive' if i % 4 == 0 else 'background',
                      tenant='load', isl=256, osl=16, source='synthetic-load-training')
        rows.append(TraceRow(**common, turn_index=0, t_request=t, t_first_token=end, t_last_token=end,
                             tool_name='fake-tool', backend_id='load', t_tool_start=end, t_tool_end=end + duration,
                             progress_events=[{'t': end + duration / 2, 'completed': 50, 'total': 100}]))
        rows.append(TraceRow(**common, turn_index=1, t_request=end + duration,
                             t_first_token=end + duration + cfg.worker_service_s,
                             t_last_token=end + duration + cfg.worker_service_s))
    return TraceTable.from_rows(rows)


def fake_worker(cfg: LoadConfig) -> FastAPI:
    app = FastAPI()
    worker = _FakeWorker(cfg)
    app.state.worker = worker.state
    app.get('/healthz')(worker.health)
    app.post('/v1/chat/completions')(worker.chat)
    return app


class _FakeWorker:
    def __init__(self, cfg):
        self.cfg, self.semaphore = cfg, asyncio.Semaphore(cfg.worker_slots)
        self.state = dict(received=0, waiting=0, active=0, completed=0, failed=0, max_active=0)

    async def health(self):
        return {'ok': True}

    async def chat(self, request: Request):
        body = await request.json()
        if body.get('stream'):
            return JSONResponse({'error': 'load worker supports nonstreaming requests only'}, status_code=400)
        self.state['received'] += 1
        sequence = self.state['received']
        self.state['waiting'] += 1
        acquired = False
        try:
            await self.semaphore.acquire()
            acquired = True
            return await self._serve(sequence)
        finally:
            if acquired:
                self.state['active'] -= 1
                self.semaphore.release()
            else:
                self.state['waiting'] -= 1

    async def _serve(self, sequence):
        state = self.state
        state['waiting'] -= 1
        state['active'] += 1
        state['max_active'] = max(state['max_active'], state['active'])
        await asyncio.sleep(self.cfg.worker_service_s)
        if self.cfg.worker_fail_every and sequence % self.cfg.worker_fail_every == 0:
            state['failed'] += 1
            return JSONResponse({'error': 'injected fake-worker failure'}, status_code=503)
        state['completed'] += 1
        return {'id': f'fake-{sequence}', 'object': 'chat.completion',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'synthetic response'},
                             'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 256, 'completion_tokens': 16, 'total_tokens': 272}}


def build_app(role: str, payload: dict) -> FastAPI:
    cfg = LoadConfig.model_validate(payload['config'])
    directory = Path(payload['directory'])
    if role == 'worker':
        app = fake_worker(cfg)
    elif role == 'board':
        app = _board_app(cfg, directory)
    elif role == 'proxy':
        app = _proxy_app(cfg, directory, payload)
    else:
        raise ValueError(f'unknown role {role}')
    _diagnostics(app, role, cfg, directory)
    return app


def _diagnostics(app, role, cfg, directory):
    diagnostics = Diagnostics()
    app.add_middleware(RequestTiming, diagnostics=diagnostics)
    _lifespan(app, role, cfg, directory, diagnostics)
    @app.get('/__load/metrics')
    async def metrics():
        result = diagnostics.snapshot()
        result['role'] = role
        if role == 'worker':
            result['worker'] = dict(app.state.worker)
        elif role == 'proxy':
            result['queue'] = app.state.queue.stats()
            result['predictions'] = app.state.prediction_runner.snapshot()
        else:
            result.update(_board_diagnostics(app.state))
        return result


def _board_diagnostics(state):
    reader = state.reader.status()
    t = reader['snapshot_t']
    return dict(snapshot_t=t, snapshot_age_s=None if t is None else time.time() - t,
                sessions=reader['sessions'], board_state=reader, control=state.control_worker.snapshot(),
                control_stages=state.control_worker.timings.snapshot())


def _lifespan(app, role, cfg, directory, diagnostics):
    original_lifespan = app.router.lifespan_context

    heartbeat = _heartbeat_task(diagnostics)
    @asynccontextmanager
    async def lifespan(application):
        async with original_lifespan(application):
            task = asyncio.create_task(heartbeat())
            try:
                with profile_serving(directory) if role == 'proxy' and cfg.profile_proxy else nullcontext():
                    yield
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
    app.router.lifespan_context = lifespan


def _heartbeat_task(diagnostics):
    async def heartbeat():
        while True:
            start = time.perf_counter()
            await asyncio.sleep(.05)
            diagnostics.lag.append(max(0., time.perf_counter() - start - .05))

    return heartbeat



def _proxy_app(cfg, directory, payload):
    app = create_app(ProxyConfig(upstream_url=payload['worker_url'], board_url=payload['board_url'],
                                upstream_keepalive_connections=cfg.upstream_keepalive_connections,
                                window=cfg.proxy_window, board_timeout_s=cfg.prediction_budget_s, prediction_limit=cfg.prediction_limit,
                                max_hold_s=cfg.max_hold_s, default_osl=16,
                                events_path=str(directory / 'events.jsonl'), trace_path=str(directory / 'proxy-trace.jsonl')))
    return app


def _board_app(cfg, directory):
    trace = training_trace(cfg)
    horizons = sorted({cfg.slot_s, min(3, cfg.slots) * cfg.slot_s, cfg.slots * cfg.slot_s})
    forecast = SessionForecaster(ProgressPredictor().fit(trace), ExogenousModel().fit(trace), horizons, n=cfg.draws)
    board = LiveBoard(SessionRegistry(), forecast, tick_s=cfg.control_interval_s)
    app = create_board_app(board, bus=JsonlBus(directory / 'events.jsonl'), rng=np.random.default_rng(cfg.seed),
                           gdp=GdpPlanner(slot_s=cfg.slot_s, horizon_s=cfg.slot_s * cfg.slots, max_hold_s=cfg.max_hold_s),
                           prediction_max_age_s=cfg.prediction_max_age_s,
                           capacity={'kv_blocks': cfg.capacity_kv_blocks, 'prefill_tokens': cfg.capacity_prefill_tokens})
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', required=True, choices=['worker', 'board', 'proxy'])
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--fd', type=int, required=True)
    args = parser.parse_args()
    app = build_app(args.role, json.loads(args.payload.read_text()))
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False, loop='asyncio',
                                          timeout_graceful_shutdown=2))
    with socket.socket(fileno=args.fd) as listener:
        server.run(sockets=[listener])


if __name__ == '__main__':
    main()
