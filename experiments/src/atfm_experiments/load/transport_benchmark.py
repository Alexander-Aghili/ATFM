"""Closed-loop real-HTTP transport trials; no board, proxy or external service."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import time

import httpx
from pydantic import BaseModel, Field

from .config import LoadConfig
from .metrics import distribution
from .runtime import _launch, _shutdown, _wait_ready, provenance, write_json
from .transport_pool import ShardedTransport


class TransportConfig(BaseModel):
    model_config = {'extra': 'forbid'}
    concurrency: int = Field(default=64, ge=1, le=256)
    turns: int = Field(default=48, ge=1, le=1000)
    service_s: float = Field(default=.005, ge=0, le=10, allow_inf_nan=False)
    repeats: int = Field(default=3, ge=1, le=20)


@contextmanager
def worker(cfg, directory):
    load = LoadConfig(worker_slots=cfg.concurrency, worker_service_s=cfg.service_s)
    payload = dict(config=load.model_dump(), directory=str(directory.resolve()))
    with ExitStack() as stack:
        process, url = _launch('worker', directory, payload, dict(os.environ, NO_PROXY='127.0.0.1'), stack)
        try:
            _wait_ready('worker', directory, process, url)
            yield url
        finally:
            _shutdown(directory, {'worker': process})


def transport(mode):
    if mode == 'default':
        return httpx.AsyncHTTPTransport()
    if mode == 'retained':
        return httpx.AsyncHTTPTransport(limits=httpx.Limits(max_connections=100, max_keepalive_connections=100))
    if mode in ('shards8', 'shards16'):
        return ShardedTransport(shards=int(mode.removeprefix('shards')))
    raise ValueError(f'unknown transport {mode}')


async def request(client, records, lane, turn):
    started, events = time.perf_counter(), {}
    async def trace(name, info):
        events[name] = events.get(name, 0) + 1
    row = dict(lane=lane, turn=turn, status=None, error=None)
    try:
        response = await client.post('/v1/chat/completions', json={}, extensions={'trace': trace})
        row['status'] = response.status_code
    except httpx.HTTPError as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    row.update(duration_s=time.perf_counter() - started, transport_events=events)
    records.append(row)


async def trial(url, cfg, mode):
    records = []
    async with httpx.AsyncClient(base_url=url, timeout=30, transport=transport(mode), trust_env=False) as client:
        async def lane(i):
            for turn in range(cfg.turns):
                await request(client, records, i, turn)
        started, cpu = time.perf_counter(), time.process_time()
        await asyncio.gather(*(lane(i) for i in range(cfg.concurrency)))
        summary = dict(mode=mode, wall_s=time.perf_counter() - started, cpu_s=time.process_time() - cpu)
    summary.update(attempted=len(records), ok=sum(r['status'] == 200 for r in records),
                   connects=sum(r['transport_events'].get('connection.connect_tcp.complete', 0) for r in records),
                   latency_s=distribution(r['duration_s'] for r in records))
    return summary, records


def run(cfg, directory, modes=('default', 'retained', 'shards8', 'shards16')):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'config.json', cfg.model_dump())
    write_json(directory / 'environment.json', provenance())
    results = []
    with worker(cfg, directory) as url:
        for repeat in range(cfg.repeats):
            for mode in modes if repeat % 2 == 0 else reversed(modes):
                summary, records = asyncio.run(trial(url, cfg, mode))
                summary['repeat'] = repeat
                results.append(summary)
                write_json(directory / f'{repeat}-{mode}.json', summary)
                (directory / f'{repeat}-{mode}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                print(json.dumps(summary), flush=True)
    write_json(directory / 'summary.json', results)
    return all(r['ok'] == cfg.concurrency * cfg.turns for r in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=64)
    parser.add_argument('--turns', type=int, default=48)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    cfg = TransportConfig(concurrency=args.concurrency, turns=args.turns, repeats=args.repeats)
    raise SystemExit(0 if run(cfg, args.out) else 1)


if __name__ == '__main__':
    main()
