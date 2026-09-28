"""Sweep local ATFM services against a bounded-concurrency fake model worker."""
from __future__ import annotations

import argparse
import asyncio
from importlib.util import find_spec
from pathlib import Path

from .config import LoadConfig
from .runtime import local_stack, provenance, write_json
from .workload import exercise


def run_case(config: LoadConfig, directory: Path) -> dict:
    if config.profile_proxy and find_spec('yappi') is None:
        raise RuntimeError('proxy profiling requires the atfm-experiments[profiling] extra')
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'config.json', config.model_dump())
    write_json(directory / 'environment.json', provenance())
    with local_stack(config, directory) as endpoints:
        result = asyncio.run(exercise(config, endpoints, directory))
    if config.profile_proxy and not (directory / 'proxy.pstats').is_file():
        raise RuntimeError(f'proxy profile was not saved; inspect shutdown and server logs in {directory}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', type=int, nargs='+', default=[16, 64, 256])
    parser.add_argument('--patterns', nargs='+', choices=['staggered', 'burst'], default=['staggered', 'burst'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='JSON overrides for LoadConfig; session/pattern sweep overrides these fields')
    args = parser.parse_args()
    if find_spec('uvicorn') is None:
        parser.error('install serving dependencies with uv sync --extra dev --extra serve')
    base = LoadConfig.model_validate_json(args.config.read_text()) if args.config else LoadConfig()
    cases = [LoadConfig.model_validate(base.model_dump() | {'sessions': size, 'pattern': pattern})
             for pattern in dict.fromkeys(args.patterns) for size in dict.fromkeys(args.sessions)]
    paths = [args.out / f'{c.pattern}-{c.sessions}' for c in cases]
    if any(path.exists() for path in paths):
        parser.error('case directories already exist; choose a fresh --out to preserve evidence')
    results = []
    for config, directory in zip(cases, paths):
        result = run_case(config, directory)
        results.append(result)
        write_json(args.out / 'summary.json', results)
        print(f'{config.pattern:9s} sessions={config.sessions:6d} '
              f'ok={result["requests_ok"]}/{result["maximum_requests"]} '
              f'p95={result["client_duration_s"]["p95"]}s '
              f'queue_peak={result["proxy_queue_peak_observed"]} '
              f'control_errors={result["control_errors"]}', flush=True)
    if any(r['requests_ok'] != r['maximum_requests'] or r['session_errors'] or r['drain_deadline_reached']
           or r['tool_publish_errors'] or r['control_errors'] or r['control_http_errors']
           or r['probe_errors'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
