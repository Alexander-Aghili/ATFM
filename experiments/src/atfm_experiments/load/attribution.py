"""Repeat paired control-on/off trials; alternate order to expose run variability."""
from __future__ import annotations

import argparse
from pathlib import Path

from .__main__ import run_case
from .config import LoadConfig
from .runtime import write_json


def cases(base: LoadConfig, sizes: list[int], repeats: int):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    for size in dict.fromkeys(sizes):
        for repeat in range(repeats):
            for enabled in ((True, False) if repeat % 2 == 0 else (False, True)):
                cfg = LoadConfig.model_validate(base.model_dump() | {
                    'sessions': size, 'seed': base.seed + repeat, 'control_enabled': enabled})
                yield f'{size}-seed-{cfg.seed}-control-{int(enabled)}', cfg


def main():
    args, planned = _planned_cases()
    results = []
    for name, cfg in planned:
        result = run_case(cfg, args.out / name)
        results.append(result)
        write_json(args.out / 'summary.json', results)
        print(f'{name}: {result["requests_ok"]}/{result["maximum_requests"]} '
              f'p95={result["client_duration_s"]["p95"]}', flush=True)
    if any(r['requests_ok'] != r['maximum_requests'] or r['session_errors']
           or r['drain_deadline_reached'] or r['tool_publish_errors'] or r['control_errors']
           or r['control_http_errors'] or r['probe_errors'] for r in results):
        raise SystemExit(1)


def _planned_cases():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--sessions', type=int, nargs='+', default=[256, 1024])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    base = LoadConfig.model_validate_json(args.config.read_text()) if args.config else LoadConfig(pattern='burst')
    try:
        planned = list(cases(base, args.sessions, args.repeats))
    except ValueError as exc:
        parser.error(str(exc))
    if any((args.out / name).exists() for name, _ in planned):
        parser.error('case directories exist; choose a fresh --out')
    return args, planned


if __name__ == '__main__':
    main()
