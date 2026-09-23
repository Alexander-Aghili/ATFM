#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
uv run python scripts/dynamo_local.py up
uv run python scripts/run_proxy.py --upstream http://127.0.0.1:8000 --port 8799 --window 4 &
PROXY=$!
trap 'kill $PROXY 2>/dev/null || true; uv run python scripts/dynamo_local.py down' EXIT
sleep 2
sed 's#^proxy_url: null#proxy_url: http://127.0.0.1:8799#' experiments/l1_jobs.yaml > runs/l1_jobs_proxy.yaml
uv run python scripts/collect_traces.py runs/l1_jobs_proxy.yaml
uv run python scripts/run_board.py --events runs/collect/l1_events.jsonl --snapshots runs/collect/snapshots.jsonl --train data/tracelab/tracelab.parquet --once
