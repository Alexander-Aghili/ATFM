#!/usr/bin/env bash
# Deploy path end to end on the laptop: Mocker -> proxy (asking the board) -> scripted sessions through the
# sidecar -> board service with controllers and a metrics scrape -> control loop pushing holds and touches.
# A functional check, not an experiment: it proves the wiring, it measures nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/control
EV=runs/control/events.jsonl; : > "$EV"
uv run python scripts/dynamo_local.py up
uv run python scripts/run_board.py --events "$EV" --snapshots runs/control/snapshots.jsonl --train data/tracelab/tracelab.parquet \
    --serve 8081 --control experiments/control_local.yaml &
BOARD=$!
sleep 4
uv run python scripts/run_proxy.py --upstream http://127.0.0.1:8000 --port 8799 --window 4 --events "$EV" \
    --trace runs/control/calls.jsonl --board http://127.0.0.1:8081 &
PROXY=$!
trap 'kill $PROXY $BOARD 2>/dev/null || true; uv run python scripts/dynamo_local.py down' EXIT
sleep 2
uv run python scripts/run_control.py --board http://127.0.0.1:8081 --proxy http://127.0.0.1:8799 --interval 2 --steps 30 --log runs/control/control.jsonl &
CONTROL=$!
sed 's#^proxy_url: null#proxy_url: http://127.0.0.1:8799#' experiments/l1_jobs.yaml > runs/control/l1_jobs_proxy.yaml
uv run python scripts/collect_traces.py runs/control/l1_jobs_proxy.yaml
wait $CONTROL || true
echo "--- control loop log (last 3):"; tail -3 runs/control/control.jsonl
echo "--- proxy state:"; curl -s http://127.0.0.1:8799/state; echo
echo "--- board snapshot t:"; curl -s http://127.0.0.1:8081/snapshot | head -c 200; echo
