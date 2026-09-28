import argparse

import uvicorn

from atfm.proxy.app import create_app
from atfm.proxy.config import ProxyConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--upstream-keepalive", type=int, default=None, help="idle upstream connections (default: max(20, window))")
    ap.add_argument("--events", default="runs/proxy/events.jsonl")
    ap.add_argument("--trace", default="runs/proxy/calls.jsonl")
    ap.add_argument("--board", default=None, help="board service URL for per-request predictions")
    ap.add_argument("--max-queue", type=int, default=None)
    ap.add_argument("--prediction-limit", type=int, default=4, help="maximum unfinished prediction jobs")
    ap.add_argument("--board-timeout", type=float, default=.05, help="total prediction wait budget in seconds")
    a = ap.parse_args()
    cfg = ProxyConfig(upstream_url=a.upstream, window=a.window, events_path=a.events, trace_path=a.trace,
                      board_url=a.board, max_queue_size=a.max_queue, upstream_keepalive_connections=a.upstream_keepalive,
                      prediction_limit=a.prediction_limit, board_timeout_s=a.board_timeout)
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
