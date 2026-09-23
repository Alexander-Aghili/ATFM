import argparse

import uvicorn

from atfm.proxy.app import create_app
from atfm.proxy.config import ProxyConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--events", default="runs/proxy/events.jsonl")
    ap.add_argument("--trace", default="runs/proxy/calls.jsonl")
    a = ap.parse_args()
    cfg = ProxyConfig(upstream_url=a.upstream, window=a.window, events_path=a.events, trace_path=a.trace)
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
